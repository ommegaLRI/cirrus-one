"""
deid.runner.execute
-------------------

Sequential pipeline runner.

Stages are executed in topological order with idempotent skipping.

This runner does not implement parallelism yet; stage boundaries are
designed so Ray/Celery can be added later.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import yaml

from deid.config.models import DEIDConfig
from deid.core.errors import DEIDError, ArtifactIOError
from deid.core.logging import log_info, log_warning, log_error
from deid.core.versioning import PIPELINE_VERSION
from deid.storage.io import write_json, wrap_artifact
from deid.storage.paths import provenance_dir
from deid.runner.dag import get_stage_defs, topo_order
from deid.runner.job import Job, JobState, StageState
from deid.runner.storage import stage_is_done, write_stage_done_marker


StageCallable = Callable[[Path, Dict[str, Any], DEIDConfig, Dict[str, Any]], None]


def _write_error_blob(run_dir: Path, stage_id: str, tb: str) -> None:
    p = provenance_dir(run_dir) / "errors"
    p.mkdir(parents=True, exist_ok=True)
    out = p / f"{stage_id}.txt"
    out.write_text(tb, encoding="utf-8")


def _write_timings(
    run_dir: Path,
    timings: Dict[str, float],
    config_hash: str,
    input_hashes: Dict[str, str],
) -> None:
    prov = provenance_dir(run_dir)
    prov.mkdir(parents=True, exist_ok=True)

    wrapped = wrap_artifact(
        payload={"stage_timings_s": dict(timings)},
        schema_version="timings_v1",
        config_hash=config_hash,
        input_hashes=input_hashes,
        artifact_type="timings",
        pipeline_version=PIPELINE_VERSION,
    )
    write_json(prov / "timings.json", wrapped)


def _write_config_provenance(run_dir: Path, config: DEIDConfig, config_hash: str) -> None:
    """
    Persist the EXACT config used for this run.
    This is critical both for provenance AND to debug config propagation.
    """
    prov = provenance_dir(run_dir)
    prov.mkdir(parents=True, exist_ok=True)

    # Write canonical config.yaml
    cfg_path = prov / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )

    # Write config hash
    (prov / "config_hash.txt").write_text(config_hash, encoding="utf-8")


def run_sequential(
    *,
    job: Job,
    run_dir: Path,
    inputs: Dict[str, Any],
    config: DEIDConfig,
    config_hash: str,
    input_hashes: Dict[str, str],
    stage_fns: Dict[str, StageCallable],
    stop_after_stage: Optional[str] = None,
) -> Job:
    """
    Run pipeline sequentially.
    """
    stage_defs = get_stage_defs()
    order = topo_order(stage_defs)

    # WRITE CONFIG PROVENANCE AT START
    _write_config_provenance(run_dir, config, config_hash)

    job.mark_started()
    log_info(
        "job_started",
        job_id=job.job_id,
        session_id=job.session_id,
        run_id=job.run_id,
        run_dir=str(run_dir),
    )

    timings: Dict[str, float] = {}

    for stage_id in order:
        sd = stage_defs[stage_id]
        job.stage_status.setdefault(stage_id, StageState.NOT_STARTED)

        if stop_after_stage is not None and stage_id not in order[: order.index(stop_after_stage) + 1]:
            continue

        fn = stage_fns.get(stage_id)
        if fn is None:
            log_warning("stage_fn_missing_skipping", stage_id=stage_id)
            job.stage_status[stage_id] = StageState.SKIPPED
            continue

        if stage_is_done(
            run_dir,
            stage_id,
            expected_outputs=sd.outputs,
            config_hash=config_hash,
            input_hashes=input_hashes,
        ):
            log_info("stage_skipped_already_done", stage_id=stage_id)
            job.stage_status[stage_id] = StageState.SKIPPED
            continue

        log_info("stage_started", stage_id=stage_id)
        job.stage_status[stage_id] = StageState.RUNNING
        t0 = time.time()

        try:
            # PASS CONFIG INTO EVERY STAGE
            context: Dict[str, Any] = {
                "pipeline_version": PIPELINE_VERSION,
                "stage_id": stage_id,
                "config_hash": config_hash,
            }

            fn(run_dir, inputs, config, context)

            # Verify expected outputs exist
            missing = [rel for rel in sd.outputs if not (Path(run_dir) / rel).exists()]
            if missing:
                raise ArtifactIOError(
                    "Stage did not produce expected outputs",
                    details={"stage_id": stage_id, "missing_outputs": missing},
                )

            write_stage_done_marker(
                run_dir,
                stage_id,
                schema_version="stage_done_v1",
                config_hash=config_hash,
                input_hashes=input_hashes,
                outputs=sd.outputs,
            )

            dt = time.time() - t0
            timings[stage_id] = float(dt)
            job.stage_timings_s[stage_id] = float(dt)
            job.stage_status[stage_id] = StageState.DONE
            log_info("stage_done", stage_id=stage_id, elapsed_s=dt)

        except Exception as e:
            dt = time.time() - t0
            timings[stage_id] = float(dt)
            job.stage_timings_s[stage_id] = float(dt)
            job.stage_status[stage_id] = StageState.FAILED

            tb = traceback.format_exc()
            _write_error_blob(run_dir, stage_id, tb)

            if isinstance(e, DEIDError):
                err_dict = e.to_dict()
            else:
                err_dict = {
                    "type": e.__class__.__name__,
                    "code": "UNHANDLED",
                    "message": str(e),
                    "details": {},
                }

            job.errors[stage_id] = err_dict
            log_error("stage_failed", stage_id=stage_id, error=err_dict, elapsed_s=dt)

            if config.runner.fail_fast:
                job.mark_finished(succeeded=False)
                _write_timings(run_dir, timings, config_hash, input_hashes)
                return job

    job.mark_finished(succeeded=(job.state != JobState.FAILED))
    _write_timings(run_dir, timings, config_hash, input_hashes)
    log_info("job_finished", job_id=job.job_id, state=job.state)
    return job