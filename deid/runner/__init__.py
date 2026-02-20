"""
deid.runner
-----------

Public runner entrypoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from deid.config.hashing import compute_config_hash
from deid.config.models import DEIDConfig
from deid.core.ids import stable_id
from deid.runner.execute import run_sequential, StageCallable
from deid.runner.job import Job
from deid.storage.paths import run_root


def compute_run_id(session_id: str, *, config_hash: str, input_hashes: Dict[str, str]) -> str:
    """
    Stable run id derived from session_id + config_hash + input hashes.
    """
    return stable_id(
        "run",
        {
            "session_id": session_id,
            "config_hash": config_hash,
            "input_hashes": dict(sorted(input_hashes.items())),
        },
        length=16,
    )


def run_pipeline(
    *,
    session_id: str,
    config: DEIDConfig,
    run_root_dir: Path,
    input_hashes: Dict[str, str],
    inputs: Dict[str, Any],
    stage_fns: Dict[str, StageCallable],
    stop_after_stage: Optional[str] = None,
) -> tuple[Job, Path]:
    config_hash = compute_config_hash(config)
    run_id = compute_run_id(session_id, config_hash=config_hash, input_hashes=input_hashes)
    run_dir = run_root(run_root_dir, session_id, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    job = Job(
        job_id=stable_id("job", {"session_id": session_id, "run_id": run_id}, length=16),
        session_id=session_id,
        run_id=run_id,
    )

    job = run_sequential(
        job=job,
        run_dir=run_dir,
        inputs=inputs,
        config=config,
        config_hash=config_hash,
        input_hashes=input_hashes,
        stage_fns=stage_fns,
        stop_after_stage=stop_after_stage,
    )
    return job, run_dir