"""
deid.alignment.stage_alignment
------------------------------

Runner stage implementation:

Stage B — Alignment + Integrity
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from deid.alignment.aligner import build_session_alignment
from deid.config.models import DEIDConfig
from deid.core.logging import log_info
from deid.core.types import ThermalCubeRef
from deid.core.versioning import SCHEMA_ALIGNMENT
from deid.storage.io import read_json, read_parquet, write_json, wrap_artifact
from deid.storage.paths import inputs_dir, intermediate_dir


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _load_thermal_ref(run_dir: Path) -> ThermalCubeRef:
    ref_path = inputs_dir(run_dir) / "thermal_ref.json"
    data = read_json(ref_path)
    payload = data.get("payload", data)
    return ThermalCubeRef(**payload)


def _maybe_load_parquet(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return read_parquet(path)


def _config_to_dict(obj: Any) -> Dict[str, Any]:
    """
    Safely normalize a config subsection into a plain dict.

    Supports:
    - dict
    - pydantic v2 models via model_dump()
    - generic objects via __dict__
    """
    if obj is None:
        return {}

    if isinstance(obj, dict):
        return dict(obj)

    if hasattr(obj, "model_dump"):
        try:
            return dict(obj.model_dump())
        except Exception:
            pass

    if hasattr(obj, "dict"):
        try:
            return dict(obj.dict())
        except Exception:
            pass

    if hasattr(obj, "__dict__"):
        try:
            return {
                k: v
                for k, v in vars(obj).items()
                if not k.startswith("_")
            }
        except Exception:
            pass

    return {}


# -------------------------------------------------------------------
# Stage Entry
# -------------------------------------------------------------------


def alignment_stage(
    run_dir: Path,
    inputs: Dict[str, Any],
    config: DEIDConfig,
    context: Dict[str, Any],
) -> None:
    log_info("alignment_stage_start", run_dir=str(run_dir))

    config_hash: str = inputs["config_hash"]
    input_hashes: Dict[str, str] = inputs.get("input_hashes", {})

    inp_dir = inputs_dir(run_dir)
    out_dir = intermediate_dir(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Load ingest artifacts
    # ---------------------------------------------------------------

    thermal_ref = _load_thermal_ref(run_dir)

    particle_df = _maybe_load_parquet(inp_dir / "particle.parquet")
    processed_df = _maybe_load_parquet(inp_dir / "processed.parquet")

    # ---------------------------------------------------------------
    # Alignment configuration
    # ---------------------------------------------------------------

    align_cfg: Dict[str, Any] = _config_to_dict(getattr(config, "alignment", None))

    fallback_dt_seconds = float(align_cfg.get("fallback_dt_seconds", 1.0))
    particle_time_offset_seconds = float(
        align_cfg.get("particle_time_offset_seconds", 0.0)
    )

    log_info(
        "alignment_config_loaded",
        fallback_dt_seconds=fallback_dt_seconds,
        particle_time_offset_seconds=particle_time_offset_seconds,
    )

    # ---------------------------------------------------------------
    # IMPORTANT CONTRACT (Model A)
    #
    # Raw particle timestamps are authoritative.
    # NO mutation occurs here.
    #
    # The declared particle offset is applied only inside the aligner
    # during particle→frame mapping.
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # Compute alignment + integrity
    # ---------------------------------------------------------------

    alignment_dict, integrity_dict = build_session_alignment(
        thermal_ref=thermal_ref,
        particle_table=particle_df,
        processed_series=processed_df,
        config=align_cfg,
    )

    # ===============================================================
    # ENFORCE FRAME TIMEBASE INVARIANT
    # ===============================================================

    ftb = dict(alignment_dict.get("frame_timebase", {}) or {})

    dt = ftb.get("dt_seconds")
    dt_missing = dt is None or (isinstance(dt, float) and np.isnan(dt))

    if dt_missing:
        dt = fallback_dt_seconds
        ftb["dt_seconds"] = dt
        ftb["source"] = ftb.get("source") or "thermal_inferred"
        ftb["confidence"] = float(min(ftb.get("confidence", 0.1), 0.1))
    else:
        dt = float(dt)

    frame_ts = ftb.get("frame_timestamps_utc")

    # Only build timestamps if a real anchor exists
    if frame_ts is None and ftb.get("t0_utc") is not None:
        T = int(thermal_ref.shape[0])
        t0 = datetime.fromisoformat(ftb["t0_utc"])

        ftb["frame_timestamps_utc"] = [
            (t0 + timedelta(seconds=i * dt)).isoformat()
            for i in range(T)
        ]

    alignment_dict["frame_timebase"] = ftb

    # ---------------------------------------------------------------
    # Debug proof that invariant is satisfied
    # ---------------------------------------------------------------

    log_info(
        "alignment_timebase_final",
        dt_seconds=ftb.get("dt_seconds"),
        n_frame_ts=len(ftb.get("frame_timestamps_utc") or []),
    )

    offsets = alignment_dict.get("offsets_seconds", {}) or {}

    log_info(
        "alignment_offsets_final",
        particle_vs_thermal=offsets.get("particle_vs_thermal"),
        processed_vs_thermal=offsets.get("processed_vs_thermal"),
    )

    # ---------------------------------------------------------------
    # Write artifacts (wrapped)
    # ---------------------------------------------------------------

    alignment_wrapped = wrap_artifact(
        payload=alignment_dict,
        schema_version=SCHEMA_ALIGNMENT,
        config_hash=config_hash,
        input_hashes=input_hashes,
        artifact_type="alignment",
    )

    integrity_wrapped = wrap_artifact(
        payload=integrity_dict,
        schema_version="integrity_v1",
        config_hash=config_hash,
        input_hashes=input_hashes,
        artifact_type="integrity",
    )

    write_json(out_dir / "alignment.json", alignment_wrapped)
    write_json(out_dir / "integrity.json", integrity_wrapped)

    log_info(
        "alignment_stage_complete",
        n_particle_mapped=len(alignment_dict.get("particle_to_frame", {})),
        n_processed_mapped=len(alignment_dict.get("processed_to_frame", {})),
        confidence=alignment_dict.get("confidence"),
    )