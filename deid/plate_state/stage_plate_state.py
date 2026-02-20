"""
deid.plate_state.stage_plate_state
----------------------------------

Runner stage implementation:

Stage C — Plate / Field State + Instrument QC

Inputs:
    inputs/thermal_ref.json
    intermediate/alignment.json   (for timebase context)

Outputs:
    intermediate/plate_state.npz
    intermediate/plate_state.json
    intermediate/instrument_health.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from deid.config.models import DEIDConfig
from deid.core.logging import log_info
from deid.core.types import ThermalCubeRef
from deid.core.versioning import SCHEMA_PLATE_STATE
from deid.storage.io import (
    read_json,
    write_json,
    write_npz,
    wrap_artifact,
)
from deid.storage.paths import inputs_dir, intermediate_dir

from deid.plate_state.quiescent import select_quiescent_frames
from deid.plate_state.estimator import estimate_plate_state
from deid.plate_state.changepoints import changepoints_to_dict, detect_baseline_changepoints
from deid.plate_state.health import compute_instrument_health


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _load_thermal_ref(run_dir: Path) -> ThermalCubeRef:
    ref_path = inputs_dir(run_dir) / "thermal_ref.json"
    data = read_json(ref_path)
    return ThermalCubeRef(**data)


def _load_alignment_payload(run_dir: Path) -> Dict[str, Any]:
    """
    alignment.json is wrapped via wrap_artifact(), so we extract payload.
    """
    path = intermediate_dir(run_dir) / "alignment.json"
    wrapped = read_json(path)
    return wrapped.get("payload", {})


# -------------------------------------------------------------------
# Stage Entry
# -------------------------------------------------------------------


def plate_state_stage(
    run_dir: Path,
    inputs: Dict[str, Any],
    config: DEIDConfig,
    context: Dict[str, Any],
) -> None:
    """
    Execute Plate State estimation stage.
    """

    log_info("plate_state_stage_start", run_dir=str(run_dir))

    config_hash: str = inputs["config_hash"]
    input_hashes: Dict[str, str] = inputs.get("input_hashes", {})

    out_dir = intermediate_dir(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Load upstream artifacts
    # ---------------------------------------------------------------

    thermal_ref = _load_thermal_ref(run_dir)
    alignment_payload = _load_alignment_payload(run_dir)

    # ---------------------------------------------------------------
    # PlateState config
    # ---------------------------------------------------------------

    try:
        ps_cfg = dict(config.plate_state)
    except Exception:
        ps_cfg = {}

    quiescent_fraction = float(ps_cfg.get("quiescent_fraction", 0.20))
    min_frames = int(ps_cfg.get("min_quiescent_frames", 30))
    dead_noise_eps = float(ps_cfg.get("dead_noise_eps", 1e-6))

    # ---------------------------------------------------------------
    # Select quiescent frames
    # ---------------------------------------------------------------

    qsel = select_quiescent_frames(
        thermal_ref,
        quiescent_fraction=quiescent_fraction,
        min_frames=min_frames,
    )

    # ---------------------------------------------------------------
    # Estimate Plate State arrays
    # ---------------------------------------------------------------

    plate = estimate_plate_state(
        thermal_ref,
        qsel,
        dead_noise_eps=dead_noise_eps,
    )

    # ---------------------------------------------------------------
    # Change points (baseline)
    # ---------------------------------------------------------------

    cps = detect_baseline_changepoints(plate.baseline_B_t)
    cps_dict = changepoints_to_dict(cps)

    # ---------------------------------------------------------------
    # Instrument health report
    # ---------------------------------------------------------------

    health = compute_instrument_health(
        plate,
        activity=qsel.activity,
    )

    # ---------------------------------------------------------------
    # Write NPZ arrays
    # ---------------------------------------------------------------

    write_npz(
        out_dir / "plate_state.npz",
        baseline_B_t=plate.baseline_B_t.astype(np.float64),
        nonuniformity_G_yx=plate.nonuniformity_G_yx.astype(np.float32),
        noise_N_yx=plate.noise_N_yx.astype(np.float32),
        dead_pixel_mask_yx=plate.dead_pixel_mask_yx.astype(bool),
    )

    # ---------------------------------------------------------------
    # plate_state.json (metadata + methods)
    # ---------------------------------------------------------------

    plate_state_payload = {
        "method": plate.method,
        "params": plate.params,
        "quiescent_indices": plate.quiescent_indices,
        "change_points": cps_dict,
        "alignment_timebase_source": alignment_payload.get("frame_timebase", {}).get("source"),
    }

    plate_state_wrapped = wrap_artifact(
        payload=plate_state_payload,
        schema_version=SCHEMA_PLATE_STATE,
        config_hash=config_hash,
        input_hashes=input_hashes,
        artifact_type="plate_state",
    )

    write_json(out_dir / "plate_state.json", plate_state_wrapped)

    # ---------------------------------------------------------------
    # instrument_health.json
    # ---------------------------------------------------------------

    health_payload = {
        "overall_score": health.overall_score,
        "qc_metrics": health.qc_metrics,
        "flags": health.flags,
        "change_points": health.change_points,
    }

    health_wrapped = wrap_artifact(
        payload=health_payload,
        schema_version="instrument_health_v1",
        config_hash=config_hash,
        input_hashes=input_hashes,
        artifact_type="instrument_health",
    )

    write_json(out_dir / "instrument_health.json", health_wrapped)

    log_info(
        "plate_state_stage_complete",
        quiescent_frames=len(qsel.quiescent_indices),
        dead_pixel_fraction=health.qc_metrics.get("dead_pixel_fraction"),
        health_score=health.overall_score,
    )