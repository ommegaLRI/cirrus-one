"""
deid.swe.stage_swe_closure
---------------------------

Stage F — SWE Reconstruction + Closure Engine
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from deid.config.models import DEIDConfig
from deid.storage.io import read_json, read_parquet, write_parquet, write_json, wrap_artifact
from deid.storage.paths import intermediate_dir, inputs_dir
from deid.core.logging import log_info

from deid.swe.calibrators.energy_linear import EnergyLinearCalibrator
from deid.swe.reconstruct import reconstruct_swe_series
from deid.swe.rate import windowed_rate, robust_rate, event_based_rate
from deid.swe.closure import compute_closure


def _load_wrapped(path: Path):
    return read_json(path).get("payload", {})


def swe_closure_stage(run_dir: Path, inputs: Dict[str, Any], config: DEIDConfig, context: Dict[str, Any]):

    log_info("swe_closure_stage_start")

    config_hash = inputs["config_hash"]
    input_hashes = inputs.get("input_hashes", {})

    inp_dir = inputs_dir(run_dir)
    out_dir = intermediate_dir(run_dir)

    event_df = read_parquet(out_dir / "event_catalog.parquet")
    alignment = _load_wrapped(out_dir / "alignment.json")

    processed_df = None
    ppath = inp_dir / "processed.parquet"
    if ppath.exists():
        processed_df = read_parquet(ppath)

    try:
        swe_cfg = dict(config.swe)
    except Exception:
        swe_cfg = {}

    sensing_area_mm2 = float(swe_cfg.get("sensing_area_mm2", 1.0))

    # Calibrate mass
    calib = EnergyLinearCalibrator()
    mass, uncert = calib.predict_mass(event_df)
    event_df["mass_mg_authoritative"] = mass

    # Reconstruct SWE
    swe_df = reconstruct_swe_series(
        event_df=event_df,
        frame_timebase=alignment.get("frame_timebase", {}),
        sensing_area_mm2=sensing_area_mm2,
    )

    dt_seconds = alignment.get("frame_timebase", {}).get("dt_seconds", 1.0)

    swe_df["rate_windowed_mmhr"] = windowed_rate(
        swe_df["swe_reconstructed_mm"], dt_seconds
    )
    swe_df["rate_robust_mmhr"] = robust_rate(
        swe_df["swe_reconstructed_mm"], dt_seconds
    )
    swe_df["rate_event_based_mmhr"] = event_based_rate(
        event_df, len(swe_df), dt_seconds
    )

    # Attach processed SWE if available
    if processed_df is not None:
        swe_df = swe_df.merge(processed_df[["t_utc", "swe_mm"]], on="t_utc", how="left")
        swe_df["swe_residual_mm"] = swe_df["swe_reconstructed_mm"] - swe_df["swe_mm"]

    write_parquet(out_dir / "swe_products.parquet", swe_df)

    closure = compute_closure(swe_df, processed_df)

    wrapped = wrap_artifact(
        payload=closure,
        schema_version="closure_report_v1",
        config_hash=config_hash,
        input_hashes=input_hashes,
        artifact_type="closure_report",
    )

    write_json(out_dir / "closure_report.json", wrapped)

    log_info("swe_closure_stage_complete", closure_score=closure.get("closure_score"))