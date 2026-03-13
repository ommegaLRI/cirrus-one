"""
deid.fusion.stage_fusion
------------------------

Stage E — Fusion + Particle Validation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from deid.config.models import DEIDConfig
from deid.core.logging import log_info
from deid.storage.io import (
    read_json,
    read_parquet,
    write_json,
    write_parquet,
    wrap_artifact,
)
from deid.storage.paths import inputs_dir, intermediate_dir

from deid.fusion.matcher import match_events
from deid.fusion.validation import compute_discrepancies, build_validation_report
from deid.fusion.tables import build_matched_events_df


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _load_wrapped(path: Path) -> Dict[str, Any]:
    w = read_json(path)
    return w.get("payload", {})


# -------------------------------------------------------------------
# Stage Entry
# -------------------------------------------------------------------


def fusion_stage(
    run_dir: Path,
    inputs: Dict[str, Any],
    config: DEIDConfig,
    context: Dict[str, Any],
) -> None:

    log_info("fusion_stage_start", run_dir=str(run_dir))

    config_hash = inputs["config_hash"]
    input_hashes = inputs.get("input_hashes", {})

    inp_dir = inputs_dir(run_dir)
    out_dir = intermediate_dir(run_dir)

    particle_path = inp_dir / "particle.parquet"
    event_path = out_dir / "event_catalog.parquet"
    alignment_path = out_dir / "alignment.json"

    if not particle_path.exists():
        log_info("fusion_skipped_no_particle_table")
        return

    if not event_path.exists():
        log_info("fusion_skipped_no_event_catalog")
        return

    # ---------------------------------------------------------------
    # Load artifacts
    # ---------------------------------------------------------------

    particle_df = read_parquet(particle_path)
    event_df = read_parquet(event_path)
    alignment_payload = _load_wrapped(alignment_path)

    # ---------------------------------------------------------------
    # Normalize timestamps (no mutation of particle time)
    # ---------------------------------------------------------------

    if "t_utc" in particle_df.columns:
        particle_df["t_utc"] = pd.to_datetime(particle_df["t_utc"], utc=True)

    if "t_peak_utc" in event_df.columns:
        event_df["t_peak_utc"] = pd.to_datetime(event_df["t_peak_utc"], utc=True)

    # ---------------------------------------------------------------
    # Fusion configuration
    # ---------------------------------------------------------------

    try:
        fcfg = dict(config.fusion)
    except Exception:
        fcfg = {}

    tolerance_seconds = float(fcfg.get("time_tolerance_s", 8.0))
    tolerance_px = float(fcfg.get("spatial_tolerance_px", 10.0))

    log_info(
        "fusion_matching_config",
        time_tolerance_s=tolerance_seconds,
        spatial_tolerance_px=tolerance_px,
    )

    # ---------------------------------------------------------------
    # Match events
    # ---------------------------------------------------------------

    matches = match_events(
        event_df=event_df,
        particle_df=particle_df,
        alignment_payload=alignment_payload,
        tolerance_seconds=tolerance_seconds,
        tolerance_px=tolerance_px,
    )

    # ---------------------------------------------------------------
    # Compute discrepancies
    # ---------------------------------------------------------------

    rows = compute_discrepancies(matches, event_df, particle_df)
    matched_df = build_matched_events_df(rows)

    write_parquet(out_dir / "matched_events.parquet", matched_df)

    # ---------------------------------------------------------------
    # Validation report
    # ---------------------------------------------------------------

    report = build_validation_report(rows)

    wrapped = wrap_artifact(
        payload=report,
        schema_version="particle_validation_v1",
        config_hash=config_hash,
        input_hashes=input_hashes,
        artifact_type="particle_validation",
    )

    write_json(out_dir / "particle_validation.json", wrapped)

    log_info(
        "fusion_stage_complete",
        particle_rows=len(particle_df),
        event_rows=len(event_df),
        n_matches=len(rows),
    )