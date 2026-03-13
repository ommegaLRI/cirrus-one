"""
deid.cli.commands_analyze
-------------------------

Implements:
    deid analyze

Phase 6 scope:
- Wire ingest stage into runner
- Produce run bundle with inputs + manifest
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import typer
import yaml

from deid.config.models import DEIDConfig
from deid.config.hashing import compute_config_hash
from deid.core.logging import log_info
from deid.core.versioning import SCHEMA_PARTICLE, SCHEMA_PROCESSED
from deid.ingest.thermal_reader_hdf5 import (
    read_thermal_cube_ref,
    thermal_cube_ref_to_dict,
)
from deid.ingest.particle_reader import read_particle_table_normalized
from deid.ingest.processed_reader import read_processed_series_normalized
from deid.ingest.manifest import (
    build_input_manifest,
    extract_input_hashes,
)
from deid.runner import run_pipeline
from deid.storage.io import write_json, write_parquet
from deid.storage.paths import inputs_dir

from deid.alignment.stage_alignment import alignment_stage
from deid.plate_state.stage_plate_state import plate_state_stage
from deid.events.stage_event_extract import event_extract_stage
from deid.fusion.stage_fusion import fusion_stage
from deid.swe.stage_swe_closure import swe_closure_stage
from deid.inference.stage_inference import inference_stage


# -------------------------------------------------------------------
# Config loading
# -------------------------------------------------------------------


def _load_config(config_path: Optional[Path]) -> DEIDConfig:
    if config_path is None:
        return DEIDConfig()

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return DEIDConfig(**data)


# -------------------------------------------------------------------
# Stage function: ingest
# -------------------------------------------------------------------


def ingest_stage(run_dir: Path, inputs: Dict[str, Any], config: DEIDConfig, context: Dict[str, Any]) -> None:
    """
    Stage A — Ingest + Normalize + Manifest

    Writes:
        inputs/thermal_ref.json
        inputs/particle.parquet
        inputs/processed.parquet
        inputs/manifest.json
    """

    particle_path: Optional[Path] = inputs.get("particle")
    processed_path: Optional[Path] = inputs.get("processed")
    hdf5_path: Path = inputs["hdf5"]
    session_id: str = inputs["session_id"]

    config_hash: str = inputs["config_hash"]

    inp_dir = inputs_dir(run_dir)
    inp_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Thermal cube metadata
    # ---------------------------------------------------------------

    thermal_ref = read_thermal_cube_ref(
        hdf5_path,
        dataset_path=config.ingest.hdf5_dataset_path,
    )

    write_json(inp_dir / "thermal_ref.json", thermal_cube_ref_to_dict(thermal_ref))

    # ---------------------------------------------------------------
    # Particle table normalization
    # ---------------------------------------------------------------

    particle_valid = None
    if particle_path:
        log_info("ingest_particle_start", path=str(particle_path))

        valid_df, invalid_df = read_particle_table_normalized(
            particle_path,
            timezone_str=config.ingest.timezone,
        )

        if not valid_df.empty:
            write_parquet(inp_dir / "particle.parquet", valid_df)

        if not invalid_df.empty:
            write_parquet(inp_dir / "particle_invalid.parquet", invalid_df)

        particle_valid = valid_df

    # ---------------------------------------------------------------
    # Processed table normalization
    # ---------------------------------------------------------------

    processed_valid = None
    if processed_path:
        log_info("ingest_processed_start", path=str(processed_path))

        valid_df, invalid_df = read_processed_series_normalized(
            processed_path,
            timezone_str=config.ingest.timezone,
        )

        if not valid_df.empty:
            write_parquet(inp_dir / "processed.parquet", valid_df)

        if not invalid_df.empty:
            write_parquet(inp_dir / "processed_invalid.parquet", invalid_df)

        processed_valid = valid_df

    # ---------------------------------------------------------------
    # Manifest
    # ---------------------------------------------------------------

    manifest = build_input_manifest(
        session_id=session_id,
        config_hash=config_hash,
        thermal_hdf5_path=hdf5_path,
        thermal_ref=thermal_ref,
        particle_path=particle_path,
        processed_path=processed_path,
    )

    write_json(inp_dir / "manifest.json", manifest)


# -------------------------------------------------------------------
# CLI command
# -------------------------------------------------------------------


def analyze_command(
    particle: Optional[Path] = typer.Option(None, help="Particle CSV/XLSX"),
    processed: Optional[Path] = typer.Option(None, help="Processed CSV/XLSX"),
    hdf5: Path = typer.Option(..., help="Thermal cube HDF5"),
    session_id: str = typer.Option(..., help="Session identifier"),
    config: Optional[Path] = typer.Option(None, help="Config YAML"),
) -> None:
    """
    Run DEID analysis pipeline.
    """

    cfg = _load_config(config)
    config_hash = compute_config_hash(cfg)

    input_hashes: Dict[str, str] = {
        "thermal_hdf5": str(hdf5),
        "particle": str(particle) if particle else "",
        "processed": str(processed) if processed else "",
    }

    stage_fns = {
        "ingest": ingest_stage,
        "alignment": alignment_stage,
        "plate_state": plate_state_stage,
        "event_extract": event_extract_stage,
        "fusion": fusion_stage,
        "swe_closure": swe_closure_stage,
        "inference": inference_stage,
    }

    inputs = {
        "session_id": session_id,
        "particle": particle,
        "processed": processed,
        "hdf5": hdf5,
        "config_hash": config_hash,
    }

    job, run_dir = run_pipeline(
        session_id=session_id,
        config=cfg,
        run_root_dir=cfg.storage.run_root,
        input_hashes=input_hashes,
        inputs=inputs,
        stage_fns=stage_fns,
    )

    typer.echo("")
    typer.echo(f"Run directory: {run_dir}")
    typer.echo(f"Job state: {job.state}")