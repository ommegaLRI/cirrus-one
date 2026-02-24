"""
Runner adapter for Stage A — Ingest.

This is intentionally minimal for bring-up:
- build manifest
- write thermal_ref.json
- no particle/processed required yet
"""

from pathlib import Path
from typing import Any, Dict
import pandas as pd

from deid.ingest.manifest import build_input_manifest
from deid.ingest.thermal_reader_hdf5 import read_thermal_cube_ref
from deid.storage.io import write_json
from deid.storage.paths import inputs_dir
from deid.storage.io import write_parquet


def stage_ingest(run_dir: Path, inputs: Dict[str, Any], config, context) -> None:
    in_dir = inputs_dir(run_dir)
    in_dir.mkdir(parents=True, exist_ok=True)

    thermal_path = Path(inputs["thermal_hdf5"])

    # Build thermal reference
    ref = read_thermal_cube_ref(
        thermal_path,
        dataset_path=config.ingest.hdf5_dataset_path,
        sample_read=True,
    )

    # Build manifest
    manifest = build_input_manifest(
        session_id=context.get("session_id", "unknown"),
        config_hash=context.get("config_hash", "unknown"),
        thermal_hdf5_path=thermal_path,
        thermal_ref=ref,
    )

    write_json(in_dir / "manifest.json", manifest)

    # Also store thermal_ref payload
    write_json(in_dir / "thermal_ref.json", ref.__dict__)

    # ---------------------------------------------------
    # Ensure required parquet artifacts exist (bring-up mode)
    # ---------------------------------------------------

    particle_path = in_dir / "particle.parquet"
    processed_path = in_dir / "processed.parquet"

    if not particle_path.exists():
        write_parquet(particle_path, pd.DataFrame())

    if not processed_path.exists():
        write_parquet(processed_path, pd.DataFrame())