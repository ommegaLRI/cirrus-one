"""
deid.alignment.stage_alignment
------------------------------

Runner stage implementation:

Stage B — Alignment + Integrity

Inputs (from ingest stage):
    inputs/thermal_ref.json
    inputs/particle.parquet   (optional)
    inputs/processed.parquet  (optional)

Outputs:
    intermediate/alignment.json
    intermediate/integrity.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from deid.alignment.aligner import build_session_alignment
from deid.config.models import DEIDConfig
from deid.core.errors import AlignmentError
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
    # Stored as dict during ingest
    return ThermalCubeRef(**data)


def _maybe_load_parquet(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return read_parquet(path)


# -------------------------------------------------------------------
# Stage Entry
# -------------------------------------------------------------------


def alignment_stage(
    run_dir: Path,
    inputs: Dict[str, Any],
    config: DEIDConfig,
    context: Dict[str, Any],
) -> None:
    """
    Execute alignment stage.

    Parameters
    ----------
    run_dir : Path
    inputs : dict
        Runner input dict (contains config_hash + input_hashes).
    config : DEIDConfig
    context : dict
        pipeline metadata
    """

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

    align_cfg: Dict[str, Any] = {}
    try:
        align_cfg = dict(config.alignment)
    except Exception:
        align_cfg = {}

    # ---------------------------------------------------------------
    # Compute alignment + integrity
    # ---------------------------------------------------------------

    alignment_dict, integrity_dict = build_session_alignment(
        thermal_ref=thermal_ref,
        particle_table=particle_df,
        processed_series=processed_df,
        config=align_cfg,
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