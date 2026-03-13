"""
deid.api.pipeline_adapter
-------------------------

Adapter that allows the API to launch the same pipeline used by the CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from deid.config.hashing import compute_config_hash
from deid.config.models import DEIDConfig

from deid.runner import run_pipeline

from deid.cli.commands_analyze import ingest_stage

from deid.alignment.stage_alignment import alignment_stage
from deid.plate_state.stage_plate_state import plate_state_stage
from deid.events.stage_event_extract import event_extract_stage
from deid.fusion.stage_fusion import fusion_stage
from deid.swe.stage_swe_closure import swe_closure_stage
from deid.inference.stage_inference import inference_stage


def build_stage_registry():
    return {
        "ingest": ingest_stage,
        "alignment": alignment_stage,
        "plate_state": plate_state_stage,
        "event_extract": event_extract_stage,
        "fusion": fusion_stage,
        "swe_closure": swe_closure_stage,
        "inference": inference_stage,
    }


def start_pipeline(session_id: str, config: DEIDConfig, inputs: Dict[str, Any]):
    """
    Starts pipeline exactly like CLI.
    """

    config_hash = compute_config_hash(config)

    input_hashes = {
        "thermal_hdf5": str(inputs["hdf5"]),
        "particle": str(inputs.get("particle") or ""),
        "processed": str(inputs.get("processed") or ""),
    }

    stage_fns = build_stage_registry()

    job, run_dir = run_pipeline(
        session_id=session_id,
        config=config,
        run_root_dir=config.storage.run_root,
        input_hashes=input_hashes,
        inputs=inputs,
        stage_fns=stage_fns,
    )

    return job, run_dir, stage_fns, input_hashes, config_hash