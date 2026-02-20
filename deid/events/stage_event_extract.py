"""
deid.events.stage_event_extract
-------------------------------

Runner stage:

Stage D — Event Extraction (Authoritative)

Inputs:
  inputs/thermal_ref.json
  intermediate/alignment.json
  intermediate/plate_state.npz
  intermediate/plate_state.json

Outputs:
  intermediate/event_catalog.parquet
  intermediate/event_qc.json
  intermediate/event_masks/index.json
  intermediate/event_masks/<event_id>.npz
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from deid.config.models import DEIDConfig
from deid.core.logging import log_info
from deid.core.types import ThermalCubeRef
from deid.core.versioning import SCHEMA_EVENT_CATALOG
from deid.storage.io import read_json, write_json, write_parquet, wrap_artifact
from deid.storage.paths import inputs_dir, intermediate_dir

from deid.events.masks.store import RLEMaskStore
from deid.events.extractors.threshold_morph import ThresholdMorphExtractor


def _load_thermal_ref(run_dir: Path) -> ThermalCubeRef:
    data = read_json(inputs_dir(run_dir) / "thermal_ref.json")
    return ThermalCubeRef(**data)


def _load_wrapped_payload(path: Path) -> Dict[str, Any]:
    w = read_json(path)
    return w.get("payload", {})


def event_extract_stage(
    run_dir: Path,
    inputs: Dict[str, Any],
    config: DEIDConfig,
    context: Dict[str, Any],
) -> None:
    log_info("event_extract_stage_start", run_dir=str(run_dir))

    config_hash: str = inputs["config_hash"]
    input_hashes: Dict[str, str] = inputs.get("input_hashes", {})

    out_dir = intermediate_dir(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    thermal_ref = _load_thermal_ref(run_dir)
    alignment_payload = _load_wrapped_payload(out_dir / "alignment.json")
    plate_meta = _load_wrapped_payload(out_dir / "plate_state.json")
    plate_npz = out_dir / "plate_state.npz"

    # Config
    try:
        ex_cfg = dict(config.event_extraction)
    except Exception:
        ex_cfg = {}

    # Mask store
    masks_dir = out_dir / "event_masks"
    store = RLEMaskStore(masks_dir)

    # Extractor (v1)
    extractor = ThresholdMorphExtractor()
    catalog_df, qc_report = extractor.extract(
        thermal_ref=thermal_ref,
        plate_state_npz_path=plate_npz,
        plate_state_meta=plate_meta,
        alignment_payload=alignment_payload,
        config=ex_cfg,
        mask_store=store,
    )

    # Finalize mask index
    store.finalize()

    # Write catalog parquet
    write_parquet(out_dir / "event_catalog.parquet", catalog_df)

    # Write qc json (wrapped)
    qc_wrapped = wrap_artifact(
        payload=qc_report,
        schema_version="event_qc_v1",
        config_hash=config_hash,
        input_hashes=input_hashes,
        artifact_type="event_qc",
    )
    write_json(out_dir / "event_qc.json", qc_wrapped)

    log_info("event_extract_stage_complete", n_events=int(len(catalog_df)))