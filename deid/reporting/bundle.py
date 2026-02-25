"""
deid.reporting.bundle
---------------------

Build and validate the final run bundle.

Responsibilities:
- Verify required artifacts exist
- Produce qc_summary.json
- Produce summary.json (top-level quick overview)
- Deterministic artifact manifest

Consumes only artifacts written by previous stages.
"""

import pandas as pd
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

from deid.storage.io import read_json, write_json, wrap_artifact
from deid.storage.paths import intermediate_dir, outputs_dir


REQUIRED_INTERMEDIATE = [
    "alignment.json",
    "instrument_health.json",
    "event_catalog.parquet",
]

REQUIRED_OUTPUTS = [
    "closure_report.json",
    "swe_products.parquet",
]


def _safe_load_payload(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return read_json(path).get("payload", {})
    except Exception:
        return {}


def _artifact_manifest(run_dir: Path) -> Dict[str, List[str]]:
    inter = intermediate_dir(run_dir)
    out = outputs_dir(run_dir)

    return {
        "intermediate": sorted([p.name for p in inter.glob("*")]),
        "outputs": sorted([p.name for p in out.glob("*")]),
    }


def _build_qc_summary(
    instrument_health: Dict[str, Any],
    closure_report: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "health_score": instrument_health.get("overall_score"),
        "closure_score": closure_report.get("closure_score"),
        "failure_modes": closure_report.get("failure_modes", []),
        "recommendations": closure_report.get("recommendations", []),
    }

def _export_event_catalog_json(
    *,
    run_dir: Path,
    config_hash: str,
    input_hashes: Dict[str, str],
) -> None:
    """
    Export event_catalog.parquet → event_catalog.json

    This is a deterministic, frontend-friendly representation
    of the authoritative event catalog.

    Rules:
    - Preserve ordering by event_id
    - Convert NaN → None
    - Wrap with provenance metadata
    """

    inter = intermediate_dir(run_dir)
    parquet_path = inter / "event_catalog.parquet"
    json_path = inter / "event_catalog.json"

    if not parquet_path.exists():
        return

    df = pd.read_parquet(parquet_path)

    # Ensure deterministic ordering
    if "event_id" in df.columns:
        df = df.sort_values("event_id")

    # Convert NaN → None
    payload = df.where(pd.notnull(df), None).to_dict(orient="records")

    wrapped = wrap_artifact(
        payload=payload,
        schema_version="event_catalog_json_v1",
        config_hash=config_hash,
        input_hashes=input_hashes,
        artifact_type="event_catalog_json",
    )

    write_json(json_path, wrapped)


def build_run_bundle(
    *,
    run_dir: Path,
    config_hash: str,
    input_hashes: Dict[str, str],
) -> None:
    inter = intermediate_dir(run_dir)
    out = outputs_dir(run_dir)

    # -------------------------------------------------
    # Derived exports for frontend consumption
    # -------------------------------------------------
    _export_event_catalog_json(
        run_dir=run_dir,
        config_hash=config_hash,
        input_hashes=input_hashes,
    )

    instrument_health = _safe_load_payload(inter / "instrument_health.json")
    closure_report = _safe_load_payload(out / "closure_report.json")

    qc_summary = _build_qc_summary(instrument_health, closure_report)

    qc_wrapped = wrap_artifact(
        payload=qc_summary,
        schema_version="qc_summary_v1",
        config_hash=config_hash,
        input_hashes=input_hashes,
        artifact_type="qc_summary",
    )
    write_json(out / "qc_summary.json", qc_wrapped)

    summary = {
        "schema_version": "run_summary_v1",
        "artifact_manifest": _artifact_manifest(run_dir),
        "qc_summary": qc_summary,
    }

    summary_wrapped = wrap_artifact(
        payload=summary,
        schema_version="run_summary_v1",
        config_hash=config_hash,
        input_hashes=input_hashes,
        artifact_type="run_summary",
    )
    write_json(out / "summary.json", summary_wrapped)