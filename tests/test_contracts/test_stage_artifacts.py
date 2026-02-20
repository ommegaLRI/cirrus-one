"""
Stage Contract Tests
--------------------

Purpose:
Validate that each pipeline stage produces artifacts that satisfy
the architectural contracts defined in the DEID spec.

These tests DO NOT validate scientific correctness.
They enforce structure, schema, and invariants.

They should remain stable even as algorithms evolve.

Assumptions:
- Golden framework already generates a run directory.
- We reuse one synthetic run per session to avoid recomputation.

If you change schema versions intentionally,
update EXPECTED_SCHEMA below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import pytest

from tests.golden_runs.golden_specs import golden_cases
from tests.fixtures.synthetic_generator import generate_synthetic_session

from deid.storage.io import read_json
from deid.runner import run_pipeline


# ----------------------------
# Config
# ----------------------------

EXPECTED_SCHEMA = {
    "alignment": "alignment_v1",
    "instrument_health": "instrument_health_v1",
    "closure_report": "closure_report_v1",
    "findings": "findings_v1",
    "regimes": "regimes_v1",
    "phenotype_vector": "phenotype_vector_v1",
}


# ----------------------------
# Helpers
# ----------------------------

def _payload(path: Path) -> Dict[str, Any]:
    assert path.exists(), f"Missing artifact: {path}"
    return read_json(path)


def _assert_wrapped(obj: Dict[str, Any], name: str) -> None:
    assert "schema_version" in obj, f"{name}: missing schema_version"
    assert "config_hash" in obj, f"{name}: missing config_hash"
    assert "pipeline_version" in obj, f"{name}: missing pipeline_version"
    assert "payload" in obj, f"{name}: missing payload"


def _run_single_case(tmp_path: Path):
    """
    Build ONE synthetic run for contract testing.

    We intentionally only run the first golden case
    to keep contract tests fast.
    """
    case = golden_cases()[0]

    synth_dir = tmp_path / "contract_case"
    paths = generate_synthetic_session(
        out_dir=synth_dir,
        cfg=case.cfg,
        events=case.events,
        dataset_path="/data",
        write_particle=True,
        write_processed=True,
    )

    run_dir = run_pipeline(
        particle=str(paths["particle_csv"]),
        processed=str(paths["processed_csv"]),
        hdf5=str(paths["thermal_hdf5"]),
        config=None,
    )

    run_dir = Path(run_dir)
    assert run_dir.exists()
    return run_dir


# ----------------------------
# Tests
# ----------------------------

def test_alignment_contract(tmp_path: Path):
    run_dir = _run_single_case(tmp_path)
    inter = run_dir / "intermediate"

    w = _payload(inter / "alignment.json")
    _assert_wrapped(w, "alignment")

    assert w["schema_version"] == EXPECTED_SCHEMA["alignment"]

    p = w["payload"]
    assert "frame_timebase" in p
    assert "confidence" in p
    assert isinstance(p.get("integrity_flags", []), list)


def test_plate_state_contract(tmp_path: Path):
    run_dir = _run_single_case(tmp_path)
    inter = run_dir / "intermediate"

    # NPZ existence + shapes
    npz_path = inter / "plate_state.npz"
    assert npz_path.exists()

    with np.load(npz_path) as z:
        assert "baseline_B_t" in z.files
        assert "nonuniformity_G_yx" in z.files
        assert "noise_N_yx" in z.files
        assert "dead_pixel_mask_yx" in z.files

        B = z["baseline_B_t"]
        assert B.ndim == 1
        assert len(B) > 0

    # Health JSON wrapper
    w = _payload(inter / "instrument_health.json")
    _assert_wrapped(w, "instrument_health")
    assert w["schema_version"] == EXPECTED_SCHEMA["instrument_health"]

    p = w["payload"]
    assert "overall_score" in p
    assert 0.0 <= float(p["overall_score"]) <= 1.0


def test_event_catalog_contract(tmp_path: Path):
    run_dir = _run_single_case(tmp_path)
    inter = run_dir / "intermediate"

    path = inter / "event_catalog.parquet"
    assert path.exists()

    df = pd.read_parquet(path)

    required_cols = [
        "event_id",
        "frame_start",
        "frame_peak",
        "frame_end",
        "delta_peak",
        "snr",
        "mask_ref",
    ]

    for c in required_cols:
        assert c in df.columns, f"Missing column {c}"

    # invariant: frame_start < frame_peak < frame_end
    if len(df):
        assert ((df["frame_start"] < df["frame_peak"]) & (df["frame_peak"] < df["frame_end"])).all()


def test_fusion_contract(tmp_path: Path):
    run_dir = _run_single_case(tmp_path)
    inter = run_dir / "intermediate"

    path = inter / "matched_events.parquet"
    assert path.exists()

    df = pd.read_parquet(path)
    assert "event_id" in df.columns
    assert "match_confidence" in df.columns


def test_swe_closure_contract(tmp_path: Path):
    run_dir = _run_single_case(tmp_path)
    inter = run_dir / "intermediate"
    out = run_dir / "outputs"

    # swe_products
    sp = out / "swe_products.parquet"
    assert sp.exists()
    df = pd.read_parquet(sp)

    assert "swe_reconstructed_mm" in df.columns

    # closure report
    w = _payload(out / "closure_report.json") if (out / "closure_report.json").exists() else _payload(inter / "closure_report.json")
    _assert_wrapped(w, "closure_report")
    assert w["schema_version"] == EXPECTED_SCHEMA["closure_report"]

    p = w["payload"]
    assert "closure_score" in p
    cs = float(p["closure_score"])
    assert 0.0 <= cs <= 1.0


def test_inference_contract(tmp_path: Path):
    run_dir = _run_single_case(tmp_path)
    out = run_dir / "outputs"

    # regimes
    r = _payload(out / "regimes.json")
    _assert_wrapped(r, "regimes")
    assert r["schema_version"] == EXPECTED_SCHEMA["regimes"]

    # phenotype
    ph = _payload(out / "phenotype_vector.json")
    _assert_wrapped(ph, "phenotype_vector")
    assert ph["schema_version"] == EXPECTED_SCHEMA["phenotype_vector"]

    # findings
    f = _payload(out / "findings.json")
    _assert_wrapped(f, "findings")
    assert f["schema_version"] == EXPECTED_SCHEMA["findings"]

    findings = f["payload"].get("findings", [])
    assert isinstance(findings, list)


def test_provenance_contract(tmp_path: Path):
    run_dir = _run_single_case(tmp_path)

    prov = run_dir / "provenance"

    assert (prov / "config.yaml").exists()
    assert (prov / "config_hash.txt").exists()
    assert (prov / "input_hashes.json").exists()