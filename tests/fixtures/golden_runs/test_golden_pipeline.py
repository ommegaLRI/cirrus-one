from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any

import pytest

from tests.golden_runs.golden_specs import golden_cases
from tests.golden_runs.metrics import extract_run_metrics, within_range
from tests.fixtures.synthetic_generator import generate_synthetic_session

# IMPORTANT: adjust this import if your runner exports a different symbol.
# Your Tier 3 API example used: from deid.runner import run_pipeline
from deid.runner import run_pipeline


GOLDENS_ROOT = Path(__file__).resolve().parent / "goldens"


def _write_golden(case_id: str, metrics: Dict[str, Any]) -> None:
    d = GOLDENS_ROOT / case_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "golden_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


def _read_golden(case_id: str) -> Dict[str, Any]:
    p = GOLDENS_ROOT / case_id / "golden_metrics.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing golden metrics for case '{case_id}'. "
            f"Run with DEID_REGEN_GOLDEN=1 to generate: {p}"
        )
    return json.loads(p.read_text(encoding="utf-8"))


def _assert_metrics_against_expected(metrics: Dict[str, Any], expected: Dict[str, Any]) -> None:
    # event count range
    lo, hi = expected["event_count_range"]
    assert within_range(metrics.get("event_count"), lo, hi), f"event_count={metrics.get('event_count')} not in [{lo},{hi}]"

    # closure score range (may be None if closure stage missing)
    lo, hi = expected["closure_score_range"]
    cs = metrics.get("closure_score")
    assert cs is None or within_range(cs, lo, hi), f"closure_score={cs} not in [{lo},{hi}]"

    # health score range
    lo, hi = expected["health_score_range"]
    hs = metrics.get("health_score")
    assert hs is None or within_range(hs, lo, hi), f"health_score={hs} not in [{lo},{hi}]"

    # baseline drift abs range
    lo, hi = expected["baseline_drift_abs_range"]
    bd = metrics.get("baseline_drift_abs")
    assert bd is None or within_range(bd, lo, hi), f"baseline_drift_abs={bd} not in [{lo},{hi}]"

    # n_regimes range
    lo, hi = expected["n_regimes_range"]
    assert within_range(metrics.get("n_regimes"), lo, hi), f"n_regimes={metrics.get('n_regimes')} not in [{lo},{hi}]"


@pytest.mark.parametrize("case", golden_cases(), ids=lambda c: c.case_id)
def test_golden_pipeline(tmp_path: Path, case) -> None:
    regen = os.getenv("DEID_REGEN_GOLDEN", "0") == "1"

    # 1) Generate synthetic inputs
    synth_dir = tmp_path / case.case_id
    paths = generate_synthetic_session(
        out_dir=synth_dir,
        cfg=case.cfg,
        events=case.events,
        dataset_path="/data",
        write_particle=True,
        write_processed=True,
    )

    # 2) Run pipeline end-to-end
    # NOTE: This assumes run_pipeline returns the run_dir (as used by your API file).
    run_dir = run_pipeline(
        particle=str(paths["particle_csv"]) if paths.get("particle_csv") else None,
        processed=str(paths["processed_csv"]) if paths.get("processed_csv") else None,
        hdf5=str(paths["thermal_hdf5"]),
        config=None,
    )

    run_dir = Path(run_dir)
    assert run_dir.exists(), f"run_dir does not exist: {run_dir}"

    # 3) Extract metrics
    metrics = extract_run_metrics(run_dir)

    # 4) Regen or compare
    if regen:
        _write_golden(case.case_id, metrics)
        # still enforce basic invariants
        _assert_metrics_against_expected(metrics, case.expected)
    else:
        golden = _read_golden(case.case_id)

        # Compare exact keys (but allow extra keys to be added later)
        for k in ["event_count", "closure_score", "health_score", "baseline_drift_abs", "n_regimes", "inference_skipped"]:
            assert k in golden, f"golden missing key: {k}"

        # Primary regression checks: numeric values should be close-ish.
        # We do range-based checks using the spec’s expected ranges (stable),
        # and also assert we didn’t drift wildly from the stored golden.
        _assert_metrics_against_expected(metrics, case.expected)

        # Stored golden deltas (tighten as your system stabilizes)
        # event_count exact match is usually stable once extractor stabilizes; keep a tolerance now.
        assert abs(int(metrics["event_count"]) - int(golden["event_count"])) <= 50

        # closure/health can vary slightly; bound the deviation
        if metrics["closure_score"] is not None and golden["closure_score"] is not None:
            assert abs(float(metrics["closure_score"]) - float(golden["closure_score"])) <= 0.20
        if metrics["health_score"] is not None and golden["health_score"] is not None:
            assert abs(float(metrics["health_score"]) - float(golden["health_score"])) <= 0.20