from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from deid.storage.io import read_json


def _load_payload(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path).get("payload", {})


def extract_run_metrics(run_dir: Path) -> Dict[str, Any]:
    """
    Extract a stable, low-dimensional signature from a run bundle.

    This should change only when something meaningful breaks.
    """
    run_dir = Path(run_dir)
    inter = run_dir / "intermediate"
    out = run_dir / "outputs"

    metrics: Dict[str, Any] = {}

    # event count
    event_path = inter / "event_catalog.parquet"
    if event_path.exists():
        ev = pd.read_parquet(event_path)
        metrics["event_count"] = int(len(ev))
    else:
        metrics["event_count"] = 0

    # closure score
    closure_payload = _load_payload(out / "closure_report.json") or _load_payload(inter / "closure_report.json")
    metrics["closure_score"] = closure_payload.get("closure_score", None)

    # instrument health score
    health_payload = _load_payload(inter / "instrument_health.json")
    metrics["health_score"] = health_payload.get("overall_score", None)

    # baseline drift magnitude
    plate_npz = inter / "plate_state.npz"
    if plate_npz.exists():
        with np.load(plate_npz) as z:
            B = z["baseline_B_t"].astype(float)
            # robust drift metric
            metrics["baseline_drift_abs"] = float(abs(B[-1] - B[0])) if len(B) >= 2 else 0.0
            # add noise map summary (useful regression signal)
            if "noise_N_yx" in z.files:
                N = z["noise_N_yx"].astype(float)
                metrics["noise_mean"] = float(np.nanmean(N))
                metrics["noise_p95"] = float(np.nanpercentile(N, 95))
    else:
        metrics["baseline_drift_abs"] = None
        metrics["noise_mean"] = None
        metrics["noise_p95"] = None

    # regimes
    regimes_payload = _load_payload(out / "regimes.json")
    if regimes_payload:
        metrics["n_regimes"] = int(regimes_payload.get("n_regimes", 0) or 0)
    else:
        metrics["n_regimes"] = 0

    # inference gating outcome (presence of inference_skipped)
    metrics["inference_skipped"] = bool((out / "inference_skipped.json").exists())

    return metrics


def within_range(value: Any, lo: float, hi: float) -> bool:
    if value is None:
        return False
    try:
        v = float(value)
    except Exception:
        return False
    return (v >= float(lo)) and (v <= float(hi))