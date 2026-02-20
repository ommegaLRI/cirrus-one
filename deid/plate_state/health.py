"""
deid.plate_state.health
-----------------------

InstrumentHealthReport (v1).

Computes health metrics from PlateStateEstimate + activity signal:
- baseline drift magnitude
- dead pixel fraction
- noise level summary
- noise stability proxy (optional)
- activity distribution (optional)
- overall_score in [0,1]

This is designed for QC gating later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from deid.plate_state.estimator import PlateStateEstimate
from deid.plate_state.changepoints import ChangePoint, detect_baseline_changepoints


@dataclass(frozen=True)
class InstrumentHealthReport:
    overall_score: float
    qc_metrics: Dict[str, float]
    flags: list[str]
    change_points: list[dict]


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def compute_instrument_health(
    plate: PlateStateEstimate,
    *,
    activity: Optional[np.ndarray] = None,
) -> InstrumentHealthReport:
    B = np.asarray(plate.baseline_B_t, dtype=np.float64)
    N = np.asarray(plate.noise_N_yx, dtype=np.float64)
    dead = np.asarray(plate.dead_pixel_mask_yx, dtype=bool)

    flags: list[str] = []

    # Baseline drift magnitude (counts)
    drift_range = float(np.nanmax(B) - np.nanmin(B))
    drift_iqr = float(np.subtract(*np.nanpercentile(B, [75, 25])))

    # Noise summaries (counts)
    noise_med = float(np.nanmedian(N))
    noise_p90 = float(np.nanpercentile(N, 90))
    noise_p10 = float(np.nanpercentile(N, 10))
    noise_spread = float(noise_p90 - noise_p10)

    # Dead pixels
    dead_frac = float(np.mean(dead.astype(np.float64)))

    # Change points in baseline (v1)
    cps = detect_baseline_changepoints(B)
    cp_count = float(len(cps))
    cp_dicts = [
        {"frame_idx": c.frame_idx, "metric": c.metric, "delta": c.delta, "confidence": c.confidence} for c in cps
    ]

    # Optional activity metrics
    act_med = None
    act_p90 = None
    if activity is not None and len(activity) == len(B):
        a = np.asarray(activity, dtype=np.float64)
        act_med = float(np.nanmedian(a))
        act_p90 = float(np.nanpercentile(a, 90))

    # Flags
    if dead_frac > 0.01:
        flags.append("dead_pixels_detected")
    if drift_range > 500:  # heuristic; device-specific, tunable later
        flags.append("large_baseline_drift")
    if cp_count >= 3:
        flags.append("multiple_baseline_changepoints")
    if noise_med <= 0:
        flags.append("invalid_noise_map")

    # Score components (heuristic v1, explainable)
    # Each component yields penalty; overall = 1 - weighted_penalty
    p_dead = _clip01(dead_frac / 0.02)              # 2% dead -> full penalty
    p_drift = _clip01(drift_range / 1000.0)         # 1000 counts -> full penalty
    p_noise = _clip01(noise_spread / (noise_med * 4.0 + 1e-9))  # large spread -> penalty
    p_cp = _clip01(cp_count / 5.0)                  # 5 changepoints -> full penalty

    penalty = 0.35 * p_dead + 0.30 * p_drift + 0.20 * p_noise + 0.15 * p_cp
    overall = _clip01(1.0 - penalty)

    qc_metrics: Dict[str, float] = {
        "baseline_drift_range_counts": drift_range,
        "baseline_drift_iqr_counts": drift_iqr,
        "noise_median_counts": noise_med,
        "noise_spread_p90_p10_counts": noise_spread,
        "dead_pixel_fraction": dead_frac,
        "baseline_changepoint_count": cp_count,
        "overall_score": overall,
    }
    if act_med is not None:
        qc_metrics["activity_median"] = float(act_med)
    if act_p90 is not None:
        qc_metrics["activity_p90"] = float(act_p90)

    return InstrumentHealthReport(
        overall_score=overall,
        qc_metrics=qc_metrics,
        flags=sorted(set(flags)),
        change_points=cp_dicts,
    )