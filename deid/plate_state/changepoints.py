"""
deid.plate_state.changepoints
-----------------------------

Detect simple change points in baseline and noise proxies.

v1:
- baseline change points from baseline_B_t using rolling median and robust threshold.
- (Optional) noise proxy computed from activity series (if provided).

This is intentionally not Bayesian; it's a robust threshold detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class ChangePoint:
    frame_idx: int
    metric: str
    delta: float
    confidence: float


def _rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.astype(np.float64)
    w = int(window)
    n = len(x)
    out = np.zeros((n,), dtype=np.float64)
    half = w // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = float(np.median(x[lo:hi]))
    return out


def detect_baseline_changepoints(
    baseline_B_t: np.ndarray,
    *,
    window: int = 101,
    z_thresh: float = 6.0,
    min_separation: int = 30,
) -> List[ChangePoint]:
    """
    Detect abrupt baseline shifts.

    Steps:
    - compute rolling median m(t)
    - compute residual r(t) = B(t) - m(t)
    - compute robust scale via MAD of r
    - flag points where |r| > z_thresh * scale
    """
    B = np.asarray(baseline_B_t, dtype=np.float64)
    if B.size < 10:
        return []

    m = _rolling_median(B, window=window)
    r = B - m

    med_r = float(np.median(r))
    mad = float(np.median(np.abs(r - med_r)))
    scale = 1.4826 * mad if mad > 0 else float(np.std(r) + 1e-9)

    thresh = float(z_thresh) * scale
    idx = np.where(np.abs(r) > thresh)[0].tolist()

    cps: List[ChangePoint] = []
    last = -10**9
    for i in idx:
        if i - last < int(min_separation):
            continue
        delta = float(r[i])
        # confidence grows with exceedance
        conf = float(min(1.0, max(0.1, abs(delta) / (thresh + 1e-9))))
        cps.append(ChangePoint(frame_idx=int(i), metric="baseline", delta=delta, confidence=conf))
        last = i

    return cps


def changepoints_to_dict(cps: List[ChangePoint]) -> List[Dict[str, Any]]:
    return [
        {"frame_idx": cp.frame_idx, "metric": cp.metric, "delta": cp.delta, "confidence": cp.confidence}
        for cp in cps
    ]