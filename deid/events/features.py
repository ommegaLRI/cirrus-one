"""
deid.events.features
--------------------

Compute minimal impulse features for an event from:
- corrected frames (float32)
- per-frame masks

v1 provides:
- delta_peak: max magnitude cooling (positive number)
- area_peak_px: max mask area
- energy_proxy_E: sum of (-corrected) over mask over time
- duration_frames, duration_s (if dt available)
- centroid per frame (mask centroid)
- motion_score (normalized centroid displacement)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(ys)), float(np.mean(xs))


def compute_event_features(
    *,
    corrected_frames: List[np.ndarray],   # each (H,W) float32
    masks: List[np.ndarray],              # each (H,W) bool
    frame_indices: List[int],
    dt_seconds: Optional[float],
) -> Dict[str, Any]:
    assert len(corrected_frames) == len(masks) == len(frame_indices)
    F = len(frame_indices)

    delta_peak = 0.0
    area_peak_px = 0.0
    energy_proxy_E = 0.0

    centroids: List[Tuple[float, float]] = []

    for frame, mask in zip(corrected_frames, masks):
        if not np.any(mask):
            centroids.append((float("nan"), float("nan")))
            continue
        vals = frame[mask]
        # cooling appears as negative; magnitude is -min
        d = float(-np.min(vals))
        delta_peak = max(delta_peak, d)

        area = float(np.sum(mask))
        area_peak_px = max(area_peak_px, area)

        energy_proxy_E += float(np.sum(-vals))

        centroids.append(_mask_centroid(mask))

    # duration
    duration_frames = float(F)
    duration_s = float(F * dt_seconds) if dt_seconds is not None else None

    # motion score: total centroid displacement / sqrt(H^2+W^2) (computed later by caller if needed)
    # Here: compute raw displacement
    valid = [(y, x) for (y, x) in centroids if np.isfinite(y) and np.isfinite(x)]
    motion = 0.0
    if len(valid) >= 2:
        dy = valid[-1][0] - valid[0][0]
        dx = valid[-1][1] - valid[0][1]
        motion = float(np.hypot(dy, dx))

    return {
        "delta_peak": float(delta_peak),
        "area_peak_px": float(area_peak_px),
        "energy_proxy_E": float(energy_proxy_E),
        "duration_s": duration_s,
        "centroids": centroids,
        "motion_disp_px": motion,
        "frame_start": int(frame_indices[0]),
        "frame_end": int(frame_indices[-1]),
    }