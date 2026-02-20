"""
deid.events.qc
--------------

Compute minimal event quality metrics.

v1:
- snr = delta_peak / (noise_at_centroid + eps)
- saturation_flag: if any corrected pixel <= -sat_threshold_counts (heuristic)
- fragmentation_score: currently 0 (placeholder; filled once we do multi-component tracking)
- overlap_score: currently 0 (placeholder; filled once we implement split/merge)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def compute_snr(delta_peak: float, noise_at_centroid: float, eps: float = 1e-6) -> float:
    return float(delta_peak / (noise_at_centroid + eps))


def compute_event_qc_metrics(
    *,
    delta_peak: float,
    noise_at_centroid: float,
    saturation_hit: bool,
) -> Dict[str, Any]:
    snr = compute_snr(delta_peak, noise_at_centroid)
    return {
        "snr": float(snr),
        "overlap_score": 0.0,
        "fragmentation_score": 0.0,
        "saturation_flag": bool(saturation_hit),
        "quality_flags": [],
    }