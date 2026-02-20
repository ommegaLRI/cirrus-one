"""
deid.plate_state.estimator
--------------------------

Compute PlateState components:
- baseline_B_t (per-frame robust baseline)
- nonuniformity_G_yx (per-pixel deviation from global median on quiescent frames)
- noise_N_yx (per-pixel robust noise estimate from quiescent frames)
- dead_pixel_mask_yx (simple v1 heuristic)

v1 baseline:
- baseline_B_t[t] = median(frame[t])  (robust to sparse events)

v1 nonuniformity:
- per_pixel_median = median over quiescent frames
- global_median = median of per_pixel_median over all pixels
- G = per_pixel_median - global_median

v1 noise:
- per_pixel_MAD over quiescent frames, scaled to sigma via 1.4826
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

from deid.core.errors import SchemaError
from deid.core.types import ThermalCubeRef
from deid.plate_state.quiescent import QuiescentSelection, _open_dataset


@dataclass(frozen=True)
class PlateStateEstimate:
    baseline_B_t: np.ndarray           # (T,) float64
    nonuniformity_G_yx: np.ndarray     # (H,W) float32
    noise_N_yx: np.ndarray             # (H,W) float32
    dead_pixel_mask_yx: np.ndarray     # (H,W) bool
    method: str
    params: Dict[str, Any]
    quiescent_indices: List[int]


def _robust_mad_sigma(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Robust sigma estimate via MAD * 1.4826.
    """
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis)
    return mad * 1.4826


def estimate_plate_state(
    ref: ThermalCubeRef,
    quiescent: QuiescentSelection,
    *,
    dead_noise_eps: float = 1e-6,
    dtype_float: Any = np.float32,
) -> PlateStateEstimate:
    """
    Compute plate state arrays using quiescent frames.
    """
    T, H, W = ref.shape
    q_idx = quiescent.quiescent_indices
    if not q_idx:
        raise SchemaError("No quiescent frames provided")

    # 1) baseline B(t): per-frame median
    baseline_B_t = np.zeros((T,), dtype=np.float64)

    h5, ds = _open_dataset(ref)
    try:
        for t in range(T):
            frame = ds[t, :, :]
            baseline_B_t[t] = float(np.median(frame))

        # 2) Stack quiescent frames into array for per-pixel robust stats
        # Shape (Q,H,W) as float32 to keep memory reasonable
        Q = len(q_idx)
        q_stack = np.empty((Q, H, W), dtype=np.float32)
        for i, t in enumerate(q_idx):
            q_stack[i, :, :] = ds[int(t), :, :].astype(np.float32)

    finally:
        h5.close()

    # 3) per-pixel median over quiescent frames
    per_pix_med = np.median(q_stack, axis=0)  # (H,W) float32
    global_med = float(np.median(per_pix_med))

    nonuniformity_G = (per_pix_med - global_med).astype(dtype_float)

    # 4) noise map via robust sigma MAD
    noise_sigma = _robust_mad_sigma(q_stack, axis=0).astype(dtype_float)

    # 5) dead pixels: extremely low noise OR always saturated/extreme (v1 heuristic)
    dead_by_noise = noise_sigma <= float(dead_noise_eps)

    # Saturation heuristic: pixel median at extrema in uint16 space (0 or 65535) over quiescent frames
    # (Not all devices use full range, but this flags obvious dead/saturated channels.)
    med_uint = per_pix_med
    dead_by_extreme = (med_uint <= 0.0) | (med_uint >= 65535.0)

    dead_mask = (dead_by_noise | dead_by_extreme).astype(bool)

    return PlateStateEstimate(
        baseline_B_t=baseline_B_t,
        nonuniformity_G_yx=nonuniformity_G,
        noise_N_yx=noise_sigma,
        dead_pixel_mask_yx=dead_mask,
        method="v1_median_baseline_quiescent_mad_noise",
        params={"dead_noise_eps": float(dead_noise_eps), **quiescent.params},
        quiescent_indices=list(q_idx),
    )