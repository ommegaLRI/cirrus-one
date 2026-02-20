"""
deid.plate_state.quiescent
--------------------------

Select quiescent frames (low activity) from the thermal cube.

v1 method:
- For each frame, compute a robust "activity" metric:
    activity[t] = mean(abs(frame - median(frame)))
  (L1 deviation from median; robust to sparse events)
- Select bottom q fraction as quiescent.

This is deterministic, fast, and works without event masks.

Outputs:
- quiescent_indices: list[int]
- activity: np.ndarray shape (T,) float64
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

from deid.core.errors import AlignmentError, SchemaError
from deid.core.types import ThermalCubeRef


@dataclass(frozen=True)
class QuiescentSelection:
    quiescent_indices: List[int]
    activity: np.ndarray  # shape (T,)
    method: str
    params: Dict[str, Any]


def _open_dataset(ref: ThermalCubeRef) -> Tuple[h5py.File, h5py.Dataset]:
    h5 = h5py.File(ref.uri, "r")
    if ref.dataset_path not in h5:
        h5.close()
        raise SchemaError(
            "ThermalCubeRef.dataset_path not found in HDF5",
            details={"uri": ref.uri, "dataset_path": ref.dataset_path},
        )
    ds = h5[ref.dataset_path]
    return h5, ds


def compute_frame_activity_l1(ref: ThermalCubeRef) -> np.ndarray:
    """
    Compute activity per frame: mean(abs(frame - median(frame))).

    Reads frames sequentially (T is ~2k so this is feasible).
    """
    T, H, W = ref.shape
    activity = np.zeros((T,), dtype=np.float64)

    h5, ds = _open_dataset(ref)
    try:
        for t in range(T):
            frame = ds[t, :, :]
            med = np.median(frame)
            activity[t] = float(np.mean(np.abs(frame.astype(np.float32) - med)))
    finally:
        h5.close()

    return activity


def select_quiescent_frames(
    ref: ThermalCubeRef,
    *,
    quiescent_fraction: float = 0.20,
    min_frames: int = 30,
) -> QuiescentSelection:
    """
    Select quiescent frames using bottom quantile of activity metric.
    """
    if not (0.0 < quiescent_fraction <= 1.0):
        raise SchemaError("quiescent_fraction must be in (0,1]", details={"quiescent_fraction": quiescent_fraction})

    activity = compute_frame_activity_l1(ref)
    T = int(ref.shape[0])

    k = max(int(round(quiescent_fraction * T)), int(min_frames))
    k = min(k, T)

    # argsort is deterministic; choose k smallest activity frames
    idx = np.argsort(activity)[:k]
    idx_sorted = sorted(int(i) for i in idx.tolist())

    return QuiescentSelection(
        quiescent_indices=idx_sorted,
        activity=activity,
        method="activity_l1_bottom_quantile",
        params={"quiescent_fraction": float(quiescent_fraction), "min_frames": int(min_frames)},
    )