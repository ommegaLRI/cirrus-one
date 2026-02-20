"""
deid.alignment.aligner
----------------------

Map particle events and processed rows onto thermal frame indices.

Outputs (SessionAlignment-like dict):
- frame_timebase
- particle_to_frame: {particle_event_id: frame_idx}
- processed_to_frame: {processed_row_id: frame_idx}
- offsets_seconds: {particle_vs_thermal: float, processed_vs_thermal: float}
- gaps: list (from integrity report)
- integrity_flags
- confidence

This is a v1 alignment: constant-cadence spine with nearest-frame mapping.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from deid.alignment.cadence import FrameTimebase, build_frame_timebase, time_utc_to_frame_index, frame_index_to_time_utc
from deid.alignment.integrity import build_integrity_report
from deid.core.errors import AlignmentError
from deid.core.types import ThermalCubeRef


def _nearest_frame_idx(
    tb: FrameTimebase,
    t_utc: datetime,
    *,
    T: int,
    tolerance_seconds: Optional[float],
) -> Optional[int]:
    idx = time_utc_to_frame_index(tb, t_utc)
    if idx is None:
        return None
    if idx < 0 or idx >= T:
        return None

    if tolerance_seconds is None:
        return idx

    # check mapping residual if time is computable
    t_hat = frame_index_to_time_utc(tb, idx)
    if t_hat is None:
        return idx
    resid = abs((t_utc.astimezone(timezone.utc) - t_hat).total_seconds())
    if resid <= float(tolerance_seconds):
        return idx
    return None


def _median_offset_seconds(tb: FrameTimebase, mapping: Dict[str, int], times: Dict[str, datetime]) -> Optional[float]:
    """
    Offset defined as: observed_time - mapped_frame_time
    (positive means table clock ahead of thermal timebase)
    """
    resids = []
    for key, idx in mapping.items():
        t_obs = times.get(key)
        t_hat = frame_index_to_time_utc(tb, idx)
        if t_obs is None or t_hat is None:
            continue
        resids.append((t_obs.astimezone(timezone.utc) - t_hat).total_seconds())
    if not resids:
        return None
    return float(np.median(np.array(resids, dtype=float)))


def build_session_alignment(
    thermal_ref: ThermalCubeRef,
    particle_table: Optional[pd.DataFrame],
    processed_series: Optional[pd.DataFrame],
    config: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
        alignment_dict, integrity_report_dict
    """
    T, _, _ = thermal_ref.shape

    # Build timebase
    tb = build_frame_timebase(thermal_ref, particle_table, processed_series, config)

    # Integrity report (gaps/flags)
    integrity = build_integrity_report(
        particle_table=particle_table,
        processed_series=processed_series,
        dt_hint_seconds=tb.dt_seconds,
    )

    # tolerances (seconds)
    tol = config.get("timestamp_tolerance_seconds", 2.0)  # conservative default

    particle_to_frame: Dict[str, int] = {}
    processed_to_frame: Dict[str, int] = {}

    particle_times: Dict[str, datetime] = {}
    processed_times: Dict[str, datetime] = {}

    # Map particle rows
    if particle_table is not None and not particle_table.empty:
        if "particle_event_id" not in particle_table.columns:
            raise AlignmentError("particle_table missing particle_event_id")
        if "t_utc" not in particle_table.columns:
            raise AlignmentError("particle_table missing t_utc")

        for _, r in particle_table.iterrows():
            pid = str(r["particle_event_id"])
            t = pd.to_datetime(r["t_utc"], utc=True, errors="coerce")
            if pd.isna(t):
                continue
            t_py = t.to_pydatetime()
            particle_times[pid] = t_py

            idx = _nearest_frame_idx(tb, t_py, T=T, tolerance_seconds=tol)
            if idx is not None:
                particle_to_frame[pid] = int(idx)

    # Map processed rows
    if processed_series is not None and not processed_series.empty:
        if "processed_row_id" not in processed_series.columns:
            raise AlignmentError("processed_series missing processed_row_id")
        if "t_utc" not in processed_series.columns:
            raise AlignmentError("processed_series missing t_utc")

        for _, r in processed_series.iterrows():
            rid = str(r["processed_row_id"])
            t = pd.to_datetime(r["t_utc"], utc=True, errors="coerce")
            if pd.isna(t):
                continue
            t_py = t.to_pydatetime()
            processed_times[rid] = t_py

            idx = _nearest_frame_idx(tb, t_py, T=T, tolerance_seconds=tol)
            if idx is not None:
                processed_to_frame[rid] = int(idx)

    # Compute offsets (median residuals)
    off_particle = _median_offset_seconds(tb, particle_to_frame, particle_times)
    off_processed = _median_offset_seconds(tb, processed_to_frame, processed_times)

    offsets_seconds = {
        "particle_vs_thermal": float(off_particle) if off_particle is not None else None,
        "processed_vs_thermal": float(off_processed) if off_processed is not None else None,
    }

    # Confidence: combine timebase confidence + mapping coverage
    pt_cov = (len(particle_to_frame) / max(1, len(particle_times))) if particle_times else 0.0
    pr_cov = (len(processed_to_frame) / max(1, len(processed_times))) if processed_times else 0.0
    cov = max(pt_cov, pr_cov)  # whichever provides better anchoring
    confidence = float(np.clip(0.6 * tb.confidence + 0.4 * cov, 0.0, 1.0))

    alignment = {
        "frame_timebase": {
            "source": tb.source,
            "confidence": tb.confidence,
            "t0_utc": tb.t0_utc,
            "dt_seconds": tb.dt_seconds,
            "frame_timestamps_utc": tb.frame_timestamps_utc,
        },
        "particle_to_frame": dict(sorted(particle_to_frame.items())),
        "processed_to_frame": dict(sorted(processed_to_frame.items())),
        "offsets_seconds": offsets_seconds,
        "gaps": integrity.get("gaps", []),
        "integrity_flags": integrity.get("integrity_flags", []),
        "confidence": confidence,
    }

    return alignment, integrity