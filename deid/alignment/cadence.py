"""
deid.alignment.cadence
----------------------

Frame timebase inference.

v1 strategy:
- If config provides dt_seconds, use it (source="config").
- Else infer dt from particle and/or processed timestamps by robust median of diffs.
- If we can infer dt AND we have at least one timestamp, build a constant-cadence timebase
  spanning the thermal cube length T, anchored at t0 = min(observed timestamps).
- If insufficient info, return a timebase with dt_seconds=None and t0_utc=None.

Important:
- HDF5 time metadata is not assumed to exist (DEID deployments often omit it).
- This module does not attempt to align offsets; it only builds a plausible frame time spine.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from deid.core.errors import AlignmentError
from deid.core.types import ThermalCubeRef


@dataclass(frozen=True)
class FrameTimebase:
    t0_utc: Optional[str]          # ISO string in UTC, or None
    dt_seconds: Optional[float]    # constant cadence, or None
    frame_timestamps_utc: Optional[list[str]]  # ISO list if irregular/explicit
    source: str                   # "config" | "inferred" | "aligned_from_tables" | "unknown"
    confidence: float             # 0..1


def _to_utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _extract_times_utc(df: Optional[pd.DataFrame]) -> list[datetime]:
    if df is None or df.empty:
        return []
    if "t_utc" not in df.columns:
        return []
    s = pd.to_datetime(df["t_utc"], utc=True, errors="coerce")
    s = s.dropna()
    # Convert to python datetimes
    return [x.to_pydatetime() for x in s.to_list()]


def _robust_median_dt_seconds(times: list[datetime]) -> Optional[float]:
    if len(times) < 3:
        return None
    times_sorted = sorted(times)
    diffs = []
    for a, b in zip(times_sorted[:-1], times_sorted[1:]):
        d = (b - a).total_seconds()
        if d > 0:
            diffs.append(d)
    if len(diffs) < 2:
        return None
    # robust median
    return float(np.median(np.array(diffs, dtype=float)))


def infer_frame_cadence(
    thermal_ref: ThermalCubeRef,
    particle_table: Optional[pd.DataFrame],
    processed_series: Optional[pd.DataFrame],
    config: Dict[str, Any],
) -> Tuple[Optional[float], float, str]:
    """
    Infer constant frame cadence dt_seconds.

    Returns:
        (dt_seconds or None, confidence 0..1, source string)
    """
    # 1) explicit config override
    dt_cfg = None
    try:
        dt_cfg = config.get("dt_seconds", None)
    except Exception:
        dt_cfg = None
    if dt_cfg is not None:
        try:
            dt = float(dt_cfg)
            if dt <= 0:
                raise ValueError()
            return dt, 0.95, "config"
        except Exception:
            raise AlignmentError("Invalid dt_seconds in alignment config", details={"dt_seconds": dt_cfg})

    # 2) infer from tables (particle preferred, then processed)
    pt = _extract_times_utc(particle_table)
    pr = _extract_times_utc(processed_series)

    dt_pt = _robust_median_dt_seconds(pt) if pt else None
    dt_pr = _robust_median_dt_seconds(pr) if pr else None

    # Prefer particle-derived cadence if available
    if dt_pt is not None:
        # confidence grows with number of diffs available
        conf = min(0.9, 0.4 + 0.1 * max(0, len(pt) - 2))
        return dt_pt, float(conf), "aligned_from_tables"

    if dt_pr is not None:
        conf = min(0.8, 0.3 + 0.05 * max(0, len(pr) - 2))
        return dt_pr, float(conf), "aligned_from_tables"

    # 3) no inference possible
    return None, 0.1, "unknown"


def build_frame_timebase(
    thermal_ref: ThermalCubeRef,
    particle_table: Optional[pd.DataFrame],
    processed_series: Optional[pd.DataFrame],
    config: Dict[str, Any],
) -> FrameTimebase:
    """
    Build FrameTimebase for the thermal cube.

    v1 produces either:
    - constant cadence timebase: (t0_utc, dt_seconds, frame_timestamps_utc=None)
    - unknown timebase: all None except source/confidence

    Anchoring:
    - If we have any timestamps from tables, t0 = min(observed timestamps).
    """
    T, _, _ = thermal_ref.shape

    dt, conf, source = infer_frame_cadence(thermal_ref, particle_table, processed_series, config)

    times = _extract_times_utc(particle_table) + _extract_times_utc(processed_series)
    t0 = min(times) if times else None

    if dt is None or t0 is None:
        return FrameTimebase(
            t0_utc=_to_utc_iso(t0) if t0 else None,
            dt_seconds=None,
            frame_timestamps_utc=None,
            source=source,
            confidence=float(conf),
        )

    # Build constant cadence spanning full cube length
    # We store only t0 + dt (not full timestamps) to keep artifacts small.
    return FrameTimebase(
        t0_utc=_to_utc_iso(t0),
        dt_seconds=float(dt),
        frame_timestamps_utc=None,
        source=source if source != "unknown" else "inferred",
        confidence=float(conf),
    )


def frame_timebase_to_dict(tb: FrameTimebase) -> Dict[str, Any]:
    return asdict(tb)


def frame_index_to_time_utc(tb: FrameTimebase, frame_idx: int) -> Optional[datetime]:
    """
    Map frame_idx -> timestamp if possible.
    """
    if tb.frame_timestamps_utc:
        if 0 <= frame_idx < len(tb.frame_timestamps_utc):
            return datetime.fromisoformat(tb.frame_timestamps_utc[frame_idx])
        return None
    if tb.t0_utc and tb.dt_seconds is not None:
        t0 = datetime.fromisoformat(tb.t0_utc)
        return t0 + timedelta(seconds=float(tb.dt_seconds) * int(frame_idx))
    return None


def time_utc_to_frame_index(tb: FrameTimebase, t_utc: datetime) -> Optional[int]:
    """
    Map timestamp -> frame_idx if possible.
    """
    if t_utc.tzinfo is None:
        t_utc = t_utc.replace(tzinfo=timezone.utc)
    t_utc = t_utc.astimezone(timezone.utc)

    if tb.frame_timestamps_utc:
        # nearest neighbor search (linear; good enough for v1, T~2k)
        ts = [datetime.fromisoformat(s) for s in tb.frame_timestamps_utc]
        diffs = [abs((x - t_utc).total_seconds()) for x in ts]
        return int(np.argmin(np.array(diffs))) if diffs else None

    if tb.t0_utc and tb.dt_seconds is not None:
        t0 = datetime.fromisoformat(tb.t0_utc).astimezone(timezone.utc)
        dt = float(tb.dt_seconds)
        idx = int(round((t_utc - t0).total_seconds() / dt))
        return idx

    return None