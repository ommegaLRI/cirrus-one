"""
deid.alignment.integrity
------------------------

Detect input pathologies early and emit flags; do not "fix" data.

Functions:
- check_swe_monotonicity(processed_series)
- detect_duplicate_times(times)
- detect_time_gaps(times, dt_hint_seconds)

Outputs:
- integrity flags list[str]
- gaps list[dict] (compatible with your TimeGap concept)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _extract_times(df: Optional[pd.DataFrame]) -> List[datetime]:
    if df is None or df.empty or "t_utc" not in df.columns:
        return []
    s = pd.to_datetime(df["t_utc"], utc=True, errors="coerce").dropna()
    return [x.to_pydatetime() for x in s.to_list()]


def detect_duplicate_times(times: List[datetime]) -> List[Dict[str, Any]]:
    """
    Detect exact duplicate timestamps.
    Returns list of {t_start_utc, t_end_utc, kind="duplicate"}.
    """
    if not times:
        return []
    times_sorted = sorted(times)
    out = []
    for a, b in zip(times_sorted[:-1], times_sorted[1:]):
        if a == b:
            out.append(
                {
                    "source": "unknown",
                    "t_start_utc": a.astimezone(timezone.utc).isoformat(),
                    "t_end_utc": b.astimezone(timezone.utc).isoformat(),
                    "kind": "duplicate",
                }
            )
    return out


def detect_time_gaps(
    times: List[datetime],
    *,
    source: str,
    dt_hint_seconds: Optional[float] = None,
    gap_factor: float = 5.0,
) -> List[Dict[str, Any]]:
    """
    Detect gaps where delta_t exceeds gap_factor * dt.

    If dt_hint_seconds is None:
      uses median positive diff as dt estimate.
    """
    if len(times) < 3:
        return []

    times_sorted = sorted(times)
    diffs = []
    for a, b in zip(times_sorted[:-1], times_sorted[1:]):
        d = (b - a).total_seconds()
        if d > 0:
            diffs.append(d)

    if len(diffs) < 2:
        return []

    dt = float(dt_hint_seconds) if dt_hint_seconds and dt_hint_seconds > 0 else float(np.median(diffs))
    threshold = gap_factor * dt

    gaps: List[Dict[str, Any]] = []
    for a, b in zip(times_sorted[:-1], times_sorted[1:]):
        d = (b - a).total_seconds()
        if d > threshold:
            gaps.append(
                {
                    "source": source,
                    "t_start_utc": a.astimezone(timezone.utc).isoformat(),
                    "t_end_utc": b.astimezone(timezone.utc).isoformat(),
                    "kind": "missing",
                    "delta_seconds": float(d),
                    "dt_reference_seconds": float(dt),
                }
            )
    return gaps


def check_swe_monotonicity(processed_series: Optional[pd.DataFrame]) -> Tuple[List[str], Dict[str, Any]]:
    """
    Check non-decreasing constraints on processed SWE.
    Returns (flags, summary dict).
    """
    if processed_series is None or processed_series.empty:
        return [], {"status": "no_processed"}

    flags: List[str] = []
    summary: Dict[str, Any] = {"status": "ok"}

    if "swe_mm" not in processed_series.columns:
        flags.append("processed_missing_swe_mm")
        summary["status"] = "missing_column"
        return flags, summary

    swe = pd.to_numeric(processed_series["swe_mm"], errors="coerce")
    t = pd.to_datetime(processed_series.get("t_utc", None), utc=True, errors="coerce")

    # filter valid pairs
    mask = (~swe.isna()) & (~t.isna())
    swe = swe[mask].to_numpy(dtype=float)
    if len(swe) < 3:
        return flags, {"status": "insufficient_points", "n": int(len(swe))}

    diffs = np.diff(swe)
    n_viol = int(np.sum(diffs < 0))
    if n_viol > 0:
        flags.append("swe_non_monotonic")
        summary["status"] = "non_monotonic"
        summary["n_violations"] = n_viol
        summary["min_diff_mm"] = float(np.min(diffs))
    else:
        summary["n_violations"] = 0

    return flags, summary


def build_integrity_report(
    *,
    particle_table: Optional[pd.DataFrame],
    processed_series: Optional[pd.DataFrame],
    dt_hint_seconds: Optional[float],
) -> Dict[str, Any]:
    """
    Consolidated integrity report (v1).
    """
    flags: List[str] = []
    gaps: List[Dict[str, Any]] = []

    # duplicates + gaps in particle times
    pt = _extract_times(particle_table)
    if pt:
        gaps.extend(detect_duplicate_times(pt))
        gaps.extend(detect_time_gaps(pt, source="particle", dt_hint_seconds=dt_hint_seconds))

    # duplicates + gaps in processed times
    pr = _extract_times(processed_series)
    if pr:
        gaps.extend(detect_duplicate_times(pr))
        gaps.extend(detect_time_gaps(pr, source="processed", dt_hint_seconds=dt_hint_seconds))

    swe_flags, swe_summary = check_swe_monotonicity(processed_series)
    flags.extend(swe_flags)

    # High-level flags from gaps
    if any(g.get("kind") == "duplicate" for g in gaps):
        flags.append("duplicate_timestamps_detected")
    if any(g.get("kind") == "missing" for g in gaps):
        flags.append("time_gaps_detected")

    # determinism: sort flags, stable gap ordering by (source, start)
    flags = sorted(set(flags))
    gaps_sorted = sorted(gaps, key=lambda g: (str(g.get("source")), str(g.get("t_start_utc"))))

    return {
        "integrity_flags": flags,
        "gaps": gaps_sorted,
        "swe_monotonicity": swe_summary,
    }