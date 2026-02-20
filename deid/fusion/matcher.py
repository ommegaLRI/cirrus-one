"""
deid.fusion.matcher
-------------------

Match authoritative events to particle rows.

Matching rules (v1):
    |Δt| <= tolerance_seconds
    spatial distance <= tolerance_px

Many-to-many allowed.
No corrections applied.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _euclid(y0: float, x0: float, y1: float, x1: float) -> float:
    return float(np.hypot(y0 - y1, x0 - x1))


def match_events(
    *,
    event_df: pd.DataFrame,
    particle_df: Optional[pd.DataFrame],
    alignment_payload: Dict[str, Any],
    tolerance_seconds: float,
    tolerance_px: float,
) -> List[Dict[str, Any]]:
    """
    Returns list of match dicts (not yet table-shaped).
    """

    if particle_df is None or particle_df.empty:
        return []

    # Build mapping frame_idx -> timestamp from alignment payload
    timebase = alignment_payload.get("frame_timebase", {})
    frame_ts = timebase.get("frame_timestamps_utc")

    if frame_ts is None:
        # cannot compute Δt if timestamps missing
        return []

    matches: List[Dict[str, Any]] = []

    # Pre-extract particle data arrays
    part_times = particle_df["t_utc"].values
    part_y = particle_df["y_px"].values
    part_x = particle_df["x_px"].values
    part_ids = particle_df["particle_event_id"].values

    for _, ev in event_df.iterrows():
        fp = int(ev["frame_peak"])
        if fp < 0 or fp >= len(frame_ts):
            continue

        t_ev = np.datetime64(frame_ts[fp])
        cy = float(ev["centroid_start_y"])
        cx = float(ev["centroid_start_x"])

        # vectorized candidate filtering
        dt_s = np.abs((part_times - t_ev) / np.timedelta64(1, "s"))
        cand_idx = np.where(dt_s <= tolerance_seconds)[0]

        for i in cand_idx:
            dxy = _euclid(cy, cx, float(part_y[i]), float(part_x[i]))
            if dxy <= tolerance_px:
                conf = float(
                    max(
                        0.0,
                        1.0
                        - 0.5 * (dt_s[i] / (tolerance_seconds + 1e-9))
                        - 0.5 * (dxy / (tolerance_px + 1e-9)),
                    )
                )

                matches.append(
                    {
                        "event_id": str(ev["event_id"]),
                        "particle_event_id": str(part_ids[i]),
                        "match_confidence": conf,
                        "dt_s": float(dt_s[i]),
                        "dxy_px": float(dxy),
                    }
                )

    return matches