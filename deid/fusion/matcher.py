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

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _euclid(y0: float, x0: float, y1: float, x1: float) -> float:
    return float(np.hypot(y0 - y1, x0 - x1))


# -------------------------------------------------------------------
# Matching
# -------------------------------------------------------------------


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

    Applies global alignment particle time offset during matching.
    """

    # -------------------------------------------------------------
    # Guard: no particles
    # -------------------------------------------------------------
    if particle_df is None or particle_df.empty:
        return []

    # -------------------------------------------------------------
    # Frame timestamps (thermal time axis)
    # -------------------------------------------------------------
    timebase = alignment_payload.get("frame_timebase", {})
    frame_ts = timebase.get("frame_timestamps_utc")

    if frame_ts is None:
        return []

    # Always normalize to DatetimeIndex
    frame_ts = pd.to_datetime(frame_ts, utc=True)

    # -------------------------------------------------------------
    # Particle timestamps
    # -------------------------------------------------------------
    part_times = pd.to_datetime(particle_df["t_utc"], utc=True, errors="coerce")

    # Drop NaT early (prevents silent vector math corruption)
    valid_mask = ~part_times.isna()
    if not valid_mask.all():
        particle_df = particle_df.loc[valid_mask].reset_index(drop=True)
        part_times = part_times.loc[valid_mask]

    # -------------------------------------------------------------
    # APPLY GLOBAL PARTICLE CLOCK OFFSET
    # (from alignment.frame_timebase)
    # -------------------------------------------------------------
    offset_sec = (
        alignment_payload
        .get("frame_timebase", {})
        .get("particle_time_offset_seconds", 0.0)
    )

    try:
        offset_sec = float(offset_sec or 0.0)
    except Exception:
        offset_sec = 0.0

    if offset_sec != 0.0:
        part_times = part_times + pd.to_timedelta(offset_sec, unit="s")

    # Convert once to ndarray seconds during loop
    part_times = pd.Series(part_times)

    # -------------------------------------------------------------
    # Particle spatial arrays
    # -------------------------------------------------------------
    part_y = particle_df["y_px"].to_numpy(dtype=float)
    part_x = particle_df["x_px"].to_numpy(dtype=float)
    part_ids = particle_df["particle_event_id"].astype(str).to_numpy()

    matches: List[Dict[str, Any]] = []

    # -------------------------------------------------------------
    # Iterate events
    # -------------------------------------------------------------
    for _, ev in event_df.iterrows():

        # Use start frame if present (better temporal anchor)
        fp = int(ev["frame_start"]) if "frame_start" in ev else int(ev["frame_peak"])

        if fp < 0 or fp >= len(frame_ts):
            continue

        t_ev = frame_ts[fp]

        cy = float(ev["centroid_start_y"])
        cx = float(ev["centroid_start_x"])

        # ---------------------------------------------------------
        # Vectorized Δt (seconds)
        # ---------------------------------------------------------
        dt_s = np.abs((part_times - t_ev).dt.total_seconds().to_numpy())

        cand_idx = np.where(dt_s <= tolerance_seconds)[0]

        if len(cand_idx) == 0:
            continue

        # ---------------------------------------------------------
        # Spatial + temporal gating
        # ---------------------------------------------------------
        for i in cand_idx:

            dxy = float(np.hypot(cy - part_y[i], cx - part_x[i]))

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
                        "particle_event_id": part_ids[i],
                        "match_confidence": conf,
                        "dt_s": float(dt_s[i]),
                        "dxy_px": float(dxy),
                    }
                )

    return matches