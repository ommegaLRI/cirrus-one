"""
deid.fusion.matcher
-------------------

Match authoritative events to particle rows.

Matching rules (v1):

    |Δframe| <= tolerance_frames
    spatial distance <= tolerance_px

where:

    tolerance_frames = tolerance_seconds / dt_seconds

Frame indices are authoritative. UTC timestamps are only used
for reporting dt_s when available.

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


def _particle_offset_for_reporting(alignment_payload: Dict[str, Any]) -> float:
    """
    Determine particle→thermal offset for timestamp reporting only.

    Under the repaired alignment contract:

        particle_offset_application == "alignment_mapping_only"

    means the offset was applied inside alignment mapping and should
    NOT be applied to particle_df itself, but we must apply it when
    computing dt_s so timestamps align with event UTCs.
    """
    frame_tb = alignment_payload.get("frame_timebase", {}) or {}

    if frame_tb.get("particle_offset_application") == "alignment_mapping_only":
        return float(frame_tb.get("particle_time_offset_seconds", 0.0) or 0.0)

    offsets = alignment_payload.get("offsets_seconds", {}) or {}
    return float(offsets.get("particle_vs_thermal", 0.0) or 0.0)


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
    """

    print("\n========== MATCHER DEBUG START ==========")

    # -------------------------------------------------------------
    # Guard: no particles
    # -------------------------------------------------------------
    if particle_df is None or particle_df.empty:
        print("DEBUG: particle_df empty")
        return []

    # -------------------------------------------------------------
    # Alignment data
    # -------------------------------------------------------------
    timebase = alignment_payload.get("frame_timebase", {})
    particle_to_frame = alignment_payload.get("particle_to_frame", {})

    dt_seconds = timebase.get("dt_seconds")

    print("DEBUG: dt_seconds =", dt_seconds)
    print("DEBUG: particle_to_frame size =", len(particle_to_frame))

    if dt_seconds is None or dt_seconds <= 0:
        print("DEBUG: invalid dt_seconds")
        return []

    tolerance_frames = tolerance_seconds / float(dt_seconds)

    print("DEBUG: tolerance_seconds =", tolerance_seconds)
    print("DEBUG: tolerance_frames =", tolerance_frames)

    # -------------------------------------------------------------
    # Build particle → frame mapping
    # -------------------------------------------------------------
    map_df = pd.DataFrame(
        list(particle_to_frame.items()),
        columns=["particle_event_id", "frame_idx"],
    )

    particle_df = particle_df.merge(
        map_df,
        on="particle_event_id",
        how="left",
    )

    print("DEBUG: particle_df rows after merge =", len(particle_df))

    valid_mask = particle_df["frame_idx"].notna()

    print("DEBUG: mapped particle rows =", valid_mask.sum())

    if not valid_mask.any():
        print("DEBUG: no particles mapped to frames")
        return []

    part_ids = particle_df.loc[valid_mask, "particle_event_id"].astype(str).to_numpy()
    part_frames = particle_df.loc[valid_mask, "frame_idx"].to_numpy(dtype=float)
    part_y = particle_df.loc[valid_mask, "y_px"].to_numpy(dtype=float)
    part_x = particle_df.loc[valid_mask, "x_px"].to_numpy(dtype=float)

    print("DEBUG: particle frame sample =", part_frames[:10])
    print("DEBUG: particle id sample =", part_ids[:10])

    valid_mask = ~np.isnan(part_frames)

    if not valid_mask.any():
        print("DEBUG: particle frames all NaN")
        return []

    part_ids = part_ids[valid_mask]
    part_y = part_y[valid_mask]
    part_x = part_x[valid_mask]
    part_frames = part_frames[valid_mask]

    # -------------------------------------------------------------
    # Optional timestamps (for dt_s reporting)
    # -------------------------------------------------------------
    part_times = None

    if "t_utc" in particle_df.columns:
        part_times = pd.to_datetime(
            particle_df.loc[valid_mask, "t_utc"],
            utc=True,
            errors="coerce",
        )

        offset_s = _particle_offset_for_reporting(alignment_payload)

        if offset_s != 0.0:
            part_times = part_times + pd.to_timedelta(offset_s, unit="s")

    matches: List[Dict[str, Any]] = []

    # -------------------------------------------------------------
    # Event frame diagnostics
    # -------------------------------------------------------------
    print(
        "DEBUG: event frame range =",
        int(event_df["frame_peak"].min()),
        int(event_df["frame_peak"].max()),
    )

    print(
        "DEBUG: particle frame range =",
        int(np.min(part_frames)),
        int(np.max(part_frames)),
    )

    # -------------------------------------------------------------
    # Iterate events
    # -------------------------------------------------------------
    debug_event_counter = 0

    for _, ev in event_df.iterrows():

        fs = int(ev["frame_start"])
        fp = int(ev["frame_peak"])
        fe = int(ev["frame_end"])

        cy = float(ev["centroid_start_y"])
        cx = float(ev["centroid_start_x"])

        # ---------------------------------------------------------
        # Frame distance gating against EVENT INTERVAL, not frame_peak
        # distance = 0 for particles inside [fs, fe]
        # ---------------------------------------------------------
        dt_frames = np.where(
            part_frames < fs,
            fs - part_frames,
            np.where(part_frames > fe, part_frames - fe, 0.0),
        )

        cand_idx = np.where(dt_frames <= tolerance_frames)[0]

        # Debug first few events
        if debug_event_counter < 5:
            print("\nDEBUG EVENT", debug_event_counter)
            print("  event_id =", ev["event_id"])
            print("  frame_start, frame_peak, frame_end =", fs, fp, fe)
            print("  min_dt_frames =", float(np.min(dt_frames)))
            print("  candidate_count =", len(cand_idx))
            debug_event_counter += 1

        if len(cand_idx) == 0:
            continue

        # ---------------------------------------------------------
        # Spatial gating
        # ---------------------------------------------------------
        for i in cand_idx:

            dxy = float(np.hypot(cy - part_y[i], cx - part_x[i]))

            if dxy > tolerance_px:
                continue

            # -----------------------------------------------------
            # Compute dt_s
            # -----------------------------------------------------
            dt_s = None

            if part_times is not None and "t_peak_utc" in ev and pd.notna(ev["t_peak_utc"]):
                try:
                    t_ev = pd.to_datetime(ev["t_peak_utc"], utc=True)
                    dt_s = float(
                        abs((part_times.iloc[i] - t_ev).total_seconds())
                    )
                except Exception:
                    dt_s = float(dt_frames[i] * dt_seconds)

            else:
                dt_s = float(dt_frames[i] * dt_seconds)

            # -----------------------------------------------------
            # Confidence
            # -----------------------------------------------------
            conf = float(
                max(
                    0.0,
                    1.0
                    - 0.5 * (dt_frames[i] / (tolerance_frames + 1e-9))
                    - 0.5 * (dxy / (tolerance_px + 1e-9)),
                )
            )

            matches.append(
                {
                    "event_id": str(ev["event_id"]),
                    "particle_event_id": part_ids[i],
                    "match_confidence": conf,
                    "dt_s": float(dt_s),
                    "dxy_px": float(dxy),
                }
            )

    print("\nDEBUG: total matches =", len(matches))
    print("========== MATCHER DEBUG END ==========\n")

    return matches