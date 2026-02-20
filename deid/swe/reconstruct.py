"""
deid.swe.reconstruct
--------------------

Reconstruct SWE from authoritative events.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd

from deid.core.units import mass_mg_to_swe_mm


def reconstruct_swe_series(
    *,
    event_df: pd.DataFrame,
    frame_timebase: Dict[str, Any],
    sensing_area_mm2: float,
) -> pd.DataFrame:
    """
    Returns DataFrame with:
        frame_idx
        t_utc
        swe_reconstructed_mm
    """

    frame_ts = frame_timebase.get("frame_timestamps_utc")
    if frame_ts is None:
        raise ValueError("Frame timestamps required for SWE reconstruction")

    n_frames = len(frame_ts)
    swe_inc = np.zeros((n_frames,), dtype=np.float64)

    # events contribute at peak frame
    for _, ev in event_df.iterrows():
        f = int(ev["frame_peak"])
        mass = float(ev.get("mass_mg_authoritative", 0.0) or 0.0)
        swe_inc[f] += mass_mg_to_swe_mm(mass, sensing_area_mm2)

    swe_cum = np.cumsum(swe_inc)

    df = pd.DataFrame(
        {
            "frame_idx": np.arange(n_frames, dtype=np.int32),
            "t_utc": pd.to_datetime(frame_ts),
            "swe_reconstructed_mm": swe_cum,
        }
    )
    return df