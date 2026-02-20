"""
deid.swe.rate
-------------

Multiple SWE rate estimators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def windowed_rate(swe: pd.Series, dt_seconds: float, window: int = 5) -> pd.Series:
    if dt_seconds <= 0:
        return pd.Series([None] * len(swe))
    diff = swe.diff(window)
    rate = diff / (window * dt_seconds) * 3600.0
    return rate


def robust_rate(swe: pd.Series, dt_seconds: float) -> pd.Series:
    if dt_seconds <= 0:
        return pd.Series([None] * len(swe))
    # simple median derivative
    ds = swe.diff().rolling(5, center=True).median()
    return ds / dt_seconds * 3600.0


def event_based_rate(event_df: pd.DataFrame, n_frames: int, dt_seconds: float) -> pd.Series:
    r = np.zeros((n_frames,), dtype=float)
    if dt_seconds <= 0:
        return pd.Series([None] * n_frames)
    for _, ev in event_df.iterrows():
        f = int(ev["frame_peak"])
        m = float(ev.get("mass_mg_authoritative", 0.0) or 0.0)
        r[f] += m
    return pd.Series(r / dt_seconds * 3600.0)