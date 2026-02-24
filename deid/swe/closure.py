"""
deid.swe.closure
----------------

Compute closure metrics comparing reconstructed vs processed SWE.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd


def _normalized_rmse(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return 1.0
    err = np.sqrt(np.mean((a - b) ** 2))
    scale = max(np.max(np.abs(b)), 1e-6)
    return float(err / scale)


def compute_closure(
    swe_df: pd.DataFrame,
    processed_df: pd.DataFrame | None,
) -> Dict[str, Any]:
    if processed_df is None or processed_df.empty:
        return {
            "closure_score": None,
            "residual_stats": {},
            "attribution": {},
            "failure_modes": ["no_processed_series"],
            "recommendations": [],
        }

    merged = pd.merge_asof(
        swe_df.sort_values("t_utc"),
        processed_df.sort_values("t_utc"),
        on="t_utc",
        direction="nearest",
    )

    a = merged["swe_reconstructed_mm"].fillna(0.0).values
    # Accept canonical v1 name first
    if "swe_processed_mm" in merged.columns:
        b = merged["swe_processed_mm"].fillna(0.0).values
    elif "swe_mm" in merged.columns:
        # backward compatibility fallback
        b = merged["swe_mm"].fillna(0.0).values
    else:
        raise KeyError("No processed SWE column found (expected swe_processed_mm)")

    resid = a - b
    nrmse = _normalized_rmse(a, b)

    closure_score = float(max(0.0, 1.0 - nrmse))

    return {
        "closure_score": closure_score,
        "residual_stats": {
            "mean": float(np.mean(resid)),
            "std": float(np.std(resid)),
        },
        "attribution": {},
        "failure_modes": [],
        "recommendations": [],
    }