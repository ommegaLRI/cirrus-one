"""
deid.inference.phenotype_vector
-------------------------------

Build a fixed-length, run-level phenotype vector for cohort ML / indexing.

Consumes Tier-2 artifacts (payloads / tables):
- instrument_health payload
- closure_report payload
- event_catalog.parquet
- swe_products.parquet (optional)
- regimes payload (optional)
- plate_state baseline drift stats (optional via plate_state.npz)

Outputs:
- phenotype_vector payload dict (JSON-serializable)

Design:
- Deterministic
- Explicit feature names
- Safe when inputs missing (NaN / None)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (float, int)):
            return float(x)
        if isinstance(x, str) and x.strip() == "":
            return None
        return float(x)
    except Exception:
        return None


def _quantiles(x: np.ndarray, qs=(0.1, 0.5, 0.9)) -> Dict[str, Optional[float]]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"q{int(q*100)}": None for q in qs}
    vals = np.quantile(x, qs)
    return {f"q{int(q*100)}": float(v) for q, v in zip(qs, vals)}


def phenotype_from_events(event_df: pd.DataFrame) -> Dict[str, Any]:
    if event_df is None or event_df.empty:
        return {
            "n_events": 0,
            "snr": {"q10": None, "q50": None, "q90": None},
            "delta_peak": {"q10": None, "q50": None, "q90": None},
            "energy_proxy_E": {"q10": None, "q50": None, "q90": None},
            "area_peak_px": {"q10": None, "q50": None, "q90": None},
            "duration_s": {"q10": None, "q50": None, "q90": None},
            "motion_score_mean": None,
            "edge_proximity_mean": None,
        }

    def col(name: str) -> np.ndarray:
        if name not in event_df.columns:
            return np.array([], dtype=float)
        return pd.to_numeric(event_df[name], errors="coerce").to_numpy(dtype=np.float64)

    snr = col("snr")
    dp = col("delta_peak")
    E = col("energy_proxy_E")
    area = col("area_peak_px")
    dur = col("duration_s")
    motion = col("motion_score")
    edge = col("edge_proximity")

    out = {
        "n_events": int(len(event_df)),
        "snr": _quantiles(snr),
        "delta_peak": _quantiles(dp),
        "energy_proxy_E": _quantiles(E),
        "area_peak_px": _quantiles(area),
        "duration_s": _quantiles(dur),
        "motion_score_mean": float(np.nanmean(motion)) if np.isfinite(motion).any() else None,
        "edge_proximity_mean": float(np.nanmean(edge)) if np.isfinite(edge).any() else None,
    }
    return out


def phenotype_from_rates(swe_products_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    if swe_products_df is None or swe_products_df.empty:
        return {
            "swe_reconstructed_final_mm": None,
            "rate_windowed": {"q10": None, "q50": None, "q90": None},
            "rate_robust": {"q10": None, "q50": None, "q90": None},
            "rate_event_based": {"q10": None, "q50": None, "q90": None},
        }

    swe_final = None
    if "swe_reconstructed_mm" in swe_products_df.columns:
        s = pd.to_numeric(swe_products_df["swe_reconstructed_mm"], errors="coerce")
        swe_final = _safe_float(s.dropna().iloc[-1]) if len(s.dropna()) > 0 else None

    def qcol(name: str) -> Dict[str, Optional[float]]:
        if name not in swe_products_df.columns:
            return {"q10": None, "q50": None, "q90": None}
        x = pd.to_numeric(swe_products_df[name], errors="coerce").to_numpy(dtype=np.float64)
        return _quantiles(x)

    return {
        "swe_reconstructed_final_mm": swe_final,
        "rate_windowed": qcol("rate_windowed_mmhr"),
        "rate_robust": qcol("rate_robust_mmhr"),
        "rate_event_based": qcol("rate_event_based_mmhr"),
    }


def phenotype_from_regimes(regimes_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pull regime fractions if regimes exist; else empty.
    """
    if not regimes_payload:
        return {"regime_fractions": {}}
    frac = regimes_payload.get("fractions", {}) or {}
    # ensure floats
    clean = {str(k): _safe_float(v) for k, v in frac.items()}
    return {"regime_fractions": clean}


def build_phenotype_vector(
    *,
    instrument_health: Dict[str, Any],
    closure_report: Dict[str, Any],
    event_df: pd.DataFrame,
    swe_products_df: Optional[pd.DataFrame] = None,
    regimes_payload: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Returns JSON-serializable phenotype vector.

    This is a *feature contract*; keep names stable and versioned.
    """
    health_score = _safe_float(instrument_health.get("overall_score"))
    closure_score = _safe_float(closure_report.get("closure_score"))

    events_block = phenotype_from_events(event_df)
    rates_block = phenotype_from_rates(swe_products_df)
    regimes_block = phenotype_from_regimes(regimes_payload)

    vec = {
        "schema_version": "phenotype_vector_v1",
        "health_score": health_score,
        "closure_score": closure_score,
        "events": events_block,
        "rates": rates_block,
        "regimes": regimes_block,
        "extra": dict(extra or {}),
    }
    return vec