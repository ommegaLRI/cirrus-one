"""
deid.inference.regimes
----------------------

Unsupervised regime discovery from authoritative EventCatalog.

v1:
- Build feature matrix from event_catalog.parquet
- Standardize (z-score)
- Cluster with:
    - sklearn KMeans if available
    - fallback: deterministic numpy k-means
- Emit:
    - regimes.json (summary + per-time-bin regime fractions)
    - regime_events.parquet (event_id -> regime + probabilities/confidence)

Notes:
- No claims beyond structure discovery.
- Deterministic given seed and config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


FEATURE_COLUMNS_DEFAULT = [
    "energy_proxy_E",
    "delta_peak",
    "area_peak_px",
    "duration_s",
    "motion_score",
    "edge_proximity",
    "snr",
]


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([np.nan] * len(df))
    return pd.to_numeric(df[col], errors="coerce")


def _standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score standardization with nan-safe handling (nan -> column median).
    Returns standardized X, mean, std.
    """
    X = X.astype(np.float64, copy=True)
    # impute nan with column median
    for j in range(X.shape[1]):
        col = X[:, j]
        med = np.nanmedian(col)
        col[np.isnan(col)] = med if np.isfinite(med) else 0.0
        X[:, j] = col
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    return Z, mu, sd


def _kmeans_numpy(Z: np.ndarray, k: int, seed: int, n_iter: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic numpy k-means fallback.
    Returns (labels, centers).
    """
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    # init centers from random unique points
    idx = rng.choice(n, size=min(k, n), replace=False)
    centers = Z[idx].copy()
    labels = np.zeros((n,), dtype=np.int32)

    for _ in range(n_iter):
        # assign
        d2 = ((Z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(d2, axis=1).astype(np.int32)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # update
        for j in range(centers.shape[0]):
            pts = Z[labels == j]
            if len(pts) > 0:
                centers[j] = pts.mean(axis=0)

    return labels, centers


def _kmeans(Z: np.ndarray, k: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use sklearn KMeans if available, else fallback numpy.
    """
    try:
        from sklearn.cluster import KMeans  # type: ignore

        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(Z).astype(np.int32)
        centers = km.cluster_centers_.astype(np.float64)
        return labels, centers
    except Exception:
        return _kmeans_numpy(Z, k=k, seed=seed)


def _event_times_from_alignment(event_df: pd.DataFrame, alignment_payload: Dict[str, Any]) -> pd.Series:
    """
    Build event timestamp series based on frame_peak -> frame_timestamps_utc.
    """
    ft = alignment_payload.get("frame_timebase", {}) or {}
    ts = ft.get("frame_timestamps_utc")
    if ts is None:
        # fallback: if t_peak_utc exists, use it
        if "t_peak_utc" in event_df.columns:
            return pd.to_datetime(event_df["t_peak_utc"], utc=True, errors="coerce")
        return pd.to_datetime(pd.Series([pd.NaT] * len(event_df)), utc=True)

    ts_arr = pd.to_datetime(pd.Series(ts), utc=True, errors="coerce")
    fp = pd.to_numeric(event_df["frame_peak"], errors="coerce").fillna(-1).astype(int)
    out = []
    for i in fp.tolist():
        if 0 <= i < len(ts_arr):
            out.append(ts_arr.iloc[i])
        else:
            out.append(pd.NaT)
    return pd.to_datetime(pd.Series(out), utc=True, errors="coerce")


def build_feature_matrix(
    event_df: pd.DataFrame,
    *,
    feature_columns: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    cols = feature_columns or FEATURE_COLUMNS_DEFAULT
    X_cols = []
    for c in cols:
        s = _safe_col(event_df, c)
        X_cols.append(s.to_numpy(dtype=np.float64))
    X = np.vstack(X_cols).T if X_cols else np.zeros((len(event_df), 0), dtype=np.float64)
    return X, cols


def regime_fractions_over_time(
    event_times: pd.Series,
    labels: np.ndarray,
    *,
    bin_seconds: int,
) -> Dict[str, Any]:
    """
    Compute regime fractions per time bin.
    Returns dict with bins and fractions.
    """
    t = pd.to_datetime(event_times, utc=True, errors="coerce")
    valid = ~t.isna()
    if valid.sum() == 0:
        return {"bin_seconds": int(bin_seconds), "bins": [], "fractions": {}}

    t0 = t[valid].min()
    # integer bin id
    dt_s = ((t[valid] - t0).dt.total_seconds()).to_numpy(dtype=float)
    bin_id = np.floor(dt_s / float(bin_seconds)).astype(int)

    k = int(np.max(labels) + 1) if labels.size > 0 else 0
    bins = sorted(set(bin_id.tolist()))
    frac = {str(r): [] for r in range(k)}

    for b in bins:
        idx = np.where(bin_id == b)[0]
        lab = labels[valid.to_numpy()][idx]
        n = len(lab)
        for r in range(k):
            frac[str(r)].append(float(np.sum(lab == r) / max(1, n)))

    bin_starts = [(t0 + pd.Timedelta(seconds=int(b) * int(bin_seconds))).isoformat() for b in bins]
    return {"bin_seconds": int(bin_seconds), "bins": bin_starts, "fractions": frac}


def run_regime_discovery(
    *,
    event_df: pd.DataFrame,
    alignment_payload: Dict[str, Any],
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main entrypoint for regimes.

    Config (v1):
      n_regimes: int (default 4)
      seed: int (default 0)
      feature_columns: list[str] (optional)
      time_bin_seconds: int (default 60)

    Returns:
      regime_events_df (for parquet)
      regimes_json_payload (for regimes.json payload)
    """
    n_regimes = int(config.get("n_regimes", 4))
    seed = int(config.get("seed", 0))
    time_bin_seconds = int(config.get("time_bin_seconds", 60))
    feature_columns = config.get("feature_columns", None)

    if len(event_df) == 0:
        empty_df = pd.DataFrame(columns=["event_id", "regime", "regime_confidence", "schema_version"])
        payload = {
            "schema_version": "regimes_v1",
            "n_events": 0,
            "n_regimes": n_regimes,
            "method": "kmeans",
            "features": feature_columns or FEATURE_COLUMNS_DEFAULT,
            "notes": "no events",
        }
        return empty_df, payload

    X, used_cols = build_feature_matrix(event_df, feature_columns=feature_columns)
    Z, mu, sd = _standardize(X)

    # clamp regimes if too few events
    k = max(1, min(n_regimes, len(event_df)))
    labels, centers = _kmeans(Z, k=k, seed=seed)

    # confidence: inverse distance to assigned center (simple, monotone)
    d2 = ((Z - centers[labels]) ** 2).sum(axis=1)
    conf = 1.0 / (1.0 + d2)

    # event times for timeline summaries
    event_times = _event_times_from_alignment(event_df, alignment_payload)

    timeline = regime_fractions_over_time(event_times, labels, bin_seconds=time_bin_seconds)

    # regime summary stats
    counts = {str(i): int(np.sum(labels == i)) for i in range(k)}
    fractions = {str(i): float(counts[str(i)] / max(1, len(labels))) for i in range(k)}

    regime_events_df = pd.DataFrame(
        {
            "event_id": event_df["event_id"].astype(str).values,
            "regime": labels.astype(np.int32),
            "regime_confidence": conf.astype(np.float64),
            "event_time_utc": pd.to_datetime(event_times, utc=True, errors="coerce"),
            "schema_version": "regime_events_v1",
        }
    )

    payload = {
        "schema_version": "regimes_v1",
        "n_events": int(len(event_df)),
        "n_regimes": int(k),
        "method": "kmeans",
        "seed": int(seed),
        "features": list(used_cols),
        "standardization": {"mean": mu.tolist(), "std": sd.tolist()},
        "counts": counts,
        "fractions": fractions,
        "timeline": timeline,
        "notes": "v1 unsupervised structure discovery; interpret as regimes/phenotypes, not diagnosis",
    }

    return regime_events_df, payload