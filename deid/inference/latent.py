"""
deid.inference.latent
---------------------

Latent trajectory inference over time bins using a linear Gaussian state-space model.

v1 model:
    z_t = z_{t-1} + w_t,      w_t ~ N(0, Q)
    y_t = H z_t + v_t,        v_t ~ N(0, R)

Where y_t is a vector of observed proxies computed from Tier-2 products:
    - baseline drift proxy: B(t) or ΔB(t) (from PlateState baseline_B_t)
    - event rate proxy: count of events per bin (from EventCatalog)
    - reconstructed SWE_rate proxy (from SWEProducts)
    - processed SWE proxy (optional; observation only, never a controller)

We implement:
    - feature aggregation into bins (e.g., 10s)
    - Kalman filter
    - RTS smoother
    - QC penalty integration (inflate R)

Outputs (payload dict for latent.json):
    - time bins (t_utc)
    - filtered mean/cov
    - smoothed mean/cov
    - observation series y_t
    - model parameters and spec
    - QC penalties applied

Determinism:
    - No randomness in v1.

Dependencies:
    - numpy, pandas
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd


# -------------------------
# Utilities
# -------------------------

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _iso_series(ts: pd.Series) -> List[str]:
    return [t.isoformat() if pd.notna(t) else None for t in pd.to_datetime(ts, utc=True, errors="coerce")]


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


# -------------------------
# Binning / observation construction
# -------------------------

def build_time_bins(
    frame_timebase: Dict[str, Any],
    *,
    bin_seconds: int,
) -> pd.DatetimeIndex:
    """
    Build bin edges/centers from frame timestamps.

    Requirements:
    - frame_timebase["frame_timestamps_utc"] must exist.
    - Timestamps must be ISO UTC or convertible.

    Notes:
    - Uses lowercase 's' for pandas frequency (pandas >= 2.x).
    - Returns left-closed bins covering [t0, t1].
    """

    ts = frame_timebase.get("frame_timestamps_utc")
    if ts is None:
        raise ValueError(
            "latent v1 requires frame_timestamps_utc in alignment frame_timebase"
        )

    # ---- Parse timestamps safely ----
    t = (
        pd.to_datetime(pd.Series(ts), utc=True, errors="coerce")
        .dropna()
        .sort_values()
    )

    if len(t) < 2:
        raise ValueError("insufficient frame timestamps for binning")

    t0 = t.iloc[0]
    t1 = t.iloc[-1]

    # ---- Normalize to second resolution (pandas >=2.x safe) ----
    t0 = t0.floor("s")
    t1 = t1.ceil("s")

    # ---- Build bins ----
    bins = pd.date_range(
        start=t0,
        end=t1 + pd.Timedelta(seconds=int(bin_seconds)),
        freq=f"{int(bin_seconds)}s",   # lowercase s fixes your latent error
        tz="UTC",
    )

    return bins


def bin_baseline_proxy(
    baseline_B_t: np.ndarray,
    frame_timebase: Dict[str, Any],
    bins: pd.DatetimeIndex,
    *,
    proxy: str = "level",
) -> pd.Series:
    """
    baseline proxy per bin:
      - "level": median baseline within bin
      - "diff": median first difference baseline within bin
    """
    ts = pd.to_datetime(pd.Series(frame_timebase["frame_timestamps_utc"]), utc=True, errors="coerce")
    df = pd.DataFrame({"t_utc": ts, "B": pd.to_numeric(pd.Series(baseline_B_t), errors="coerce")}).dropna()

    if proxy == "diff":
        df["B"] = df["B"].diff()
        df = df.dropna()

    # bin by t_utc
    df["bin"] = pd.cut(df["t_utc"], bins=bins, right=False)
    out = df.groupby("bin")["B"].median()
    # align to bins-1 (since pd.cut yields intervals)
    out.index = out.index.astype(str)
    # reconstruct aligned series with bin start labels
    # We'll use bin starts as canonical x-axis; length = len(bins)-1
    bin_starts = bins[:-1]
    vals = []
    for i in range(len(bin_starts)):
        key = str(pd.Interval(left=bins[i], right=bins[i+1], closed="left"))
        vals.append(_safe_float(out.get(key)))
    return pd.Series(vals, index=bin_starts, name="baseline_proxy")


def bin_event_rate(
    event_df: pd.DataFrame,
    alignment_payload: Dict[str, Any],
    bins: pd.DatetimeIndex,
) -> pd.Series:
    """
    Event rate proxy: count of events per bin using event frame_peak -> timestamp.
    """
    ft = alignment_payload.get("frame_timebase", {}) or {}
    ts = ft.get("frame_timestamps_utc")
    if ts is None:
        # fallback to t_peak_utc if present
        t_ev = pd.to_datetime(event_df.get("t_peak_utc"), utc=True, errors="coerce")
    else:
        ts_ser = pd.to_datetime(pd.Series(ts), utc=True, errors="coerce")
        fp = pd.to_numeric(event_df["frame_peak"], errors="coerce").fillna(-1).astype(int)
        t_list = []
        for i in fp.tolist():
            t_list.append(ts_ser.iloc[i] if 0 <= i < len(ts_ser) else pd.NaT)
        t_ev = pd.to_datetime(pd.Series(t_list), utc=True, errors="coerce")

    t_ev = t_ev.dropna()
    if len(t_ev) == 0:
        return pd.Series([0.0] * (len(bins) - 1), index=bins[:-1], name="event_rate")

    cats = pd.cut(t_ev, bins=bins, right=False)
    counts = cats.value_counts().sort_index()
    # align to bins-1
    vals = []
    for i in range(len(bins) - 1):
        key = pd.Interval(left=bins[i], right=bins[i+1], closed="left")
        vals.append(float(counts.get(key, 0)))
    return pd.Series(vals, index=bins[:-1], name="event_rate")


def bin_swe_rate(
    swe_products_df: Optional[pd.DataFrame],
    bins: pd.DatetimeIndex,
    *,
    column: str = "rate_robust_mmhr",
) -> pd.Series:
    """
    SWE rate proxy per bin: median of chosen estimator within bin.
    """
    if swe_products_df is None or swe_products_df.empty:
        return pd.Series([np.nan] * (len(bins) - 1), index=bins[:-1], name="swe_rate")

    if "t_utc" not in swe_products_df.columns:
        return pd.Series([np.nan] * (len(bins) - 1), index=bins[:-1], name="swe_rate")

    df = swe_products_df.copy()
    df["t_utc"] = pd.to_datetime(df["t_utc"], utc=True, errors="coerce")
    if column not in df.columns:
        return pd.Series([np.nan] * (len(bins) - 1), index=bins[:-1], name="swe_rate")

    df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["t_utc"])

    df["bin"] = pd.cut(df["t_utc"], bins=bins, right=False)
    out = df.groupby("bin")[column].median()

    vals = []
    for i in range(len(bins) - 1):
        key = pd.Interval(left=bins[i], right=bins[i+1], closed="left")
        vals.append(_safe_float(out.get(key)))
    return pd.Series(vals, index=bins[:-1], name="swe_rate")


def bin_processed_swe(
    processed_df: Optional[pd.DataFrame],
    bins: pd.DatetimeIndex,
    *,
    column: str = "swe_mm",
) -> pd.Series:
    """
    Processed SWE proxy per bin (optional observation): median swe_mm in bin.
    """
    if processed_df is None or processed_df.empty:
        return pd.Series([np.nan] * (len(bins) - 1), index=bins[:-1], name="processed_swe")

    if "t_utc" not in processed_df.columns or column not in processed_df.columns:
        return pd.Series([np.nan] * (len(bins) - 1), index=bins[:-1], name="processed_swe")

    df = processed_df.copy()
    df["t_utc"] = pd.to_datetime(df["t_utc"], utc=True, errors="coerce")
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["t_utc"])

    df["bin"] = pd.cut(df["t_utc"], bins=bins, right=False)
    out = df.groupby("bin")[column].median()

    vals = []
    for i in range(len(bins) - 1):
        key = pd.Interval(left=bins[i], right=bins[i+1], closed="left")
        vals.append(_safe_float(out.get(key)))
    return pd.Series(vals, index=bins[:-1], name="processed_swe")


# -------------------------
# Kalman filter / RTS smoother
# -------------------------

@dataclass(frozen=True)
class KalmanResult:
    x_filt: np.ndarray      # (T, d)
    P_filt: np.ndarray      # (T, d, d)
    x_pred: np.ndarray      # (T, d)
    P_pred: np.ndarray      # (T, d, d)
    x_smooth: np.ndarray    # (T, d)
    P_smooth: np.ndarray    # (T, d, d)
    loglik: float


def kalman_filter_rts(
    y: np.ndarray,              # (T, m) with nan allowed (missing obs)
    A: np.ndarray,              # (d, d)
    H: np.ndarray,              # (m, d)
    Q: np.ndarray,              # (d, d)
    R: np.ndarray,              # (m, m)
    x0: np.ndarray,             # (d,)
    P0: np.ndarray,             # (d, d)
) -> KalmanResult:
    """
    Handles missing observations by skipping update for rows with all-nan,
    or using sub-selection for partially observed y_t.
    """
    T = y.shape[0]
    d = A.shape[0]
    m = y.shape[1]

    x_pred = np.zeros((T, d), dtype=np.float64)
    P_pred = np.zeros((T, d, d), dtype=np.float64)
    x_filt = np.zeros((T, d), dtype=np.float64)
    P_filt = np.zeros((T, d, d), dtype=np.float64)

    loglik = 0.0

    x_prev = x0.astype(np.float64)
    P_prev = P0.astype(np.float64)

    I = np.eye(d)

    for t in range(T):
        # predict
        xp = A @ x_prev
        Pp = A @ P_prev @ A.T + Q

        x_pred[t] = xp
        P_pred[t] = Pp

        yt = y[t]
        obs = np.isfinite(yt)
        if not np.any(obs):
            # no update
            x_new = xp
            P_new = Pp
        else:
            # subselect observed dims
            y_o = yt[obs]
            H_o = H[obs, :]
            R_o = R[np.ix_(obs, obs)]

            S = H_o @ Pp @ H_o.T + R_o
            # stable solve
            K = Pp @ H_o.T @ np.linalg.inv(S)

            innov = y_o - (H_o @ xp)
            x_new = xp + K @ innov
            P_new = (I - K @ H_o) @ Pp

            # log-likelihood contribution
            try:
                sign, logdet = np.linalg.slogdet(S)
                if sign > 0:
                    ll = -0.5 * (len(y_o) * np.log(2 * np.pi) + logdet + innov.T @ np.linalg.inv(S) @ innov)
                    loglik += float(ll)
            except Exception:
                pass

        x_filt[t] = x_new
        P_filt[t] = P_new
        x_prev, P_prev = x_new, P_new

    # RTS smoother
    x_smooth = np.zeros_like(x_filt)
    P_smooth = np.zeros_like(P_filt)
    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]

    for t in range(T - 2, -1, -1):
        Pp_next = P_pred[t + 1]
        # smoother gain
        C = P_filt[t] @ A.T @ np.linalg.inv(Pp_next)
        x_smooth[t] = x_filt[t] + C @ (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = P_filt[t] + C @ (P_smooth[t + 1] - Pp_next) @ C.T

    return KalmanResult(
        x_filt=x_filt,
        P_filt=P_filt,
        x_pred=x_pred,
        P_pred=P_pred,
        x_smooth=x_smooth,
        P_smooth=P_smooth,
        loglik=float(loglik),
    )


# -------------------------
# Public API
# -------------------------

def run_latent_inference(
    *,
    alignment_payload: Dict[str, Any],
    plate_state_npz: np.lib.npyio.NpzFile,
    event_df: pd.DataFrame,
    swe_products_df: Optional[pd.DataFrame],
    processed_df: Optional[pd.DataFrame],
    qc_penalties: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build observations, run Kalman+RTS, return latent.json payload dict.

    Config (v1):
      bin_seconds: int (default 10)
      baseline_proxy: "level"|"diff" (default "diff")
      include_processed_swe: bool (default True if processed present)
      state_dim: int (default 1)
      obs_spec: list[str] subset of ["baseline", "event_rate", "swe_rate", "processed_swe"]
      Q_scale: float (default 1.0)
      R_scale: float (default 1.0)
      x0: float (default 0.0)
      P0: float (default 1.0)

    QC penalties:
      - if gating_penalty exists (<1), inflate R by 1/(penalty^2)
    """
    cfg = dict(config or {})
    bin_seconds = int(cfg.get("bin_seconds", 10))
    baseline_proxy = str(cfg.get("baseline_proxy", "diff")).lower()
    state_dim = int(cfg.get("state_dim", 1))

    obs_spec = cfg.get("obs_spec", ["baseline", "event_rate", "swe_rate", "processed_swe"])
    obs_spec = [str(s) for s in obs_spec]

    include_processed_swe = bool(cfg.get("include_processed_swe", True))
    if processed_df is None:
        include_processed_swe = False

    # Build bins
    bins = build_time_bins(alignment_payload.get("frame_timebase", {}), bin_seconds=bin_seconds)

    # Observations
    B = plate_state_npz["baseline_B_t"]
    y_baseline = bin_baseline_proxy(B, alignment_payload["frame_timebase"], bins, proxy=baseline_proxy)
    y_rate = bin_event_rate(event_df, alignment_payload, bins)
    y_swe_rate = bin_swe_rate(swe_products_df, bins, column=str(cfg.get("swe_rate_column", "rate_robust_mmhr")))
    y_proc = bin_processed_swe(processed_df, bins) if include_processed_swe else pd.Series([np.nan] * (len(bins) - 1), index=bins[:-1])

    obs_map = {
        "baseline": y_baseline,
        "event_rate": y_rate,
        "swe_rate": y_swe_rate,
        "processed_swe": y_proc,
    }

    used = [k for k in obs_spec if k in obs_map]
    m = len(used)
    T = len(bins) - 1

    Y = np.zeros((T, m), dtype=np.float64)
    for j, name in enumerate(used):
        Y[:, j] = pd.to_numeric(obs_map[name], errors="coerce").to_numpy(dtype=np.float64)

    # Model matrices (v1): 1D latent with linear observation weights
    d = state_dim
    if d != 1:
        raise ValueError("v1 latent model supports state_dim=1 only (extend later with proper matrix specs)")

    A = np.eye(1)
    # Observation matrix: learn fixed weights? v1 uses hand-set weights = 1 for all proxies
    H = np.ones((m, 1), dtype=np.float64)

    Q_scale = float(cfg.get("Q_scale", 1.0))
    R_scale = float(cfg.get("R_scale", 1.0))

    Q = np.array([[Q_scale]], dtype=np.float64)

    # Observation noise: diagonal, scaled; per-observation additional scaling can be configured later
    R = np.eye(m, dtype=np.float64) * R_scale

    # QC penalty integration: inflate R
    penalties = dict(qc_penalties or {})
    gp = penalties.get("gating_penalty", None)
    if gp is not None:
        gp = float(gp)
        if gp > 0 and gp < 1:
            inflate = 1.0 / (gp * gp)
            R = R * inflate
            penalties["R_inflation_from_gating"] = float(inflate)

    x0 = np.array([float(cfg.get("x0", 0.0))], dtype=np.float64)
    P0 = np.array([[float(cfg.get("P0", 1.0))]], dtype=np.float64)

    res = kalman_filter_rts(Y, A=A, H=H, Q=Q, R=R, x0=x0, P0=P0)

    # Package outputs
    t_centers = bins[:-1] + (bins[1:] - bins[:-1]) / 2

    payload = {
        "schema_version": "latent_v1",
        "bin_seconds": int(bin_seconds),
        "time_utc": _iso_series(pd.Series(t_centers)),
        "observations": {
            "names": used,
            "y": Y.tolist(),
        },
        "model": {
            "type": "linear_gaussian_kalman_rts",
            "state_dim": 1,
            "A": A.tolist(),
            "H": H.tolist(),
            "Q": Q.tolist(),
            "R": R.tolist(),
            "x0": x0.tolist(),
            "P0": P0.tolist(),
            "baseline_proxy": baseline_proxy,
            "include_processed_swe": include_processed_swe,
            "notes": "v1: fixed observation weights, diagonal R, Q random-walk; extend with configurable matrices later",
        },
        "results": {
            "filtered_mean": res.x_filt.tolist(),
            "filtered_var": [float(P[0, 0]) for P in res.P_filt],
            "smoothed_mean": res.x_smooth.tolist(),
            "smoothed_var": [float(P[0, 0]) for P in res.P_smooth],
            "loglik": float(res.loglik),
        },
        "qc_penalties": penalties,
    }
    return payload