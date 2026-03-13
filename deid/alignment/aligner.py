"""
deid.alignment.aligner
----------------------

Map particle events and processed rows onto thermal frame indices.

Authoritative contract (Model A):

- inputs/particle.parquet:t_utc is the ONLY canonical particle timeline
- particle_time_offset_seconds is applied exactly once during alignment mapping
- synthetic timebase anchors to RAW particle timestamps (never shifted copies)
- particle_to_frame must be reproducible from:
      raw particle t_utc
      declared particle offset
      frame_timebase
      mapping rule

Outputs:
- frame_timebase
- particle_to_frame
- processed_to_frame
- offsets_seconds (declared offsets)
- residual_offsets_seconds (measured offsets)
- gaps
- integrity_flags
- confidence
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from deid.alignment.cadence import (
    FrameTimebase,
    build_frame_timebase,
    time_utc_to_frame_index,
    frame_index_to_time_utc,
)
from deid.alignment.integrity import build_integrity_report
from deid.core.errors import AlignmentError
from deid.core.types import ThermalCubeRef


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _nearest_frame_idx(
    tb: FrameTimebase,
    t_utc: datetime,
    *,
    T: int,
    tolerance_seconds: Optional[float],
) -> Optional[int]:

    idx = time_utc_to_frame_index(tb, t_utc)

    if idx is None:
        return None

    if idx < 0 or idx >= T:
        return None

    if tolerance_seconds is None:
        return idx

    t_hat = frame_index_to_time_utc(tb, idx)

    if t_hat is None:
        return idx

    resid = abs((t_utc.astimezone(timezone.utc) - t_hat).total_seconds())

    if resid <= float(tolerance_seconds):
        return idx

    return None


def _median_offset_seconds(
    tb: FrameTimebase,
    mapping: Dict[str, int],
    times: Dict[str, datetime],
) -> Optional[float]:

    resids = []

    for key, idx in mapping.items():

        t_obs = times.get(key)
        t_hat = frame_index_to_time_utc(tb, idx)

        if t_obs is None or t_hat is None:
            continue

        resids.append((t_obs.astimezone(timezone.utc) - t_hat).total_seconds())

    if not resids:
        return None

    return float(np.median(np.array(resids, dtype=float)))


def _declared_particle_offset_seconds(config: Dict[str, Any]) -> float:
    return float(config.get("particle_time_offset_seconds", 0.0) or 0.0)


def _apply_declared_particle_offset(
    t: pd.Timestamp,
    offset_seconds: float,
) -> pd.Timestamp:

    if offset_seconds == 0.0:
        return t

    return t + pd.to_timedelta(offset_seconds, unit="s")


def _min_time_utc(
    df: Optional[pd.DataFrame],
    column: str = "t_utc",
) -> Optional[datetime]:

    if df is None or df.empty or column not in df.columns:
        return None

    s = pd.to_datetime(df[column], utc=True, errors="coerce").dropna()

    if s.empty:
        return None

    return s.min().to_pydatetime()


# -------------------------------------------------------------------
# Alignment
# -------------------------------------------------------------------


def build_session_alignment(
    thermal_ref: ThermalCubeRef,
    particle_table: Optional[pd.DataFrame],
    processed_series: Optional[pd.DataFrame],
    config: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    T, _, _ = thermal_ref.shape

    # -------------------------------------------------------------
    # Build timebase
    # -------------------------------------------------------------

    tb = build_frame_timebase(
        thermal_ref,
        particle_table,
        processed_series,
        config,
    )

    particle_offset_s = _declared_particle_offset_seconds(config)

    # -------------------------------------------------------------
    # Synthetic timebase override
    # -------------------------------------------------------------

    if config.get("force_synthetic_timebase", False):

        dt_val = float(config.get("synthetic_dt_seconds", 1.0))

        explicit_t0 = config.get("synthetic_t0_utc")

        if explicit_t0:

            t0 = datetime.fromisoformat(str(explicit_t0)).astimezone(timezone.utc)
            anchor_source = "config.synthetic_t0_utc"

        else:

            t0 = _min_time_utc(particle_table) or _min_time_utc(processed_series)

            if t0 is not None:

                if _min_time_utc(particle_table) is not None:
                    anchor_source = "particle.t_utc.min_raw"
                else:
                    anchor_source = "processed.t_utc.min"

            else:

                t0 = datetime(1970, 1, 1, tzinfo=timezone.utc)
                anchor_source = "epoch_fallback"

        frame_ts = [
            (t0 + timedelta(seconds=i * dt_val)).isoformat()
            for i in range(T)
        ]

        tb = FrameTimebase(
            t0_utc=t0.isoformat(),
            dt_seconds=dt_val,
            frame_timestamps_utc=frame_ts,
            source="synthetic_forced",
            confidence=0.5,
        )

    else:
        anchor_source = None

    # -------------------------------------------------------------
    # Cadence fallback
    # -------------------------------------------------------------

    dt_missing = (
        tb.dt_seconds is None
        or (isinstance(tb.dt_seconds, float) and np.isnan(tb.dt_seconds))
    )

    if dt_missing:

        try:

            fallback_dt = float(config.get("fallback_dt_seconds", 1.0))

            tb = FrameTimebase(
                t0_utc=tb.t0_utc,
                dt_seconds=fallback_dt,
                frame_timestamps_utc=tb.frame_timestamps_utc,
                source=tb.source or "thermal_inferred",
                confidence=float(min(getattr(tb, "confidence", 0.1), 0.1)),
            )

        except Exception:
            pass

    # -------------------------------------------------------------
    # Timestamp axis completion
    # -------------------------------------------------------------

    if tb.frame_timestamps_utc is None and tb.dt_seconds is not None and tb.t0_utc is not None:

        epoch0 = datetime.fromisoformat(tb.t0_utc)

        dt_val = float(tb.dt_seconds)

        frame_ts = [
            (epoch0 + timedelta(seconds=i * dt_val)).isoformat()
            for i in range(T)
        ]

        tb = FrameTimebase(
            t0_utc=tb.t0_utc,
            dt_seconds=tb.dt_seconds,
            frame_timestamps_utc=frame_ts,
            source=tb.source or "thermal_inferred",
            confidence=float(min(getattr(tb, "confidence", 0.1), 0.1)),
        )

    # -------------------------------------------------------------
    # Integrity report
    # -------------------------------------------------------------

    integrity = build_integrity_report(
        particle_table=particle_table,
        processed_series=processed_series,
        dt_hint_seconds=tb.dt_seconds,
    )

    tol = config.get("timestamp_tolerance_seconds", 2.0)

    particle_to_frame: Dict[str, int] = {}
    processed_to_frame: Dict[str, int] = {}

    particle_times_aligned: Dict[str, datetime] = {}
    processed_times: Dict[str, datetime] = {}

    # -------------------------------------------------------------
    # Particle mapping
    # -------------------------------------------------------------

    if particle_table is not None and not particle_table.empty:

        if "particle_event_id" not in particle_table.columns:
            raise AlignmentError("particle_table missing particle_event_id")

        if "t_utc" not in particle_table.columns:
            raise AlignmentError("particle_table missing t_utc")

        for _, r in particle_table.iterrows():

            pid = str(r["particle_event_id"])

            t = pd.to_datetime(r["t_utc"], utc=True, errors="coerce")

            if pd.isna(t):
                continue

            # APPLY DECLARED OFFSET EXACTLY ONCE HERE
            t = _apply_declared_particle_offset(t, particle_offset_s)

            t_py = t.to_pydatetime()

            particle_times_aligned[pid] = t_py

            idx = _nearest_frame_idx(tb, t_py, T=T, tolerance_seconds=tol)

            if idx is not None:
                particle_to_frame[pid] = int(idx)

    # -------------------------------------------------------------
    # Processed mapping
    # -------------------------------------------------------------

    if processed_series is not None and not processed_series.empty:

        if "processed_row_id" not in processed_series.columns:
            raise AlignmentError("processed_series missing processed_row_id")

        if "t_utc" not in processed_series.columns:
            raise AlignmentError("processed_series missing t_utc")

        for _, r in processed_series.iterrows():

            rid = str(r["processed_row_id"])

            t = pd.to_datetime(r["t_utc"], utc=True, errors="coerce")

            if pd.isna(t):
                continue

            t_py = t.to_pydatetime()

            processed_times[rid] = t_py

            idx = _nearest_frame_idx(tb, t_py, T=T, tolerance_seconds=tol)

            if idx is not None:
                processed_to_frame[rid] = int(idx)

    # -------------------------------------------------------------
    # Residual offsets (diagnostic)
    # -------------------------------------------------------------

    off_particle = _median_offset_seconds(
        tb,
        particle_to_frame,
        particle_times_aligned,
    )

    off_processed = _median_offset_seconds(
        tb,
        processed_to_frame,
        processed_times,
    )

    offsets_seconds = {
        "particle_vs_thermal": float(particle_offset_s),
        "processed_vs_thermal": None,
    }

    residual_offsets_seconds = {
        "particle_vs_thermal": float(off_particle)
        if off_particle is not None
        else None,
        "processed_vs_thermal": float(off_processed)
        if off_processed is not None
        else None,
    }

    # -------------------------------------------------------------
    # Confidence
    # -------------------------------------------------------------

    pt_cov = (
        len(particle_to_frame) / max(1, len(particle_times_aligned))
        if particle_times_aligned
        else 0.0
    )

    pr_cov = (
        len(processed_to_frame) / max(1, len(processed_times))
        if processed_times
        else 0.0
    )

    cov = max(pt_cov, pr_cov)

    confidence = float(np.clip(0.6 * tb.confidence + 0.4 * cov, 0.0, 1.0))

    # -------------------------------------------------------------
    # Serialize timestamps
    # -------------------------------------------------------------

    fts = tb.frame_timestamps_utc

    if fts is not None:

        frame_ts_serialized = [
            t.isoformat() if hasattr(t, "isoformat") else str(t)
            for t in fts
        ]

    else:
        frame_ts_serialized = None

    # -------------------------------------------------------------
    # Alignment artifact
    # -------------------------------------------------------------

    alignment = {
        "frame_timebase": {
            "source": tb.source,
            "confidence": tb.confidence,
            "t0_utc": tb.t0_utc,
            "dt_seconds": tb.dt_seconds,
            "frame_timestamps_utc": frame_ts_serialized,
            "particle_time_offset_seconds": float(particle_offset_s),
            "authoritative_particle_time_field": "t_utc",
            "particle_time_state": "raw",
            "particle_offset_application": "alignment_mapping_only",
            "mapping_rule": "nearest_frame_idx",
            "synthetic_anchor_source": anchor_source,
        },
        "particle_to_frame": dict(sorted(particle_to_frame.items())),
        "processed_to_frame": dict(sorted(processed_to_frame.items())),
        "offsets_seconds": offsets_seconds,
        "residual_offsets_seconds": residual_offsets_seconds,
        "gaps": integrity.get("gaps", []),
        "integrity_flags": integrity.get("integrity_flags", []),
        "confidence": confidence,
    }

    return alignment, integrity