"""
deid.events.catalog
-------------------

Build event_catalog_v1 DataFrame with required columns.

This module is schema-focused, not algorithmic.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from deid.core.errors import SchemaError
from deid.core.versioning import SCHEMA_EVENT_CATALOG


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_event_catalog_df(
    *,
    events: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    events: list of dicts with keys:
      event_id, frame_start, frame_peak, frame_end,
      centroid_start_yx, centroid_end_yx,
      motion_score, edge_proximity,
      mask_ref, split_merge (list/obj),
      delta_peak, tau1_s, tau2_s, area_peak_px, area_peak_mm2, energy_proxy_E, duration_s,
      snr, overlap_score, fragmentation_score, saturation_flag, quality_flags (list),
      baseline_at_peak, noise_at_centroid, nonuniformity_at_centroid
    """
    rows = []
    for e in events:
        rows.append(
            {
                "event_id": e["event_id"],
                "frame_start": int(e["frame_start"]),
                "frame_peak": int(e["frame_peak"]),
                "frame_end": int(e["frame_end"]),
                "t_start_utc": e.get("t_start_utc"),
                "t_peak_utc": e.get("t_peak_utc"),
                "t_end_utc": e.get("t_end_utc"),

                "centroid_start_y": float(e["centroid_start_yx"][0]),
                "centroid_start_x": float(e["centroid_start_yx"][1]),
                "centroid_end_y": float(e["centroid_end_yx"][0]),
                "centroid_end_x": float(e["centroid_end_yx"][1]),
                "motion_score": float(e.get("motion_score", 0.0)),
                "edge_proximity": float(e.get("edge_proximity", 0.0)),

                "mask_ref": str(e["mask_ref"]),
                "split_merge_json": _json(e.get("split_merge", [])),

                "delta_peak": float(e.get("delta_peak", 0.0)),
                "tau1_s": e.get("tau1_s"),
                "tau2_s": e.get("tau2_s"),
                "area_peak_px": float(e.get("area_peak_px", 0.0)),
                "area_peak_mm2": e.get("area_peak_mm2"),
                "energy_proxy_E": float(e.get("energy_proxy_E", 0.0)),
                "duration_s": e.get("duration_s"),

                "snr": float(e.get("snr", 0.0)),
                "overlap_score": float(e.get("overlap_score", 0.0)),
                "fragmentation_score": float(e.get("fragmentation_score", 0.0)),
                "saturation_flag": bool(e.get("saturation_flag", False)),
                "quality_flags_json": _json(e.get("quality_flags", [])),

                "baseline_at_peak": e.get("baseline_at_peak"),
                "noise_at_centroid": e.get("noise_at_centroid"),
                "nonuniformity_at_centroid": e.get("nonuniformity_at_centroid"),

                "schema_version": SCHEMA_EVENT_CATALOG,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        # enforce invariant
        bad = df.query("frame_start >= frame_peak or frame_peak >= frame_end")
        if len(bad) > 0:
            raise SchemaError("Invalid frame ordering in extracted events", details={"n_bad": int(len(bad))})

        if df["event_id"].duplicated().any():
            raise SchemaError("Duplicate event_id detected in extracted events")

    return df