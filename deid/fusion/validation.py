"""
deid.fusion.validation
----------------------

Diagnostic metrics comparing authoritative events and particle table.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def compute_discrepancies(
    matches: List[Dict[str, Any]],
    event_df: pd.DataFrame,
    particle_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Attach particle/event deltas.

    v1 compares:
        area_peak_px vs max_area_mm2 (placeholder)
        duration vs evap time (placeholder)
    """

    if not matches:
        return []

    ev_index = event_df.set_index("event_id")
    pt_index = particle_df.set_index("particle_event_id")

    rows: List[Dict[str, Any]] = []

    for m in matches:
        eid = m["event_id"]
        pid = m["particle_event_id"]

        ev = ev_index.loc[eid]
        pt = pt_index.loc[pid]

        darea = None
        if "max_area_mm2" in pt and ev.get("area_peak_mm2") is not None:
            darea = float(pt["max_area_mm2"] - ev["area_peak_mm2"])

        devap = None
        if "time_to_evaporate_s" in pt and ev.get("duration_s") is not None:
            devap = float(pt["time_to_evaporate_s"] - ev["duration_s"])

        row = dict(m)
        row.update(
            {
                "mass_mg_particle": pt.get("mass_mg"),
                "mass_mg_authoritative": None,
                "dmass_mg": None,
                "area_mm2_particle": pt.get("max_area_mm2"),
                "area_mm2_authoritative": ev.get("area_peak_mm2"),
                "darea_mm2": darea,
                "evap_s_particle": pt.get("time_to_evaporate_s"),
                "evap_s_authoritative": ev.get("duration_s"),
                "devap_s": devap,
                "flags_json": "[]",
            }
        )
        rows.append(row)

    return rows


def build_validation_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"schema_version": "particle_validation_v1", "n_matches": 0}

    dts = [r["dt_s"] for r in rows if r["dt_s"] is not None]
    dxys = [r["dxy_px"] for r in rows if r["dxy_px"] is not None]

    return {
        "schema_version": "particle_validation_v1",
        "n_matches": int(len(rows)),
        "dt_mean": float(np.mean(dts)) if dts else None,
        "dxy_mean": float(np.mean(dxys)) if dxys else None,
    }