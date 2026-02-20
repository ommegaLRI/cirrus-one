"""
deid.fusion.tables
------------------

Build matched_events_v1 DataFrame.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from deid.core.versioning import SCHEMA_MATCHED_EVENTS


def build_matched_events_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["event_id", "particle_event_id", "schema_version"])

    df = pd.DataFrame(rows)
    df["schema_version"] = SCHEMA_MATCHED_EVENTS
    return df