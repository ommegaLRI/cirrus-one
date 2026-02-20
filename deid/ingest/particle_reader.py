"""
deid.ingest.particle_reader
---------------------------

Read Particle table (CSV/XLSX) and normalize into particle_v1 schema.

Outputs:
- A normalized DataFrame for valid rows matching the normative schema.
- A quarantine DataFrame for invalid rows (kept for provenance/debug).

Notes:
- Does not perform event matching or any scientific inference.
- Unknown columns are preserved per-row in raw_json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from zoneinfo import ZoneInfo

from deid.core.errors import IngestError, SchemaError
from deid.core.ids import stable_id
from deid.core.time import parse_date_time
from deid.core.versioning import SCHEMA_PARTICLE

_REQUIRED_HEADERS = ["Date", "Time", "Centroid location (x, y)"]


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in [".csv", ".txt"]:
        return pd.read_csv(path)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise IngestError("Unsupported particle file type", details={"path": str(path), "suffix": suffix})


def _parse_centroid(value: Any) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse "Centroid location (x, y)" which often appears like "(12.3, 45.6)" or "12.3,45.6".
    Returns (x, y) as floats, or (None, None) if parse fails.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None, None

    s = str(value).strip()
    s = s.strip("()[]")
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        return None, None
    try:
        x = float(parts[0])
        y = float(parts[1])
        return x, y
    except Exception:
        return None, None


def _row_raw_json(row: pd.Series, extra_cols: list[str]) -> str:
    d: Dict[str, Any] = {}
    for c in extra_cols:
        v = row.get(c, None)
        if v is None:
            continue
        if isinstance(v, float) and pd.isna(v):
            continue
        # JSON safe conversion
        try:
            if hasattr(v, "item"):
                v = v.item()
        except Exception:
            pass
        d[c] = v
    return json.dumps(d, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_particle_table_normalized(
    path: str | Path,
    *,
    timezone_str: str,
    source_file_label: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (valid_df, invalid_df).
    """
    p = Path(path)
    if not p.exists():
        raise IngestError("Particle file not found", details={"path": str(p)})

    df = _read_table(p)

    missing = [h for h in _REQUIRED_HEADERS if h not in df.columns]
    if missing:
        raise SchemaError("Particle table missing required headers", details={"missing": missing, "path": str(p)})

    tz = ZoneInfo(timezone_str)

    # Extract required raw columns
    date_raw = df["Date"].astype(str)
    time_raw = df["Time"].astype(str)

    centroid_col = df["Centroid location (x, y)"]

    # Candidate known optional columns
    opt_map = {
        "Time to Evaporate (Sec)": "time_to_evaporate_s",
        "Mass (mg)": "mass_mg",
        "Max Area (mm^2)": "max_area_mm2",
        "Temp (C)": "temp_c",
    }

    # Build normalized rows
    norm_rows = []
    invalid_rows = []

    extra_cols = [c for c in df.columns if c not in set(_REQUIRED_HEADERS + list(opt_map.keys()))]

    for i, row in df.iterrows():
        d_raw_date = row.get("Date", None)
        d_raw_time = row.get("Time", None)

        # Parse time
        try:
            t_utc = parse_date_time(str(d_raw_date), str(d_raw_time), tz)
        except Exception as e:
            invalid_rows.append(
                {
                    "row_index": int(i),
                    "reason": "time_parse_failed",
                    "error": str(e),
                    "date_raw": None if d_raw_date is None else str(d_raw_date),
                    "time_raw": None if d_raw_time is None else str(d_raw_time),
                    "raw_json": _row_raw_json(row, extra_cols),
                }
            )
            continue

        # Parse centroid
        x_px, y_px = _parse_centroid(row.get("Centroid location (x, y)"))
        if x_px is None or y_px is None:
            invalid_rows.append(
                {
                    "row_index": int(i),
                    "reason": "centroid_parse_failed",
                    "error": "Could not parse centroid location",
                    "date_raw": str(d_raw_date),
                    "time_raw": str(d_raw_time),
                    "raw_json": _row_raw_json(row, extra_cols),
                }
            )
            continue

        # Optional numeric fields
        def _get_float(col: str) -> Optional[float]:
            if col not in df.columns:
                return None
            v = row.get(col)
            if v is None:
                return None
            try:
                if isinstance(v, float) and pd.isna(v):
                    return None
                return float(v)
            except Exception:
                return None

        normalized: Dict[str, Any] = {
            "particle_event_id": stable_id(
                "particle",
                {
                    "t_utc": t_utc.isoformat(),
                    "row_index": int(i),
                    "source": source_file_label or str(p.name),
                },
            ),
            "t_utc": t_utc,
            "t_local_raw": f"{str(d_raw_date).strip()} {str(d_raw_time).strip()}",
            "date_raw": None if d_raw_date is None else str(d_raw_date),
            "time_raw": None if d_raw_time is None else str(d_raw_time),
            "time_to_evaporate_s": _get_float("Time to Evaporate (Sec)"),
            "x_px": float(x_px),
            "y_px": float(y_px),
            "mass_mg": _get_float("Mass (mg)"),
            "max_area_mm2": _get_float("Max Area (mm^2)"),
            "temp_c": _get_float("Temp (C)"),
            "source_file": source_file_label or str(p.name),
            "row_index": int(i),
            "raw_json": _row_raw_json(row, extra_cols),
            "schema_version": SCHEMA_PARTICLE,
        }
        norm_rows.append(normalized)

    valid_df = pd.DataFrame(norm_rows)
    invalid_df = pd.DataFrame(invalid_rows)

    # Enforce uniqueness (best-effort; collisions extremely unlikely)
    if not valid_df.empty and valid_df["particle_event_id"].duplicated().any():
        raise SchemaError(
            "particle_event_id collision detected",
            details={"path": str(p), "hint": "Adjust stable_id fields if needed."},
        )

    # Ensure t_utc is tz-aware UTC
    if not valid_df.empty:
        valid_df["t_utc"] = pd.to_datetime(valid_df["t_utc"], utc=True)

    return valid_df, invalid_df