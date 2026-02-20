"""
deid.ingest.processed_reader
----------------------------

Read processed SWE table (CSV/XLSX) and normalize into processed_v1 schema.

Outputs:
- normalized processed_v1 DataFrame for valid rows
- quarantine DataFrame for invalid rows

Does not enforce monotonicity (that is Stage B integrity).
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
from deid.core.versioning import SCHEMA_PROCESSED

_REQUIRED_HEADERS = ["Date", "Time"]


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in [".csv", ".txt"]:
        return pd.read_csv(path)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise IngestError("Unsupported processed file type", details={"path": str(path), "suffix": suffix})


def _row_raw_json(row: pd.Series, extra_cols: list[str]) -> str:
    d: Dict[str, Any] = {}
    for c in extra_cols:
        v = row.get(c, None)
        if v is None:
            continue
        if isinstance(v, float) and pd.isna(v):
            continue
        try:
            if hasattr(v, "item"):
                v = v.item()
        except Exception:
            pass
        d[c] = v
    return json.dumps(d, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_processed_series_normalized(
    path: str | Path,
    *,
    timezone_str: str,
    source_file_label: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        raise IngestError("Processed file not found", details={"path": str(p)})

    df = _read_table(p)
    missing = [h for h in _REQUIRED_HEADERS if h not in df.columns]
    if missing:
        raise SchemaError("Processed table missing required headers", details={"missing": missing, "path": str(p)})

    tz = ZoneInfo(timezone_str)

    # Optional known columns
    col_swe = "SWE (mm)" if "SWE (mm)" in df.columns else None
    col_rate = "SWE Rate (mm/hr)" if "SWE Rate (mm/hr)" in df.columns else None

    extra_cols = [c for c in df.columns if c not in set(_REQUIRED_HEADERS + [c for c in [col_swe, col_rate] if c])]

    norm_rows = []
    invalid_rows = []

    for i, row in df.iterrows():
        d_raw_date = row.get("Date", None)
        d_raw_time = row.get("Time", None)

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

        def _get_float(col: Optional[str]) -> Optional[float]:
            if not col:
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

        swe_mm = _get_float(col_swe)
        swe_rate = _get_float(col_rate)

        normalized: Dict[str, Any] = {
            "processed_row_id": stable_id(
                "processed",
                {
                    "t_utc": t_utc.isoformat(),
                    "row_index": int(i),
                    "source": source_file_label or str(p.name),
                },
            ),
            "t_utc": t_utc,
            "t_local_raw": f"{str(d_raw_date).strip()} {str(d_raw_time).strip()}",
            "swe_mm": swe_mm,
            "swe_rate_mmhr": swe_rate,
            "source_file": source_file_label or str(p.name),
            "row_index": int(i),
            "raw_json": _row_raw_json(row, extra_cols),
            "schema_version": SCHEMA_PROCESSED,
        }
        norm_rows.append(normalized)

    valid_df = pd.DataFrame(norm_rows)
    invalid_df = pd.DataFrame(invalid_rows)

    if not valid_df.empty:
        valid_df["t_utc"] = pd.to_datetime(valid_df["t_utc"], utc=True)

    # processed_row_id uniqueness (best-effort)
    if not valid_df.empty and valid_df["processed_row_id"].duplicated().any():
        raise SchemaError("processed_row_id collision detected", details={"path": str(p)})

    return valid_df, invalid_df