"""
deid.core.time
--------------

Centralized time parsing and normalization utilities.

Rules:
- ALL timestamp normalization must go through this module.
- Returned datetimes are timezone-aware UTC.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from deid.core.errors import SchemaError

# Supported datetime formats seen in DEID exports
_DATE_TIME_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%m/%d/%Y %H:%M:%S.%f",
]


def to_utc(dt: datetime, tz: Optional[timezone] = None) -> datetime:
    """
    Normalize datetime to timezone-aware UTC.

    If dt is naive, tz must be provided.
    """

    if dt.tzinfo is None:
        if tz is None:
            raise SchemaError(
                "Naive datetime provided without timezone",
                details={"datetime": str(dt)},
            )
        dt = dt.replace(tzinfo=tz)

    return dt.astimezone(timezone.utc)


def _try_parse(s: str) -> datetime:
    last_error = None
    for fmt in _DATE_TIME_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError as e:
            last_error = e
    raise SchemaError(
        "Failed to parse datetime string",
        details={"value": s, "error": str(last_error)},
    )


def parse_date_time(
    date_str: str,
    time_str: str,
    tz: timezone,
) -> datetime:
    """
    Parse Date + Time fields and return UTC datetime.

    Parameters
    ----------
    date_str : str
    time_str : str
    tz : timezone
        Assumed local timezone of input.
    """

    if not date_str or not time_str:
        raise SchemaError(
            "Date or Time field missing",
            details={"date": date_str, "time": time_str},
        )

    combined = f"{date_str.strip()} {time_str.strip()}"
    dt_local = _try_parse(combined)
    return to_utc(dt_local, tz)