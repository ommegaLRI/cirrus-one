"""
deid.core.logging
-----------------

Lightweight structured logging wrapper.

Outputs JSON-like dicts to stdout for now.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_event(
    level: str,
    message: str,
    *,
    context: Dict[str, Any] | None = None,
) -> None:
    payload = {
        "ts": _utc_now_iso(),
        "level": level.upper(),
        "message": message,
    }

    if context:
        payload["context"] = context

    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def log_info(message: str, **ctx: Any) -> None:
    log_event("INFO", message, context=ctx or None)


def log_warning(message: str, **ctx: Any) -> None:
    log_event("WARNING", message, context=ctx or None)


def log_error(message: str, **ctx: Any) -> None:
    log_event("ERROR", message, context=ctx or None)