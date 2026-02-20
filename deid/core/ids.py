"""
deid.core.ids
-------------

Stable deterministic ID generation.

Do NOT use random UUIDs for core entities.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Any


def _canonical_json(obj: Dict[str, Any]) -> bytes:
    """
    Serialize dict into canonical JSON bytes.

    Ensures stable hashing across runs.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def stable_id(prefix: str, fields_dict: Dict[str, Any], length: int = 16) -> str:
    """
    Generate stable deterministic ID.

    Example:
        particle_ab12cd34ef56...
    """

    payload = _canonical_json(fields_dict)
    digest = hashlib.sha256(payload).hexdigest()[:length]
    return f"{prefix}_{digest}"