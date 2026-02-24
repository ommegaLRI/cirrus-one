"""
deid.config.hashing
-------------------

Canonical config hashing utilities.

Produces deterministic config_hash used for:
- run_id generation
- provenance tracking
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict
from pathlib import Path

from deid.config.models import DEIDConfig


# -------------------------------------------------------------------
# Canonicalization
# -------------------------------------------------------------------


def _normalize(obj: Any) -> Any:
    """
    Convert config into canonical JSON-safe structure.
    """

    if isinstance(obj, dict):
        return {k: _normalize(obj[k]) for k in sorted(obj.keys())}

    elif isinstance(obj, list):
        return [_normalize(v) for v in obj]

    elif isinstance(obj, Path):
        # CRITICAL FIX: make paths deterministic + JSON-safe
        return obj.as_posix()

    elif hasattr(obj, "model_dump"):
        # Pydantic v2 models
        return _normalize(obj.model_dump())

    return obj


def canonical_config_dict(config: DEIDConfig) -> Dict[str, Any]:
    return _normalize(config)


def canonical_config_json(config: DEIDConfig) -> str:
    return json.dumps(
        canonical_config_dict(config),
        sort_keys=True,
        separators=(",", ":"),
    )


# -------------------------------------------------------------------
# Hashing
# -------------------------------------------------------------------


def compute_config_hash(config: DEIDConfig, length: int = 16) -> str:
    """
    Compute deterministic config hash.
    """

    payload = canonical_config_json(config).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:length]
    return digest