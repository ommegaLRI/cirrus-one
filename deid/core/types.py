"""
deid.core.types
---------------

Lightweight shared dataclasses used across modules.

Do NOT replicate Pydantic config models here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple


# -------------------------------------------------------------------
# Thermal cube reference
# -------------------------------------------------------------------


@dataclass(frozen=True)
class ThermalCubeRef:
    """
    Reference to an on-disk thermal cube dataset.
    """

    uri: str
    dataset_path: str
    shape: Tuple[int, int, int]  # (T, H, W)
    dtype: str
    chunking: Optional[Tuple[int, int, int]] = None
    compression: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


# -------------------------------------------------------------------
# Input manifest (minimal stub; expanded later)
# -------------------------------------------------------------------


@dataclass
class InputManifest:
    session_id: str
    config_hash: str
    inputs: Dict[str, Any]


# -------------------------------------------------------------------
# Job status enum-like class
# -------------------------------------------------------------------


class JobStatus:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"