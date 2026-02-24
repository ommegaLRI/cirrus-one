"""
deid.config.models
------------------

Pydantic configuration models.

Rules:
- Strong typing
- Deterministic defaults
- No runtime logic here
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


# -------------------------------------------------------------------
# Base config behavior
# -------------------------------------------------------------------


class DEIDBaseConfig(BaseModel):
    class Config:
        extra = "forbid"
        frozen = True


# -------------------------------------------------------------------
# Ingest configuration
# -------------------------------------------------------------------


class IngestConfig(DEIDBaseConfig):
    timezone: str = Field(
        default="America/New_York",
        description="Default timezone if input timestamps are naive",
    )

    hdf5_dataset_path: Optional[str] = Field(
        default=None,
        description="Override dataset path inside HDF5",
    )


# -------------------------------------------------------------------
# Storage configuration
# -------------------------------------------------------------------


class StorageConfig(DEIDBaseConfig):
    run_root: Path = Field(
        default=Path("./runs"),
        description="Root directory for run bundles",
    )


# -------------------------------------------------------------------
# Runner configuration
# -------------------------------------------------------------------


class RunnerConfig(DEIDBaseConfig):
    allow_stage_skip: bool = True
    fail_fast: bool = True

# -------------------------------------------------------------------
# Fusion configuration
# -------------------------------------------------------------------


class FusionConfig(BaseModel):
    time_tolerance_s: float = 2.0
    spatial_tolerance_px: float = 6.0


# -------------------------------------------------------------------
# Top-level config model
# -------------------------------------------------------------------


class DEIDConfig(DEIDBaseConfig):
    ingest: IngestConfig = IngestConfig()
    storage: StorageConfig = StorageConfig()
    runner: RunnerConfig = RunnerConfig()
    fusion: FusionConfig = FusionConfig()

    # Placeholder for later stage configs
    alignment: dict = Field(default_factory=dict)
    plate_state: dict = Field(default_factory=dict)
    event_extraction: dict = Field(default_factory=dict)
    swe: dict = Field(default_factory=dict)
    inference: dict = Field(default_factory=dict)
