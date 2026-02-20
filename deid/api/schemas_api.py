"""
deid.api.schemas_api
--------------------

Pydantic models for API layer.
"""

from __future__ import annotations

from pydantic import BaseModel
from typing import Optional, Dict


class AnalyzeRequest(BaseModel):
    particle: Optional[str] = None
    processed: Optional[str] = None
    hdf5: str
    config: Optional[str] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    run_dir: Optional[str] = None
    error: Optional[str] = None