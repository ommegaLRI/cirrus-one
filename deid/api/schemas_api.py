"""
deid.api.schemas_api
--------------------
Pydantic models for API layer.
"""

from __future__ import annotations

from pydantic import BaseModel
from typing import Optional, Dict, Any


class AnalyzeRequest(BaseModel):
    session_id: str
    particle: Optional[str] = None
    processed: Optional[str] = None
    hdf5: str
    config: Optional[Dict[str, Any]] = None


class JobStatusResponse(BaseModel):
    job_id: str
    run_id: str
    state: str
    error: Optional[str] = None

class JobDetailResponse(BaseModel):
    job_id: str
    run_id: str
    state: str
    created_at_utc: str
    started_at_utc: Optional[str]
    finished_at_utc: Optional[str]
    stage_status: Dict[str, str]
    stage_timings_s: Dict[str, float]
    errors: Dict[str, Any]


class RunSummaryResponse(BaseModel):
    run_id: str
    qc_summary: Optional[Dict[str, Any]]
    closure_report: Optional[Dict[str, Any]]
    alignment_confidence: Optional[float]