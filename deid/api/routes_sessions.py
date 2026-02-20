"""
deid.api.routes_sessions
------------------------

POST /sessions/analyze
"""

from __future__ import annotations

from fastapi import APIRouter
from pathlib import Path
import uuid

from deid.api.schemas_api import AnalyzeRequest, JobStatusResponse
from deid.runner import run_pipeline

router = APIRouter()

# In-memory job registry (v1)
JOBS = {}


@router.post("/sessions/analyze", response_model=JobStatusResponse)
def analyze(req: AnalyzeRequest):
    job_id = str(uuid.uuid4())

    try:
        run_dir = run_pipeline(
            particle=req.particle,
            processed=req.processed,
            hdf5=req.hdf5,
            config=req.config,
        )
        JOBS[job_id] = {"status": "SUCCEEDED", "run_dir": str(run_dir)}
        return JobStatusResponse(job_id=job_id, status="SUCCEEDED", run_dir=str(run_dir))
    except Exception as e:
        JOBS[job_id] = {"status": "FAILED", "error": str(e)}
        return JobStatusResponse(job_id=job_id, status="FAILED", error=str(e))