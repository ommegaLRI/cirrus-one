"""
deid.api.routes_jobs
--------------------
Job status endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from deid.api.schemas_api import JobDetailResponse
from deid.api.job_registry import JOBS

router = APIRouter()


@router.get("/jobs/{job_id}", response_model=JobDetailResponse)
def get_job(job_id: str):
    job = JOBS.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    return JobDetailResponse(
        job_id=job.job_id,
        run_id=job.run_id,
        session_id=job.session_id,
        state=job.state,
        created_at_utc=job.created_at_utc,
        started_at_utc=job.started_at_utc,
        finished_at_utc=job.finished_at_utc,
        stage_status=job.stage_status,
        stage_timings_s=job.stage_timings_s,
        errors=job.errors,
    )