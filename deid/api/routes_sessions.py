"""
deid.api.routes_sessions
------------------------

POST /sessions/analyze
"""

from __future__ import annotations

import threading
from fastapi import APIRouter, HTTPException

from deid.api.schemas_api import AnalyzeRequest, JobStatusResponse, JobDetailResponse
from deid.api.job_registry import JOBS

from deid.config.models import DEIDConfig
from deid.runner.execute import run_sequential
from deid.runner.job import Job


router = APIRouter()


def _run_background(job, run_dir, inputs, config, input_hashes, stage_fns, config_hash):
    run_sequential(
        job=job,
        run_dir=run_dir,
        inputs=inputs,
        config=config,
        config_hash=config_hash,
        input_hashes=input_hashes,
        stage_fns=stage_fns,
    )
    JOBS[job.job_id] = job


@router.post("/sessions/analyze", response_model=JobStatusResponse)
def analyze(req: AnalyzeRequest):

    try:
        config = DEIDConfig.model_validate(req.config or {})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # YOU ALREADY HAVE LOGIC THAT CREATES job + run_dir
    # keep your existing helper here
    from deid.runner import run_pipeline

    job, run_dir, inputs, input_hashes, stage_fns, config_hash = run_pipeline(
        session_id=req.session_id,
        config=config,
    )

    JOBS[job.job_id] = job

    threading.Thread(
        target=_run_background,
        args=(job, run_dir, inputs, config, input_hashes, stage_fns, config_hash),
        daemon=True,
    ).start()

    return JobStatusResponse(
        job_id=job.job_id,
        run_id=job.run_id,
        state=job.state,
    )
