"""
deid.api.routes_sessions
------------------------

POST /sessions/analyze

Starts a new analysis run from frontend request.
"""

from __future__ import annotations

import threading

from fastapi import APIRouter, HTTPException

from deid.api.schemas_api import AnalyzeRequest, JobStatusResponse
from deid.api.job_registry import JOBS

from deid.api.pipeline_adapter import start_pipeline
from deid.api.file_download import download_to_temp

from deid.config.models import DEIDConfig
from deid.config.hashing import compute_config_hash

from deid.runner.execute import run_sequential


router = APIRouter()


# ---------------------------------------------------------
# Background execution
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# POST /sessions/analyze
# ---------------------------------------------------------

@router.post("/sessions/analyze", response_model=JobStatusResponse)
def analyze(req: AnalyzeRequest):

    # ---------------------------------------------------
    # Validate config
    # ---------------------------------------------------

    try:
        config = DEIDConfig.model_validate(req.config or {})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Compute config hash
    config_hash = compute_config_hash(config)

    # ---------------------------------------------------
    # Download files from Supabase signed URLs
    # ---------------------------------------------------

    try:
        hdf5_path = download_to_temp(req.hdf5)

        particle_path = download_to_temp(req.particle) if req.particle else None
        processed_path = download_to_temp(req.processed) if req.processed else None

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"file download failed: {e}")

    # ---------------------------------------------------
    # Build inputs for pipeline
    # ---------------------------------------------------

    inputs = {
        "session_id": req.session_id,
        "particle": particle_path,
        "processed": processed_path,
        "hdf5": hdf5_path,
        "config_hash": config_hash,
    }

    # ---------------------------------------------------
    # Start pipeline (same logic as CLI)
    # ---------------------------------------------------

    job, run_dir, stage_fns, input_hashes, config_hash = start_pipeline(
        session_id=req.session_id,
        config=config,
        inputs=inputs,
    )

    JOBS[job.job_id] = job

    # ---------------------------------------------------
    # Run pipeline asynchronously
    # ---------------------------------------------------

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