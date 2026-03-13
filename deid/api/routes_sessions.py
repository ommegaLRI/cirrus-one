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


router = APIRouter()


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
    # Background worker
    # ---------------------------------------------------

    def _background():

        try:
            # ---------------------------------------------
            # Download files from Supabase
            # ---------------------------------------------

            hdf5_path = download_to_temp(req.hdf5)

            particle_path = (
                download_to_temp(req.particle) if req.particle else None
            )

            processed_path = (
                download_to_temp(req.processed) if req.processed else None
            )

            inputs = {
                "session_id": req.session_id,
                "particle": particle_path,
                "processed": processed_path,
                "hdf5": hdf5_path,
                "config_hash": config_hash,
            }

            # ---------------------------------------------
            # Start pipeline
            # IMPORTANT:
            # start_pipeline() already invokes run_pipeline(),
            # and your logs show that run_pipeline() already
            # executes the stages. So DO NOT call any second
            # runner after this.
            # ---------------------------------------------

            job, run_dir, stage_fns, input_hashes, config_hash2 = start_pipeline(
                session_id=req.session_id,
                config=config,
                inputs=inputs,
            )

            JOBS[job.job_id] = job

        except Exception as e:
            print("background pipeline failed:", e)

    # ---------------------------------------------------
    # Launch background thread
    # ---------------------------------------------------

    threading.Thread(target=_background, daemon=True).start()

    # ---------------------------------------------------
    # Immediate response
    # job_id/run_id are not available yet because the
    # pipeline is started inside the background thread.
    # ---------------------------------------------------

    return JobStatusResponse(
        job_id=None,
        run_id=None,
        state="QUEUED",
    )