"""
deid.api.routes_runs
--------------------

GET /runs/{run_id}/summary
GET /runs/{run_id}/artifacts
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pathlib import Path

from deid.storage.io import read_json

router = APIRouter()


@router.get("/runs/{run_path:path}/summary")
def get_summary(run_path: str):
    run = Path(run_path)
    summary_file = run / "outputs" / "summary.json"

    if not summary_file.exists():
        raise HTTPException(status_code=404, detail="summary not found")

    return read_json(summary_file)


@router.get("/runs/{run_path:path}/artifacts")
def list_artifacts(run_path: str):
    run = Path(run_path)
    outputs = run / "outputs"

    if not outputs.exists():
        raise HTTPException(status_code=404, detail="run not found")

    return {"artifacts": [p.name for p in outputs.glob("*")]}