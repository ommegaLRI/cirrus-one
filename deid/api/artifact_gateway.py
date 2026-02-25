"""
deid.api.artifact_gateway.py
------------

"""

from __future__ import annotations

from pathlib import Path
from fastapi.responses import FileResponse
from fastapi import HTTPException

from deid.storage.io import read_json


def serve_artifact(run_dir: Path, artifact_path: str):
    target = run_dir / artifact_path

    if not target.exists():
        raise HTTPException(status_code=404, detail="artifact not found")

    # JSON artifacts returned as decoded JSON
    if target.suffix == ".json":
        return read_json(target)

    # everything else streamed
    return FileResponse(target)