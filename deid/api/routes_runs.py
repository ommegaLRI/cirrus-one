"""
deid.api.routes_runs
--------------------

Run browsing + artifact access API.

Responsibilities:
- run listing
- run summary
- artifact gateway
- thermal frame streaming
- mask metadata access

NO scientific computation here.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pathlib import Path

from deid.api.run_registry import resolve_run, RUNS_ROOT
from deid.api.artifact_gateway import serve_artifact
from deid.storage.io import read_json


router = APIRouter()


# ---------------------------------------------------------
# List runs
# ---------------------------------------------------------

@router.get("/runs")
def list_runs():
    out = []

    if not RUNS_ROOT.exists():
        return out

    # Supports BOTH layouts:
    # runs/<run_id>/
    # runs/<session_id>/<run_id>/

    for item in RUNS_ROOT.iterdir():
        if not item.is_dir():
            continue

        # Case 1 — runs directly under RUNS_ROOT
        if (item / "inputs").exists():
            out.append({"run_id": item.name})
            continue

        # Case 2 — session folders containing runs
        for run in item.iterdir():
            if not run.is_dir():
                continue

            if (run / "inputs").exists():
                out.append({"run_id": run.name})

    return out


# ---------------------------------------------------------
# Run summary endpoint
# ---------------------------------------------------------

@router.get("/runs/{run_id}/summary")
def get_run_summary(run_id: str):
    """
    Lightweight summary endpoint used by dashboard overview pages.
    """
    run_dir = resolve_run(run_id)

    def safe_read(path: Path):
        if not path.exists():
            return None
        try:
            return read_json(path)["payload"]
        except Exception:
            return None

    qc_summary = safe_read(run_dir / "outputs" / "qc_summary.json")
    closure_report = safe_read(run_dir / "outputs" / "closure_report.json")
    alignment = safe_read(run_dir / "intermediate" / "alignment.json")

    alignment_confidence = None
    if alignment and isinstance(alignment, dict):
        alignment_confidence = alignment.get("confidence")

    return {
        "run_id": run_id,
        "qc_summary": qc_summary,
        "closure_report": closure_report,
        "alignment_confidence": alignment_confidence,
    }


# ---------------------------------------------------------
# Thermal frame endpoint
# ---------------------------------------------------------

@router.get("/runs/{run_id}/frames/{frame_idx}")
def get_frame(run_id: str, frame_idx: int):
    """
    Return raw uint16 thermal frame.

    Headers provide metadata required by frontend decoder worker.
    """
    # Lazy import to avoid HDF5 overhead on startup
    from deid.ingest.thermal_reader_hdf5 import open_thermal_cube

    run_dir = resolve_run(run_id)

    ref = read_json(run_dir / "inputs" / "thermal_ref.json")
    payload = ref["payload"]

    uri = payload["uri"]
    dataset_path = payload["dataset_path"]

    cube = open_thermal_cube(uri, dataset_path)
    frame = cube.read_frames(frame_idx, frame_idx + 1)[0]

    return Response(
        content=frame.astype("uint16").tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Width": str(frame.shape[1]),
            "X-Height": str(frame.shape[0]),
            "X-Dtype": "uint16",
        },
    )


# ---------------------------------------------------------
# Event mask endpoint (metadata only)
# ---------------------------------------------------------

@router.get("/runs/{run_id}/events/{event_id}/masks")
def get_event_masks(run_id: str, event_id: str):
    """
    Return mask metadata entry for an event.
    """
    run_dir = resolve_run(run_id)

    index_path = run_dir / "intermediate" / "event_masks" / "index.json"
    index = read_json(index_path)

    entry = index["payload"].get(event_id)

    if not entry:
        raise HTTPException(status_code=404, detail="mask not found")

    return entry


# ---------------------------------------------------------
# Artifact gateway (MUST BE LAST)
# ---------------------------------------------------------

@router.get("/runs/{run_id}/{artifact_path:path}")
def get_artifact(run_id: str, artifact_path: str):
    """
    Generic artifact access.

    JSON artifacts return payload only.
    Non-JSON artifacts are streamed.
    """
    run_dir = resolve_run(run_id)
    return serve_artifact(run_dir, artifact_path)