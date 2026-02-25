"""
deid.api.run_registry
---------------------
Run resolution utilities.

Resolves runs by run_id only.
"""

from __future__ import annotations

from pathlib import Path
from fastapi import HTTPException

from deid.storage.paths import run_root

# Resolve runs directory relative to project root
RUNS_ROOT = (Path(__file__).resolve().parents[2] / "runs").resolve()


def resolve_run(run_id: str) -> Path:
    """
    Resolve a run directory by run_id.

    Supports:
        runs/<run_id>/
        runs/<session_id>/<run_id>/
    """

    # --- Direct match (flat layout) ---
    candidate = RUNS_ROOT / run_id
    if candidate.exists() and candidate.is_dir():
        return candidate.resolve()

    # --- Nested layout ---
    for sub in RUNS_ROOT.iterdir():
        if not sub.is_dir():
            continue

        nested = sub / run_id
        if nested.exists() and nested.is_dir():
            return nested.resolve()

    # --- Debug info (very important) ---
    print("DEBUG resolve_run failed")
    print("RUNS_ROOT:", RUNS_ROOT)
    print("Requested run_id:", run_id)
    print("Available dirs:", [p.name for p in RUNS_ROOT.iterdir() if p.is_dir()])

    raise HTTPException(status_code=404, detail="run not found")