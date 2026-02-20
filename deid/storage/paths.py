"""
deid.storage.paths
------------------

Canonical run-bundle path builders.

Rules:
- Return pathlib.Path objects
- Do not perform IO here (no mkdir, no writes)
"""

from __future__ import annotations

from pathlib import Path


def run_root(run_root_dir: Path, session_id: str, run_id: str) -> Path:
    return Path(run_root_dir) / f"run_{session_id}_{run_id}"


def inputs_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "inputs"


def intermediate_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "intermediate"


def outputs_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "outputs"


def provenance_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "provenance"


def figures_dir(run_dir: Path) -> Path:
    return outputs_dir(run_dir) / "figures"


def exports_dir(run_dir: Path) -> Path:
    return outputs_dir(run_dir) / "export"