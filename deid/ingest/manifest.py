"""
deid.ingest.manifest
--------------------

Input manifest generation + hashing.

Produces a provenance object suitable to be written to:
  run_*/inputs/manifest.json

Note:
- This module should not decide run_id. That is core/runner responsibility later.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from importlib.metadata import PackageNotFoundError, version as pkg_version

from deid.core.errors import IngestError
from deid.core.types import ThermalCubeRef
from deid.storage.hashing import sha256_file


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_pkg_version(name: str) -> Optional[str]:
    try:
        return pkg_version(name)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def build_software_provenance(extra_packages: Optional[List[str]] = None) -> Dict[str, Any]:
    pkgs = ["numpy", "pandas", "h5py", "pyarrow", "pydantic"]
    if extra_packages:
        pkgs.extend(extra_packages)

    package_versions: Dict[str, str] = {}
    for p in sorted(set(pkgs)):
        v = _safe_pkg_version(p)
        if v:
            package_versions[p] = v

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": package_versions,
        "git_commit": None,   # filled later by runner if desired
        "build_id": None,
    }


def _input_file_entry(path: str | Path, file_type: str, ingest_notes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise IngestError("Input file not found for manifest", details={"path": str(p), "type": file_type})

    return {
        "path": str(p),
        "type": file_type,
        "sha256": sha256_file(p),
        "size_bytes": int(p.stat().st_size),
        "ingest_notes": ingest_notes or {},
    }


def build_input_manifest(
    *,
    session_id: str,
    config_hash: str,
    thermal_hdf5_path: str | Path,
    thermal_ref: ThermalCubeRef,
    particle_path: Optional[str | Path] = None,
    processed_path: Optional[str | Path] = None,
    ingest_notes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a manifest dict (JSON-serializable).
    """
    inputs: List[Dict[str, Any]] = []

    inputs.append(
        _input_file_entry(
            thermal_hdf5_path,
            "thermal_hdf5",
            ingest_notes={"dataset_path": thermal_ref.dataset_path, **(ingest_notes or {})},
        )
    )

    if particle_path:
        inputs.append(_input_file_entry(particle_path, "particle", ingest_notes=ingest_notes))

    if processed_path:
        inputs.append(_input_file_entry(processed_path, "processed", ingest_notes=ingest_notes))

    manifest = {
        "session_id": session_id,
        "created_at_utc": _utc_now(),
        "inputs": inputs,
        "thermal_ref": asdict(thermal_ref),
        "software": build_software_provenance(),
        "config_hash": config_hash,
    }
    return manifest


def extract_input_hashes(manifest: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract {type: sha256} mapping for convenience.
    If multiple files of same type exist, last wins (not expected in v1).
    """
    out: Dict[str, str] = {}
    for entry in manifest.get("inputs", []):
        t = entry.get("type")
        h = entry.get("sha256")
        if t and h:
            out[str(t)] = str(h)
    return out