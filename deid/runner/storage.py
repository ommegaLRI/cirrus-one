"""
deid.runner.storage
-------------------

Helpers for:
- checking whether a stage is complete (idempotency)
- reading/writing stage done markers
- recording output hashes for integrity

Markers are stored under:
  provenance/stages/<stage_id>.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from deid.core.errors import ArtifactIOError
from deid.storage.hashing import sha256_file
from deid.storage.io import read_json, write_json, wrap_artifact
from deid.storage.paths import provenance_dir


def stage_marker_path(run_dir: Path, stage_id: str) -> Path:
    return provenance_dir(run_dir) / "stages" / f"{stage_id}.json"


def artifact_exists(run_dir: Path, rel_path: str) -> bool:
    return (Path(run_dir) / rel_path).exists()


def compute_outputs_hashes(run_dir: Path, outputs: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for rel in outputs:
        p = Path(run_dir) / rel
        if not p.exists():
            continue
        out[rel] = sha256_file(p)
    return out


def write_stage_done_marker(
    run_dir: Path,
    stage_id: str,
    *,
    schema_version: str,
    config_hash: str,
    input_hashes: Dict[str, str],
    outputs: List[str],
) -> None:
    """
    Write a provenance-wrapped stage completion marker with output hashes.
    """
    marker = {
        "stage_id": stage_id,
        "outputs": list(outputs),
        "output_hashes": compute_outputs_hashes(run_dir, outputs),
    }
    wrapped = wrap_artifact(
        payload=marker,
        schema_version=schema_version,
        config_hash=config_hash,
        input_hashes=input_hashes,
        artifact_type="stage_done_marker",
    )

    path = stage_marker_path(run_dir, stage_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, wrapped)


def load_stage_done_marker(run_dir: Path, stage_id: str) -> Optional[Dict[str, Any]]:
    p = stage_marker_path(run_dir, stage_id)
    if not p.exists():
        return None
    return read_json(p)


def stage_is_done(
    run_dir: Path,
    stage_id: str,
    *,
    expected_outputs: List[str],
    config_hash: str,
    input_hashes: Dict[str, str],
) -> bool:
    """
    Stage is considered DONE if:
    - all expected outputs exist
    - marker exists
    - marker header matches config_hash and input_hashes exactly
    - output hashes match current file hashes
    """
    # All outputs must exist
    for rel in expected_outputs:
        if not artifact_exists(run_dir, rel):
            return False

    marker = load_stage_done_marker(run_dir, stage_id)
    if not marker:
        return False

    header = marker.get("header", {})
    if header.get("config_hash") != config_hash:
        return False

    # Compare input hashes dict exactly (deterministic keys)
    marker_inputs = header.get("input_hashes", {})
    if dict(marker_inputs) != dict(input_hashes):
        return False

    payload = marker.get("payload", {})
    marker_hashes = payload.get("output_hashes", {})
    current_hashes = compute_outputs_hashes(run_dir, expected_outputs)
    return dict(marker_hashes) == dict(current_hashes)