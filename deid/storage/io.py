"""
deid.storage.io
---------------

Safe artifact IO helpers with:
- directory creation
- atomic write pattern
- deterministic JSON serialization
- parquet/npz utilities

Important:
- All JSON artifacts should be wrapped via wrap_artifact() so provenance
  is embedded from day 1.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from deid.core.errors import ArtifactIOError
from deid.core.versioning import PIPELINE_VERSION


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_parent_dir(path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)


def _atomic_replace(tmp_path: Path, final_path: Path) -> None:
    """
    Atomically replace final_path with tmp_path contents.
    """
    try:
        os.replace(str(tmp_path), str(final_path))
    except Exception as e:
        raise ArtifactIOError(
            "Atomic replace failed",
            details={"tmp_path": str(tmp_path), "final_path": str(final_path), "error": str(e)},
        )


def _atomic_write_bytes(final_path: Path, data: bytes) -> None:
    """
    Write bytes to a temp file in the same directory, then atomically replace.
    """
    ensure_parent_dir(final_path)

    tmp_dir = final_path.parent
    # NamedTemporaryFile(delete=False) ensures Windows compatibility too.
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".tmp_{final_path.name}.",
        dir=str(tmp_dir),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        _atomic_replace(tmp_path, final_path)
    except Exception as e:
        # Best effort cleanup
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise ArtifactIOError(
            "Atomic write failed",
            details={"path": str(final_path), "error": str(e)},
        )


# -------------------------------------------------------------------
# Provenance wrapper
# -------------------------------------------------------------------


def wrap_artifact(
    payload: Any,
    schema_version: str,
    config_hash: str,
    input_hashes: Dict[str, str],
    pipeline_version: str = PIPELINE_VERSION,
    *,
    artifact_type: Optional[str] = None,
    extra_provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Wrap an artifact payload with standard provenance fields.

    Returns a JSON-serializable dict.
    """
    header: Dict[str, Any] = {
        "schema_version": schema_version,
        "pipeline_version": pipeline_version,
        "config_hash": config_hash,
        "input_hashes": dict(input_hashes),
        "created_at_utc": utc_now_iso(),
    }
    if artifact_type:
        header["artifact_type"] = artifact_type
    if extra_provenance:
        header["provenance"] = dict(extra_provenance)

    return {
        "header": header,
        "payload": payload,
    }


# -------------------------------------------------------------------
# JSON
# -------------------------------------------------------------------


def write_json(path: str | Path, obj: Any, *, sort_keys: bool = True, indent: int = 2) -> None:
    """
    Deterministic JSON write with atomic replace.
    """
    try:
        data = json.dumps(obj, sort_keys=sort_keys, indent=indent, ensure_ascii=False).encode("utf-8")
        _atomic_write_bytes(Path(path), data)
    except ArtifactIOError:
        raise
    except Exception as e:
        raise ArtifactIOError("Failed to write JSON", details={"path": str(path), "error": str(e)})


def read_json(path: str | Path) -> Any:
    try:
        p = Path(path)
        with p.open("rb") as f:
            return json.loads(f.read().decode("utf-8"))
    except Exception as e:
        raise ArtifactIOError("Failed to read JSON", details={"path": str(path), "error": str(e)})


# -------------------------------------------------------------------
# Parquet
# -------------------------------------------------------------------


def write_parquet(path: str | Path, df: pd.DataFrame, *, index: bool = False) -> None:
    """
    Write parquet atomically.

    Note: pandas writes to a file path; we use a temp path then replace.
    """
    final_path = Path(path)
    ensure_parent_dir(final_path)

    tmp_path = final_path.parent / f".tmp_{final_path.name}.{os.getpid()}"
    try:
        df.to_parquet(tmp_path, index=index)
        _atomic_replace(tmp_path, final_path)
    except Exception as e:
        # Best effort cleanup
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise ArtifactIOError(
            "Failed to write Parquet",
            details={"path": str(final_path), "error": str(e)},
        )


def read_parquet(path: str | Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(Path(path))
    except Exception as e:
        raise ArtifactIOError("Failed to read Parquet", details={"path": str(path), "error": str(e)})


# -------------------------------------------------------------------
# NPZ
# -------------------------------------------------------------------


def write_npz(path: str | Path, **arrays: np.ndarray) -> None:
    """
    Write NPZ atomically.
    """
    final_path = Path(path)
    ensure_parent_dir(final_path)

    tmp_path = final_path.parent / f".tmp_{final_path.name}.{os.getpid()}"
    try:
        np.savez_compressed(tmp_path, **arrays)
        # numpy appends .npz if not present; ensure consistent temp naming
        if tmp_path.suffix != ".npz":
            tmp_path_npz = Path(str(tmp_path) + ".npz")
        else:
            tmp_path_npz = tmp_path
        _atomic_replace(tmp_path_npz, final_path)
    except Exception as e:
        # Best effort cleanup
        try:
            if tmp_path.exists():
                tmp_path.unlink()
            tmp_path_npz = Path(str(tmp_path) + ".npz")
            if tmp_path_npz.exists():
                tmp_path_npz.unlink()
        except Exception:
            pass
        raise ArtifactIOError("Failed to write NPZ", details={"path": str(final_path), "error": str(e)})


def read_npz(path: str | Path) -> Dict[str, np.ndarray]:
    try:
        p = Path(path)
        with np.load(p, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    except Exception as e:
        raise ArtifactIOError("Failed to read NPZ", details={"path": str(path), "error": str(e)})