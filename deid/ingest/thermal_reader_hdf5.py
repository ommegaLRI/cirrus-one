"""
deid.ingest.thermal_reader_hdf5
-------------------------------

HDF5 thermal cube reader (metadata-focused).

Rules:
- Do NOT load the whole cube.
- Return a ThermalCubeRef describing dataset path, shape, dtype, chunking, compression.
- If dataset path not provided, attempt to discover a suitable dataset.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py

from deid.core.errors import IngestError, SchemaError
from deid.core.types import ThermalCubeRef


def _is_candidate_dataset(ds: h5py.Dataset) -> bool:
    try:
        shape = ds.shape
        dtype = str(ds.dtype)
    except Exception:
        return False

    if shape is None or len(shape) != 3:
        return False
    # Expect uint16
    if "uint16" not in dtype:
        return False
    return True


def _discover_dataset_path(h5: h5py.File) -> str:
    candidates: list[str] = []

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset) and _is_candidate_dataset(obj):
            candidates.append("/" + name.strip("/"))

    h5.visititems(visitor)

    if not candidates:
        raise SchemaError(
            "No suitable uint16 3D dataset found in HDF5",
            details={"hint": "Provide ingest.hdf5_dataset_path in config if dataset is nonstandard."},
        )

    # Deterministic choice: lexicographically smallest path
    candidates.sort()
    return candidates[0]


def _compression_metadata(ds: h5py.Dataset) -> Optional[Dict[str, Any]]:
    # h5py provides ds.compression and ds.compression_opts for many common filters
    comp = ds.compression
    if comp is None:
        return None
    meta: Dict[str, Any] = {"type": comp}
    try:
        opts = ds.compression_opts
        if opts is not None:
            meta["opts"] = opts
    except Exception:
        pass
    return meta


def read_thermal_cube_ref(
    hdf5_path: str | Path,
    *,
    dataset_path: Optional[str] = None,
    sample_read: bool = False,
) -> ThermalCubeRef:
    """
    Read thermal cube metadata from an HDF5 file and return ThermalCubeRef.

    Parameters
    ----------
    hdf5_path : str | Path
        Path/URI to local HDF5 file.
    dataset_path : Optional[str]
        Dataset path inside HDF5 (e.g. "/data"). If None, auto-discover.
    sample_read : bool
        If True, reads a single frame slice to ensure dataset is readable.

    Returns
    -------
    ThermalCubeRef
    """
    p = Path(hdf5_path)
    if not p.exists():
        raise IngestError("Thermal HDF5 file not found", details={"path": str(p)})

    try:
        with h5py.File(p, "r") as h5:
            ds_path = dataset_path or _discover_dataset_path(h5)
            if ds_path not in h5:
                raise SchemaError(
                    "Specified dataset_path not found in HDF5",
                    details={"path": str(p), "dataset_path": ds_path},
                )

            ds = h5[ds_path]
            if not _is_candidate_dataset(ds):
                raise SchemaError(
                    "HDF5 dataset does not match expected constraints (uint16, 3D)",
                    details={"path": str(p), "dataset_path": ds_path, "shape": str(ds.shape), "dtype": str(ds.dtype)},
                )

            shape: Tuple[int, int, int] = tuple(int(x) for x in ds.shape)  # (T,H,W)
            dtype = str(ds.dtype)
            chunking = tuple(int(x) for x in ds.chunks) if ds.chunks is not None else None
            compression = _compression_metadata(ds)

            # Preserve small metadata deterministically
            md: Dict[str, Any] = {}
            try:
                # Common attrs (device info etc.) if present
                for k in sorted(ds.attrs.keys()):
                    v = ds.attrs.get(k)
                    # Make JSON-friendly if possible
                    try:
                        if hasattr(v, "tolist"):
                            v = v.tolist()
                    except Exception:
                        pass
                    md[k] = v
            except Exception:
                pass

            if sample_read:
                # Read one frame to validate access; do not store it.
                _ = ds[0, :, :]

            return ThermalCubeRef(
                uri=str(p),
                dataset_path=ds_path,
                shape=shape,
                dtype="uint16" if "uint16" in dtype else dtype,
                chunking=chunking,
                compression=compression,
                metadata=md or None,
            )
    except (SchemaError, IngestError):
        raise
    except Exception as e:
        raise IngestError(
            "Failed to read thermal HDF5 metadata",
            details={"path": str(p), "error": str(e)},
        )


def thermal_cube_ref_to_dict(ref: ThermalCubeRef) -> Dict[str, Any]:
    """
    JSON-safe dict for ThermalCubeRef.
    """
    d = asdict(ref)
    return d