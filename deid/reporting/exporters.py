"""
deid.reporting.exporters
------------------------

Research-friendly exports.

Functions:
- export_parquet_bundle
- export_zarr (optional lightweight implementation)

Exports are read-only transforms of the run bundle.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import json

from deid.storage.paths import intermediate_dir, outputs_dir


def export_parquet_bundle(run_dir: Path, out_dir: Path):
    """
    Copy all parquet tables + provenance summaries into a new folder.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    inter = intermediate_dir(run_dir)
    out = outputs_dir(run_dir)

    for p in inter.glob("*.parquet"):
        shutil.copy2(p, out_dir / p.name)

    for p in out.glob("*.parquet"):
        shutil.copy2(p, out_dir / p.name)

    # include summary + qc
    for name in ["summary.json", "qc_summary.json", "closure_report.json"]:
        src = out / name
        if src.exists():
            shutil.copy2(src, out_dir / name)


def export_zarr(run_dir: Path, out_dir: Path):
    """
    Minimal Zarr export:
    - Copies plate_state arrays into zarr group
    """
    try:
        import zarr
        import numpy as np
    except Exception:
        raise RuntimeError("zarr export requires zarr + numpy installed")

    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = intermediate_dir(run_dir) / "plate_state.npz"
    if not npz_path.exists():
        return

    with np.load(npz_path) as z:
        root = zarr.open(out_dir / "plate_state.zarr", mode="w")
        for k in z.files:
            root.create_dataset(k, data=z[k], overwrite=True)