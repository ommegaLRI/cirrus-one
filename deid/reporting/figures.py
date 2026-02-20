"""
deid.reporting.figures
----------------------

Standardized figures for research + debugging.

Figures produced:
- swe_vs_reconstructed.png
- baseline_drift.png
- nonuniformity_map.png
- noise_map.png
- event_rate.png
- regime_fraction_timeline.png (optional)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from deid.storage.io import read_json
from deid.storage.paths import intermediate_dir, outputs_dir


def _load_payload(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path).get("payload", {})


def _ensure_fig_dir(run_dir: Path) -> Path:
    fig_dir = outputs_dir(run_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def _plot_swe(run_dir: Path, fig_dir: Path):
    swe_path = outputs_dir(run_dir) / "swe_products.parquet"
    if not swe_path.exists():
        return

    df = pd.read_parquet(swe_path)

    if "t_utc" not in df:
        return

    plt.figure()
    if "swe_processed_mm" in df:
        plt.plot(df["t_utc"], df["swe_processed_mm"], label="processed")
    if "swe_reconstructed_mm" in df:
        plt.plot(df["t_utc"], df["swe_reconstructed_mm"], label="reconstructed")

    plt.legend()
    plt.title("SWE Processed vs Reconstructed")
    plt.tight_layout()
    plt.savefig(fig_dir / "swe_vs_reconstructed.png")
    plt.close()


def _plot_plate_state(run_dir: Path, fig_dir: Path):
    npz_path = intermediate_dir(run_dir) / "plate_state.npz"
    if not npz_path.exists():
        return

    with np.load(npz_path) as z:
        B = z["baseline_B_t"]
        G = z["nonuniformity_G_yx"]
        N = z["noise_N_yx"]

    plt.figure()
    plt.plot(B)
    plt.title("Baseline Drift B(t)")
    plt.tight_layout()
    plt.savefig(fig_dir / "baseline_drift.png")
    plt.close()

    plt.figure()
    plt.imshow(G)
    plt.colorbar()
    plt.title("Nonuniformity Map")
    plt.tight_layout()
    plt.savefig(fig_dir / "nonuniformity_map.png")
    plt.close()

    plt.figure()
    plt.imshow(N)
    plt.colorbar()
    plt.title("Noise Map")
    plt.tight_layout()
    plt.savefig(fig_dir / "noise_map.png")
    plt.close()


def _plot_event_rate(run_dir: Path, fig_dir: Path):
    event_path = intermediate_dir(run_dir) / "event_catalog.parquet"
    align_payload = _load_payload(intermediate_dir(run_dir) / "alignment.json")

    if not event_path.exists():
        return

    df = pd.read_parquet(event_path)
    if len(df) == 0:
        return

    ts = align_payload.get("frame_timebase", {}).get("frame_timestamps_utc")
    if ts is None:
        return

    ts_ser = pd.to_datetime(pd.Series(ts), utc=True, errors="coerce")
    fp = df["frame_peak"].astype(int)
    t = ts_ser.iloc[fp.values]

    counts = t.value_counts().sort_index()

    plt.figure()
    plt.plot(counts.index, counts.values)
    plt.title("Event Rate")
    plt.tight_layout()
    plt.savefig(fig_dir / "event_rate.png")
    plt.close()


def _plot_regimes(run_dir: Path, fig_dir: Path):
    regimes = _load_payload(outputs_dir(run_dir) / "regimes.json")
    timeline = regimes.get("timeline", {})
    bins = timeline.get("bins")
    fractions = timeline.get("fractions")

    if not bins or not fractions:
        return

    x = pd.to_datetime(pd.Series(bins), utc=True, errors="coerce")

    plt.figure()
    for r, vals in fractions.items():
        plt.plot(x[: len(vals)], vals, label=f"regime {r}")

    plt.legend()
    plt.title("Regime Fraction Timeline")
    plt.tight_layout()
    plt.savefig(fig_dir / "regime_fraction_timeline.png")
    plt.close()


def generate_figures(run_dir: Path):
    fig_dir = _ensure_fig_dir(run_dir)

    _plot_swe(run_dir, fig_dir)
    _plot_plate_state(run_dir, fig_dir)
    _plot_event_rate(run_dir, fig_dir)
    _plot_regimes(run_dir, fig_dir)