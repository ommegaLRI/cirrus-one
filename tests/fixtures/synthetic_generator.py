"""
tests/fixtures/synthetic_generator.py
-------------------------------------

Synthetic DEID fixture generator.

Purpose
- Generate synthetic ThermalCube HDF5 (uint16 T×H×W) with known truth:
    * baseline drift (temporal)
    * spatial gradient / nonuniformity
    * pixel noise map
    * dead pixels
    * injected events with known masks and impulse response shapes
    * overlapping events
    * rolling motion (centroid drift)
- Optionally generate:
    * particle table (derived event-level features, noisy)
    * processed SWE table (macro cumulative SWE and SWE rate, noisy)

Design principles
- Deterministic given seed.
- Outputs are scientifically interpretable and auditably parameterized.
- Truth is emitted separately to support regression tests:
    * truth_events.parquet (event masks/features ground truth)
    * truth_plate_state.npz (true baseline/noise/nonuniformity maps)
    * truth_swe.parquet (true reconstructed SWE series)

This file intentionally lives in tests/fixtures/ and has no dependency on the service runner.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import math
import numpy as np
import pandas as pd

try:
    import h5py
except Exception as e:  # pragma: no cover
    raise RuntimeError("synthetic_generator requires h5py installed") from e


# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class SyntheticEventSpec:
    """
    Defines a single injected event.

    Coordinate convention: (y, x) in pixel coordinates.
    Event signal is NEGATIVE-going (cooling) by default; consistent with threshold extractor:
        binary = baseline_corrected < -k * noise

    Parameters:
      t_start, t_peak, t_end: frame indices
      y0, x0: centroid at peak (or start if motion)
      amp: peak magnitude (in raw intensity units, before uint16 clipping)
      sigma_yx: gaussian spread of the event footprint
      motion_dy_dx: centroid drift per frame during event window (rolling motion)
      overlap_group: optional grouping label for controlled overlaps
      mass_mg_truth: truth mass contribution for SWE closure truth
    """
    t_start: int
    t_peak: int
    t_end: int
    y0: float
    x0: float
    amp: float
    sigma_y: float
    sigma_x: float
    motion_dy: float = 0.0
    motion_dx: float = 0.0
    overlap_group: Optional[str] = None
    mass_mg_truth: float = 1.0


@dataclass(frozen=True)
class SyntheticConfig:
    T: int = 2357
    H: int = 75
    W: int = 90
    seed: int = 0

    # timebase
    t0_utc: str = "2026-01-01T00:00:00+00:00"
    dt_seconds: float = 1.0

    # baseline + nonuniformity
    baseline_level: float = 30000.0
    baseline_drift_per_frame: float = 0.05  # linear drift
    baseline_step_frame: Optional[int] = None
    baseline_step_delta: float = -200.0

    gradient_y: float = 2.0   # spatial gradient term
    gradient_x: float = -1.0  # spatial gradient term
    nonuniformity_scale: float = 30.0

    # noise
    noise_sigma_base: float = 8.0
    noise_sigma_spatial_scale: float = 3.0  # adds spatial heteroskedasticity
    noise_jump_frame: Optional[int] = None
    noise_jump_multiplier: float = 2.0

    # dead pixels
    dead_pixel_fraction: float = 0.002  # ~0.2%
    dead_pixel_value: int = 0

    # event injection defaults
    event_polarity: str = "negative"  # "negative" cooling vs "positive" heating
    event_shape: str = "gaussian"     # "gaussian" only in v1
    impulse_tau_rise: float = 2.0     # frames
    impulse_tau_decay: float = 15.0   # frames
    event_min_clip: float = -5000.0
    event_max_clip: float = 5000.0

    # processed SWE generation noise
    processed_swe_noise_mm: float = 0.1
    processed_rate_noise_mmhr: float = 0.5

    # particle table noise
    particle_time_jitter_s: float = 0.5
    particle_xy_jitter_px: float = 1.0
    particle_mass_rel_noise: float = 0.15
    particle_area_rel_noise: float = 0.20
    particle_temp_noise_c: float = 0.5

    # physical conversion (truth)
    sensing_area_mm2: float = 100.0  # arbitrary test area


# -----------------------------
# Helpers
# -----------------------------

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _frame_times(cfg: SyntheticConfig) -> pd.Series:
    t0 = pd.to_datetime(cfg.t0_utc, utc=True)
    # use seconds resolution; store as pandas Timestamp
    return pd.Series([t0 + pd.Timedelta(seconds=cfg.dt_seconds * i) for i in range(cfg.T)], name="t_utc")


def _mass_mg_to_swe_mm(mass_mg: float, sensing_area_mm2: float) -> float:
    # SWE(mm) = volume(mm^3)/area(mm^2); mass mg ~ volume mm^3 assuming rho_w = 1 mg/mm^3
    # Thus SWE(mm) = mass_mg / area_mm2
    if sensing_area_mm2 <= 0:
        return 0.0
    return float(mass_mg) / float(sensing_area_mm2)


def _impulse_profile(t: np.ndarray, t_peak: int, tau_rise: float, tau_decay: float) -> np.ndarray:
    """
    Smooth impulse: exponential rise to peak then exponential decay.
    t is integer frame indices for the event window [t_start..t_end].
    """
    t = t.astype(np.float64)
    tp = float(t_peak)
    rise = np.exp(-(np.maximum(0.0, tp - t)) / max(tau_rise, 1e-6))
    decay = np.exp(-(np.maximum(0.0, t - tp)) / max(tau_decay, 1e-6))
    prof = rise * decay
    prof = prof / max(prof.max(), 1e-9)
    return prof


def _gaussian_2d(H: int, W: int, cy: float, cx: float, sy: float, sx: float) -> np.ndarray:
    y = np.arange(H, dtype=np.float64)[:, None]
    x = np.arange(W, dtype=np.float64)[None, :]
    gy = ((y - cy) ** 2) / (2.0 * max(sy, 1e-6) ** 2)
    gx = ((x - cx) ** 2) / (2.0 * max(sx, 1e-6) ** 2)
    g = np.exp(-(gy + gx))
    # normalize peak to 1.0
    return g / max(g.max(), 1e-12)


def _event_centroid_at(spec: SyntheticEventSpec, t: int) -> Tuple[float, float]:
    # drift starts at t_start
    dt = float(t - spec.t_start)
    return spec.y0 + spec.motion_dy * dt, spec.x0 + spec.motion_dx * dt


def _event_mask_threshold(field: np.ndarray, thresh: float = 0.2) -> np.ndarray:
    # binary mask from gaussian template: field is nonnegative with peak 1.0
    return (field >= thresh).astype(np.uint8)


# -----------------------------
# Core generator
# -----------------------------

def generate_synthetic_session(
    *,
    out_dir: Path,
    cfg: SyntheticConfig,
    events: List[SyntheticEventSpec],
    dataset_path: str = "/data",
    write_particle: bool = True,
    write_processed: bool = True,
) -> Dict[str, Path]:
    """
    Generate a synthetic dataset to disk.

    Writes:
      out_dir/thermal.h5
      out_dir/particle.csv (optional)
      out_dir/processed.csv (optional)
      out_dir/truth_events.parquet
      out_dir/truth_plate_state.npz
      out_dir/truth_swe.parquet
      out_dir/manifest_truth.json

    Returns dict of important paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = _rng(cfg.seed)

    T, H, W = cfg.T, cfg.H, cfg.W

    # --- True plate state components ---
    # Baseline B(t)
    B = cfg.baseline_level + cfg.baseline_drift_per_frame * np.arange(T, dtype=np.float64)
    if cfg.baseline_step_frame is not None and 0 <= cfg.baseline_step_frame < T:
        B[cfg.baseline_step_frame :] += cfg.baseline_step_delta

    # Nonuniformity G(y,x): gradient + random smooth-ish field
    yy = np.arange(H, dtype=np.float64)[:, None]
    xx = np.arange(W, dtype=np.float64)[None, :]
    gradient = cfg.gradient_y * (yy - (H - 1) / 2.0) + cfg.gradient_x * (xx - (W - 1) / 2.0)
    # random field
    rand = rng.normal(0.0, 1.0, size=(H, W))
    G = cfg.nonuniformity_scale * rand + gradient

    # Noise map N(y,x): base + spatial term
    hetero = rng.uniform(0.0, 1.0, size=(H, W))
    N = cfg.noise_sigma_base + cfg.noise_sigma_spatial_scale * hetero

    # dead pixels
    dead_mask = np.zeros((H, W), dtype=bool)
    n_dead = int(round(cfg.dead_pixel_fraction * H * W))
    if n_dead > 0:
        idx = rng.choice(H * W, size=n_dead, replace=False)
        dead_mask.flat[idx] = True

    # noise jump over time
    noise_mult_t = np.ones((T,), dtype=np.float64)
    if cfg.noise_jump_frame is not None and 0 <= cfg.noise_jump_frame < T:
        noise_mult_t[cfg.noise_jump_frame :] *= float(cfg.noise_jump_multiplier)

    # --- Build synthetic cube ---
    cube = np.zeros((T, H, W), dtype=np.float64)

    # baseline + spatial field
    for t in range(T):
        cube[t] = B[t] + G

    # add events
    truth_rows = []
    # store truth masks as RLE-like list of indices (lightweight) for tests; can be expanded later
    # For determinism and simplicity, we store mask thresholded from gaussian template at peak.
    for j, spec in enumerate(events):
        t0, tp, t1 = int(spec.t_start), int(spec.t_peak), int(spec.t_end)
        t0 = max(0, min(T - 1, t0))
        tp = max(0, min(T - 1, tp))
        t1 = max(0, min(T - 1, t1))
        if not (t0 < tp < t1):
            raise ValueError(f"event {j} violates t_start < t_peak < t_end")

        tw = np.arange(t0, t1 + 1, dtype=int)
        prof = _impulse_profile(tw, tp, cfg.impulse_tau_rise, cfg.impulse_tau_decay)

        # polarity: negative-going cooling by default
        sign = -1.0 if cfg.event_polarity == "negative" else 1.0

        # peak mask template
        cy_peak, cx_peak = _event_centroid_at(spec, tp)
        tmpl_peak = _gaussian_2d(H, W, cy_peak, cx_peak, spec.sigma_y, spec.sigma_x)
        mask_peak = _event_mask_threshold(tmpl_peak, thresh=0.2)
        mask_idx = np.flatnonzero(mask_peak.ravel()).astype(np.int32)

        # inject time-varying signal (moving centroid if motion)
        for k, t in enumerate(tw.tolist()):
            cy, cx = _event_centroid_at(spec, t)
            tmpl = _gaussian_2d(H, W, cy, cx, spec.sigma_y, spec.sigma_x)
            amp = float(np.clip(spec.amp, cfg.event_min_clip, cfg.event_max_clip))
            cube[t] += sign * amp * prof[k] * tmpl

        # truth features (idealized)
        duration_s = (t1 - t0) * cfg.dt_seconds
        energy_proxy = float(abs(spec.amp)) * float(mask_peak.sum()) * float(prof.sum())
        truth_rows.append(
            {
                "truth_event_id": f"truth_{j:05d}",
                "frame_start": t0,
                "frame_peak": tp,
                "frame_end": t1,
                "centroid_y_peak": float(cy_peak),
                "centroid_x_peak": float(cx_peak),
                "sigma_y": float(spec.sigma_y),
                "sigma_x": float(spec.sigma_x),
                "amp": float(spec.amp),
                "polarity": cfg.event_polarity,
                "duration_s": float(duration_s),
                "mask_area_px_peak": int(mask_peak.sum()),
                "mask_flat_indices_peak": mask_idx,  # array type; parquet can store as object
                "energy_proxy_E_truth": float(energy_proxy),
                "mass_mg_truth": float(spec.mass_mg_truth),
                "overlap_group": spec.overlap_group,
            }
        )

    # add noise (Gaussian)
    for t in range(T):
        noise = rng.normal(0.0, 1.0, size=(H, W)) * (N * noise_mult_t[t])
        cube[t] += noise

    # dead pixels enforce
    if dead_mask.any():
        cube[:, dead_mask] = float(cfg.dead_pixel_value)

    # clip to uint16 range
    cube_uint16 = np.clip(np.rint(cube), 0, 65535).astype(np.uint16)

    # --- Write HDF5 ---
    h5_path = out_dir / "thermal.h5"
    with h5py.File(h5_path, "w") as f:
        # chunked 1×H×W, gzip level 9 (matches your example)
        ds = f.create_dataset(
            dataset_path.strip("/"),
            data=cube_uint16,
            dtype="uint16",
            chunks=(1, H, W),
            compression="gzip",
            compression_opts=9,
        )
        ds.attrs["synthetic"] = True
        ds.attrs["t0_utc"] = cfg.t0_utc
        ds.attrs["dt_seconds"] = float(cfg.dt_seconds)

    # --- Truth plate state artifacts ---
    truth_plate_npz = out_dir / "truth_plate_state.npz"
    np.savez_compressed(
        truth_plate_npz,
        baseline_B_t=B.astype(np.float64),
        nonuniformity_G_yx=G.astype(np.float32),
        noise_N_yx=N.astype(np.float32),
        dead_pixel_mask_yx=dead_mask.astype(np.bool_),
        noise_mult_t=noise_mult_t.astype(np.float32),
    )

    # --- Truth events table ---
    truth_events_df = pd.DataFrame(truth_rows)
    truth_events_path = out_dir / "truth_events.parquet"
    # store mask indices as json strings for portability
    truth_events_df["mask_flat_indices_peak_json"] = truth_events_df["mask_flat_indices_peak"].apply(lambda a: json.dumps(list(map(int, a))))
    truth_events_df = truth_events_df.drop(columns=["mask_flat_indices_peak"])
    truth_events_df.to_parquet(truth_events_path, index=False)

    # --- Truth SWE series (cumulative) ---
    times = _frame_times(cfg)
    swe_inc = np.zeros((T,), dtype=np.float64)
    for r in truth_rows:
        fpk = int(r["frame_peak"])
        swe_inc[fpk] += _mass_mg_to_swe_mm(float(r["mass_mg_truth"]), cfg.sensing_area_mm2)
    swe_true = np.cumsum(swe_inc)

    truth_swe_df = pd.DataFrame(
        {
            "t_utc": times.values,
            "frame_idx": np.arange(T, dtype=np.int32),
            "swe_true_mm": swe_true,
            "swe_inc_true_mm": swe_inc,
        }
    )
    truth_swe_path = out_dir / "truth_swe.parquet"
    truth_swe_df.to_parquet(truth_swe_path, index=False)

    # --- Optional processed table (macro) ---
    processed_path = out_dir / "processed.csv"
    if write_processed:
        # add observation noise
        swe_obs = swe_true + rng.normal(0.0, cfg.processed_swe_noise_mm, size=T)
        # rate mm/hr from finite diff
        rate = np.concatenate([[0.0], np.diff(swe_obs)]) / max(cfg.dt_seconds, 1e-9) * 3600.0
        rate += rng.normal(0.0, cfg.processed_rate_noise_mmhr, size=T)

        processed_df = pd.DataFrame(
            {
                "Date": pd.to_datetime(times).dt.date.astype(str),
                "Time": pd.to_datetime(times).dt.strftime("%H:%M:%S"),
                "SWE (mm)": swe_obs,
                "SWE Rate (mm/hr)": rate,
            }
        )
        processed_df.to_csv(processed_path, index=False)

    # --- Optional particle table (event-level) ---
    particle_path = out_dir / "particle.csv"
    if write_particle:
        # derive particle rows from truth events with noise
        rows = []
        for r in truth_rows:
            tp = int(r["frame_peak"])
            t_peak = pd.to_datetime(times.iloc[tp], utc=True)

            # jitter time and xy
            tj = rng.normal(0.0, cfg.particle_time_jitter_s)
            t_meas = t_peak + pd.Timedelta(seconds=float(tj))

            y = float(r["centroid_y_peak"]) + rng.normal(0.0, cfg.particle_xy_jitter_px)
            x = float(r["centroid_x_peak"]) + rng.normal(0.0, cfg.particle_xy_jitter_px)

            # derived features
            mass = float(r["mass_mg_truth"]) * (1.0 + rng.normal(0.0, cfg.particle_mass_rel_noise))
            area_px = float(r["mask_area_px_peak"])
            # pretend conversion: px^2 -> mm^2 (arbitrary scale in synthetic)
            area_mm2 = area_px * (0.01) * (1.0 + rng.normal(0.0, cfg.particle_area_rel_noise))

            evap_s = float(r["duration_s"]) * (1.0 + rng.normal(0.0, 0.10))
            temp_c = 0.0 + rng.normal(0.0, cfg.particle_temp_noise_c)

            rows.append(
                {
                    "Date": t_meas.date().isoformat(),
                    "Time": t_meas.strftime("%H:%M:%S.%f")[:-3],
                    "Time to Evaporate (Sec)": evap_s,
                    "Centroid location (x, y)": f"({x:.2f}, {y:.2f})",
                    "Mass (mg)": mass,
                    "Max Area (mm^2)": area_mm2,
                    "Temp (C)": temp_c,
                }
            )

        particle_df = pd.DataFrame(rows)
        particle_df.to_csv(particle_path, index=False)

    # --- Truth manifest ---
    manifest = {
        "schema_version": "synthetic_manifest_v1",
        "config": asdict(cfg),
        "dataset_path": dataset_path,
        "n_events": len(events),
        "paths": {
            "thermal_hdf5": str(h5_path),
            "truth_plate_state_npz": str(truth_plate_npz),
            "truth_events_parquet": str(truth_events_path),
            "truth_swe_parquet": str(truth_swe_path),
            "particle_csv": str(particle_path) if write_particle else None,
            "processed_csv": str(processed_path) if write_processed else None,
        },
    }
    _write_json(out_dir / "manifest_truth.json", manifest)

    return {
        "thermal_hdf5": h5_path,
        "truth_plate_state_npz": truth_plate_npz,
        "truth_events_parquet": truth_events_path,
        "truth_swe_parquet": truth_swe_path,
        "particle_csv": particle_path if write_particle else None,
        "processed_csv": processed_path if write_processed else None,
        "manifest_truth_json": out_dir / "manifest_truth.json",
    }


# -----------------------------
# Convenience presets
# -----------------------------

def default_event_set(cfg: SyntheticConfig) -> List[SyntheticEventSpec]:
    """
    Reasonable default set of events including:
      - isolated events
      - overlapping events (same time window, different centroids)
      - rolling motion event
    """
    T, H, W = cfg.T, cfg.H, cfg.W
    mid = T // 2

    return [
        SyntheticEventSpec(t_start=100, t_peak=110, t_end=150, y0=H * 0.3, x0=W * 0.3, amp=900.0, sigma_y=3.0, sigma_x=4.0, mass_mg_truth=2.0),
        SyntheticEventSpec(t_start=400, t_peak=410, t_end=460, y0=H * 0.6, x0=W * 0.5, amp=1200.0, sigma_y=4.0, sigma_x=4.0, mass_mg_truth=3.0),
        # overlap group
        SyntheticEventSpec(t_start=mid, t_peak=mid + 5, t_end=mid + 40, y0=H * 0.45, x0=W * 0.50, amp=1400.0, sigma_y=5.0, sigma_x=5.0, overlap_group="A", mass_mg_truth=4.0),
        SyntheticEventSpec(t_start=mid + 2, t_peak=mid + 7, t_end=mid + 42, y0=H * 0.50, x0=W * 0.55, amp=1000.0, sigma_y=4.0, sigma_x=6.0, overlap_group="A", mass_mg_truth=2.5),
        # rolling motion
        SyntheticEventSpec(t_start=mid + 200, t_peak=mid + 210, t_end=mid + 280, y0=H * 0.2, x0=W * 0.8, amp=1100.0, sigma_y=3.5, sigma_x=3.5, motion_dy=0.03, motion_dx=-0.04, mass_mg_truth=3.5),
    ]


if __name__ == "__main__":  # pragma: no cover
    # Example usage:
    cfg = SyntheticConfig(T=600, seed=42, baseline_step_frame=300, noise_jump_frame=450)
    events = default_event_set(cfg)
    paths = generate_synthetic_session(out_dir=Path("./synthetic_session"), cfg=cfg, events=events)
    print("Wrote synthetic session:")
    for k, v in paths.items():
        print(" ", k, "=>", v)