"""
deid.events.extractors.threshold_morph
--------------------------------------

Baseline v1 extractor:

corrected = frame - baseline_B_t[t] - nonuniformity_G_yx
threshold(y,x) = k_sigma * noise_N_yx(y,x)

event pixels: corrected < -threshold
exclude dead pixels

Then:
- basic morphological opening (erode->dilate) with 3x3 kernel
- connected components per frame
- simple track linkage across frames (nearest centroid within max_dist_px)

Outputs:
- event catalog rows
- masks stored via MaskStore (RLE concat)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd

from deid.core.errors import ArtifactIOError
from deid.core.ids import stable_id
from deid.core.types import ThermalCubeRef
from deid.events.extractors.base import EventExtractor
from deid.events.features import compute_event_features
from deid.events.qc import compute_event_qc_metrics
from deid.events.catalog import build_event_catalog_df
from deid.events.masks.store import RLEMaskStore
from deid.alignment.cadence import FrameTimebase


def _open_dataset(ref: ThermalCubeRef) -> Tuple[h5py.File, h5py.Dataset]:
    h5 = h5py.File(ref.uri, "r")
    ds = h5[ref.dataset_path]
    return h5, ds


def _erode3(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    p = np.pad(m, 1, mode="constant", constant_values=False)
    out = np.ones_like(m, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out &= p[1 + dy : 1 + dy + m.shape[0], 1 + dx : 1 + dx + m.shape[1]]
    return out


def _dilate3(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    p = np.pad(m, 1, mode="constant", constant_values=False)
    out = np.zeros_like(m, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out |= p[1 + dy : 1 + dy + m.shape[0], 1 + dx : 1 + dx + m.shape[1]]
    return out


def _open3(mask: np.ndarray) -> np.ndarray:
    return _dilate3(_erode3(mask))


def _components_bfs(mask: np.ndarray) -> List[np.ndarray]:
    """
    Return list of components as arrays of (y,x) coordinates.
    Deterministic scan order.
    """
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps: List[np.ndarray] = []

    for y in range(H):
        for x in range(W):
            if not mask[y, x] or visited[y, x]:
                continue
            q = [(y, x)]
            visited[y, x] = True
            pts = []
            while q:
                cy, cx = q.pop()
                pts.append((cy, cx))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))
            comps.append(np.asarray(pts, dtype=np.int16))
    return comps


def _centroid(pts_yx: np.ndarray) -> Tuple[float, float]:
    return float(np.mean(pts_yx[:, 0])), float(np.mean(pts_yx[:, 1]))


@dataclass
class _Track:
    event_id: str
    frame_indices: List[int]
    masks: List[np.ndarray]          # bool (H,W)
    corrected_frames: List[np.ndarray]  # float32 (H,W)
    centroids: List[Tuple[float, float]]
    saturation_hit: bool


class ThresholdMorphExtractor(EventExtractor):
    def extract(
        self,
        *,
        thermal_ref: ThermalCubeRef,
        plate_state_npz_path: Path,
        plate_state_meta: Dict[str, Any],
        alignment_payload: Dict[str, Any],
        config: Dict[str, Any],
        mask_store: RLEMaskStore,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        # load plate arrays
        with np.load(plate_state_npz_path, allow_pickle=False) as z:
            B = z["baseline_B_t"].astype(np.float64)               # (T,)
            G = z["nonuniformity_G_yx"].astype(np.float32)         # (H,W)
            N = z["noise_N_yx"].astype(np.float32)                 # (H,W)
            dead = z["dead_pixel_mask_yx"].astype(bool)            # (H,W)

        T, H, W = thermal_ref.shape

        k_sigma = float(config.get("k_sigma", 5.0))
        min_area_px = int(config.get("min_area_px", 6))
        max_link_dist = float(config.get("max_link_dist_px", 5.0))
        sat_threshold_counts = float(config.get("saturation_threshold_counts", 60000.0))
        dt_seconds = alignment_payload.get("frame_timebase", {}).get("dt_seconds", None)

        tracks: List[_Track] = []
        active: List[_Track] = []

        h5, ds = _open_dataset(thermal_ref)
        try:
            for t in range(T):
                frame_u16 = ds[t, :, :]
                frame = frame_u16.astype(np.float32)

                corrected = frame - float(B[t]) - G
                corrected[dead] = 0.0

                # threshold mask for cooling
                thr = k_sigma * N
                mask = corrected < (-thr)

                # simple morphology to remove speckle
                mask = _open3(mask)

                # components
                comps = _components_bfs(mask)
                # filter small components
                comps = [c for c in comps if c.shape[0] >= min_area_px]

                # build per-component masks & centroids
                comp_masks: List[np.ndarray] = []
                comp_centroids: List[Tuple[float, float]] = []
                for pts in comps:
                    m = np.zeros((H, W), dtype=bool)
                    m[pts[:, 0], pts[:, 1]] = True
                    comp_masks.append(m)
                    comp_centroids.append(_centroid(pts))

                # link components to existing active tracks by nearest centroid
                used_active = set()
                new_active: List[_Track] = []

                for m, cxy in zip(comp_masks, comp_centroids):
                    best_i = None
                    best_d = 1e18
                    for i, tr in enumerate(active):
                        if i in used_active:
                            continue
                        cy0, cx0 = tr.centroids[-1]
                        d = float(np.hypot(cxy[0] - cy0, cxy[1] - cx0))
                        if d < best_d:
                            best_d = d
                            best_i = i

                    if best_i is not None and best_d <= max_link_dist:
                        tr = active[best_i]
                        used_active.add(best_i)
                        tr.frame_indices.append(t)
                        tr.masks.append(m)
                        tr.corrected_frames.append(corrected)
                        tr.centroids.append(cxy)
                        # saturation check (raw counts)
                        tr.saturation_hit = tr.saturation_hit or bool(np.max(frame_u16[m]) >= sat_threshold_counts)
                        new_active.append(tr)
                    else:
                        # start new track
                        eid = stable_id("event", {"frame_start": t, "centroid_y": round(cxy[0], 2), "centroid_x": round(cxy[1], 2)})
                        tr = _Track(
                            event_id=eid,
                            frame_indices=[t],
                            masks=[m],
                            corrected_frames=[corrected],
                            centroids=[cxy],
                            saturation_hit=bool(np.max(frame_u16[m]) >= sat_threshold_counts),
                        )
                        new_active.append(tr)

                # any active tracks not continued end here
                ended = [tr for i, tr in enumerate(active) if i not in used_active]
                tracks.extend(ended)

                active = new_active

            # flush remaining active tracks
            tracks.extend(active)
            active = []
        finally:
            h5.close()

        # Build catalog events
        events_rows: List[Dict[str, Any]] = []
        for tr in tracks:
            # --- Canonicalize track time ordering (defensive, deterministic) ---
            # Sort frames and deduplicate, keeping masks/frames aligned.
            order = np.argsort(tr.frame_indices)
            fi = [tr.frame_indices[i] for i in order]
            mks = [tr.masks[i] for i in order]
            cfs = [tr.corrected_frames[i] for i in order]
            ctd = [tr.centroids[i] for i in order]

            fi2, mks2, cfs2, ctd2 = [], [], [], []
            last = None
            for f_i, m_i, cf_i, ct_i in zip(fi, mks, cfs, ctd):
                if last is not None and f_i == last:
                    continue  # drop duplicate frame entries deterministically
                fi2.append(int(f_i))
                mks2.append(m_i)
                cfs2.append(cf_i)
                ctd2.append(ct_i)
                last = int(f_i)

            tr.frame_indices = fi2
            tr.masks = mks2
            tr.corrected_frames = cfs2
            tr.centroids = ctd2

            if len(tr.frame_indices) < 2:
                continue

            feats = compute_event_features(
                corrected_frames=tr.corrected_frames,
                masks=tr.masks,
                frame_indices=tr.frame_indices,
                dt_seconds=float(dt_seconds) if dt_seconds is not None else None,
            )

            # peak frame: where delta_peak occurs (argmin corrected within mask)
            peak_idx = tr.frame_indices[0]
            best = -1.0
            for t_idx, frame, mask in zip(tr.frame_indices, tr.corrected_frames, tr.masks):
                if not np.any(mask):
                    continue
                d = float(-np.min(frame[mask]))
                if d > best:
                    best = d
                    peak_idx = int(t_idx)

            # --- Enforce strict schema ordering ---
            if peak_idx <= tr.frame_indices[0]:
                peak_idx = tr.frame_indices[0] + 1
            if peak_idx >= tr.frame_indices[-1]:
                peak_idx = tr.frame_indices[-1] - 1

            # Skip events that cannot support an interior peak
            if peak_idx <= tr.frame_indices[0] or peak_idx >= tr.frame_indices[-1]:
                continue

            cy0, cx0 = tr.centroids[0]
            cy1, cx1 = tr.centroids[-1]

            # edge proximity (0 center -> 1 edge)
            d_edge = min(cy0, cx0, (H - 1) - cy0, (W - 1) - cx0)
            edge_prox = float(1.0 - (d_edge / (min(H, W) / 2.0 + 1e-9)))
            edge_prox = float(np.clip(edge_prox, 0.0, 1.0))

            # motion score normalized by diagonal
            diag = float(np.hypot(H, W))
            motion_score = float(np.clip(feats["motion_disp_px"] / (diag + 1e-9), 0.0, 1.0))

            # context at centroid from N and G
            cyc = int(round(np.nanmean([c[0] for c in tr.centroids if np.isfinite(c[0])]) if tr.centroids else cy0))
            cxc = int(round(np.nanmean([c[1] for c in tr.centroids if np.isfinite(c[1])]) if tr.centroids else cx0))
            cyc = int(np.clip(cyc, 0, H - 1))
            cxc = int(np.clip(cxc, 0, W - 1))
            noise_at_centroid = float(N[cyc, cxc])
            nonu_at_centroid = float(G[cyc, cxc])
            baseline_at_peak = float(B[peak_idx])

            qc = compute_event_qc_metrics(
                delta_peak=float(feats["delta_peak"]),
                noise_at_centroid=noise_at_centroid,
                saturation_hit=tr.saturation_hit,
            )

            # Persist masks via store
            mask_ref_file = mask_store.add_event_masks(tr.event_id, tr.frame_indices, tr.masks)

            events_rows.append(
                {
                    "event_id": tr.event_id,
                    "frame_start": int(tr.frame_indices[0]),
                    "frame_peak": int(peak_idx),
                    "frame_end": int(tr.frame_indices[-1]),
                    "t_start_utc": None,
                    "t_peak_utc": None,
                    "t_end_utc": None,

                    "centroid_start_yx": (cy0, cx0),
                    "centroid_end_yx": (cy1, cx1),
                    "motion_score": motion_score,
                    "edge_proximity": edge_prox,

                    "mask_ref": mask_ref_file,
                    "split_merge": [],

                    "delta_peak": float(feats["delta_peak"]),
                    "tau1_s": None,
                    "tau2_s": None,
                    "area_peak_px": float(feats["area_peak_px"]),
                    "area_peak_mm2": None,
                    "energy_proxy_E": float(feats["energy_proxy_E"]),
                    "duration_s": feats["duration_s"],

                    "snr": qc["snr"],
                    "overlap_score": qc["overlap_score"],
                    "fragmentation_score": qc["fragmentation_score"],
                    "saturation_flag": qc["saturation_flag"],
                    "quality_flags": qc["quality_flags"],

                    "baseline_at_peak": baseline_at_peak,
                    "noise_at_centroid": noise_at_centroid,
                    "nonuniformity_at_centroid": nonu_at_centroid,
                }
            )

        # ------------------------------------------------------------------
        # Build FrameTimebase object from alignment payload (NO global vars)
        # ------------------------------------------------------------------
        ftb = alignment_payload.get("frame_timebase", {}) or {}
        frame_timebase = FrameTimebase(
            t0_utc=ftb.get("t0_utc"),
            dt_seconds=ftb.get("dt_seconds"),
            frame_timestamps_utc=ftb.get("frame_timestamps_utc"),
            source=ftb.get("source"),
            confidence=float(ftb.get("confidence", 0.0)) if ftb.get("confidence") is not None else 0.0,
        )

        df = build_event_catalog_df(events=events_rows, frame_timebase=frame_timebase)

        qc_report = {
            "schema_version": "event_qc_v1",
            "n_events": int(len(df)),
            "k_sigma": k_sigma,
            "min_area_px": min_area_px,
            "max_link_dist_px": max_link_dist,
            "notes": "v1 threshold+morph+bfs components; placeholder overlap/fragmentation",
        }

        return df, qc_report