"""
deid.events.masks.store
-----------------------

Mask store for per-event, per-frame masks with deterministic random access.

Layout (under intermediate/event_masks/):
  index.json
  <event_id>.npz

Each event npz stores:
  frame_indices: (F,) int32
  offsets: (F+1,) int32  (boundaries into starts/lengths)
  starts: (K,) uint32
  lengths: (K,) uint32
  H: scalar int32
  W: scalar int32

Decoding a single frame i:
  a = offsets[i], b = offsets[i+1]
  starts[a:b], lengths[a:b]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from deid.core.errors import ArtifactIOError
from deid.storage.io import write_json, write_npz
from deid.events.masks.rle import encode_mask_rle


@dataclass
class MaskStoreIndexEntry:
    event_id: str
    file: str
    encoding: str  # "rle_concat_v1"
    n_frames: int
    shape_yx: Tuple[int, int]


class RLEMaskStore:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._index: Dict[str, Any] = {
            "schema_version": "event_masks_index_v1",
            "entries": [],
        }

    def add_event_masks(self, event_id: str, frame_indices: List[int], masks_yx: List[np.ndarray]) -> str:
        if len(frame_indices) != len(masks_yx):
            raise ArtifactIOError("frame_indices and masks length mismatch", details={"event_id": event_id})

        if not masks_yx:
            raise ArtifactIOError("No masks provided for event", details={"event_id": event_id})

        H, W = masks_yx[0].shape
        for m in masks_yx:
            if m.shape != (H, W):
                raise ArtifactIOError("Inconsistent mask shapes within event", details={"event_id": event_id})

        starts_all: List[np.ndarray] = []
        lengths_all: List[np.ndarray] = []
        offsets = [0]

        total = 0
        for m in masks_yx:
            s, l = encode_mask_rle(m)
            starts_all.append(s)
            lengths_all.append(l)
            total += int(s.size)
            offsets.append(total)

        starts_cat = np.concatenate(starts_all, axis=0).astype(np.uint32) if total > 0 else np.zeros((0,), dtype=np.uint32)
        lengths_cat = np.concatenate(lengths_all, axis=0).astype(np.uint32) if total > 0 else np.zeros((0,), dtype=np.uint32)
        offsets_arr = np.asarray(offsets, dtype=np.int32)
        frames_arr = np.asarray(frame_indices, dtype=np.int32)

        fname = f"{event_id}.npz"
        write_npz(
            self.root_dir / fname,
            frame_indices=frames_arr,
            offsets=offsets_arr,
            starts=starts_cat,
            lengths=lengths_cat,
            H=np.asarray([H], dtype=np.int32),
            W=np.asarray([W], dtype=np.int32),
        )

        self._index["entries"].append(
            {
                "event_id": event_id,
                "file": fname,
                "encoding": "rle_concat_v1",
                "n_frames": int(len(frame_indices)),
                "shape_yx": [int(H), int(W)],
            }
        )
        return fname

    def finalize(self) -> None:
        # deterministic ordering
        self._index["entries"] = sorted(self._index["entries"], key=lambda e: e["event_id"])
        write_json(self.root_dir / "index.json", self._index)