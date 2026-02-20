"""
deid.events.masks.rle
---------------------

Deterministic run-length encoding (RLE) for 2D boolean masks.

Encoding is performed on row-major flattened array.

We represent each 2D mask as two uint32 arrays:
- starts: start indices of True runs
- lengths: lengths of each run

This supports compact storage and random access decoding.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def encode_mask_rle(mask_yx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode boolean mask into (starts, lengths) uint32 arrays.
    """
    m = np.asarray(mask_yx, dtype=bool).ravel(order="C")
    if m.size == 0:
        return np.zeros((0,), dtype=np.uint32), np.zeros((0,), dtype=np.uint32)

    # Find transitions in m
    # We want runs where m == True
    diff = np.diff(m.astype(np.int8), prepend=0, append=0)
    run_starts = np.where(diff == 1)[0]
    run_ends = np.where(diff == -1)[0]
    lengths = run_ends - run_starts

    return run_starts.astype(np.uint32), lengths.astype(np.uint32)


def decode_mask_rle(starts: np.ndarray, lengths: np.ndarray, shape_yx: Tuple[int, int]) -> np.ndarray:
    """
    Decode (starts, lengths) into boolean mask of shape (H,W).
    """
    H, W = int(shape_yx[0]), int(shape_yx[1])
    n = H * W
    out = np.zeros((n,), dtype=bool)

    s = np.asarray(starts, dtype=np.uint32)
    l = np.asarray(lengths, dtype=np.uint32)
    for a, ln in zip(s.tolist(), l.tolist()):
        a_i = int(a)
        ln_i = int(ln)
        if ln_i <= 0:
            continue
        b_i = min(n, a_i + ln_i)
        if 0 <= a_i < n:
            out[a_i:b_i] = True

    return out.reshape((H, W), order="C")


def rle_roundtrip_ok(mask_yx: np.ndarray) -> bool:
    s, l = encode_mask_rle(mask_yx)
    rec = decode_mask_rle(s, l, mask_yx.shape)
    return np.array_equal(rec, np.asarray(mask_yx, dtype=bool))