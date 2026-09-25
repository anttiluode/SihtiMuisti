"""Orthonormal 2-D Haar octaves: the Sihti residue identity, critically sampled.

Sihti keeps every octave residue of an image so the stack sums back losslessly.
Here the same identity is written with the orthonormal Haar wavelet so that the
storage cost is exact: a 64x64 frame is exactly 4,096 coefficients, and dropping
its finest octave leaves exactly the 1,024-coefficient approximation one level
down. "Coarsening a memory by one octave" is therefore a precise operation with
a precise saving, and the lost squared error equals the energy of the dropped
detail band (Parseval).

Conventions
-----------
approx(x, level) returns the orthonormal approximation image at that level:
pixel value = 2**level * block mean. Level 0 is the image itself.
"""
from __future__ import annotations

import numpy as np


def analyze1(a: np.ndarray):
    """One orthonormal Haar step. Returns (approx, (h, v, d))."""
    x00 = a[0::2, 0::2]
    x01 = a[0::2, 1::2]
    x10 = a[1::2, 0::2]
    x11 = a[1::2, 1::2]
    ap = (x00 + x01 + x10 + x11) / 2.0
    h = (x00 - x01 + x10 - x11) / 2.0  # horizontal change (vertical stripes)
    v = (x00 + x01 - x10 - x11) / 2.0  # vertical change (horizontal stripes)
    d = (x00 - x01 - x10 + x11) / 2.0
    return ap, (h, v, d)


def synthesize1(ap: np.ndarray, hvd) -> np.ndarray:
    h, v, d = hvd
    n = ap.shape[0]
    out = np.empty((2 * n, 2 * n), dtype=np.float64)
    out[0::2, 0::2] = (ap + h + v + d) / 2.0
    out[0::2, 1::2] = (ap - h + v - d) / 2.0
    out[1::2, 0::2] = (ap + h - v - d) / 2.0
    out[1::2, 1::2] = (ap - h - v + d) / 2.0
    return out


def approx(x: np.ndarray, level: int) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    for _ in range(level):
        a, _ = analyze1(a)
    return a


def decompose(x: np.ndarray, levels: int):
    """Full decomposition. Returns (approx_at_levels, details) where
    details[j] for j = 1..levels is an array (3, n_j, n_j)."""
    a = np.asarray(x, dtype=np.float64)
    details = {}
    for j in range(1, levels + 1):
        a, (h, v, d) = analyze1(a)
        details[j] = np.stack([h, v, d])
    return a, details


def reconstruct(a: np.ndarray, details: dict, levels: int) -> np.ndarray:
    for j in range(levels, 0, -1):
        dj = details[j]
        a = synthesize1(a, (dj[0], dj[1], dj[2]))
    return a


def upsample_view(a_level: np.ndarray, level: int) -> np.ndarray:
    """Display an approximation at level `level` as a full-size picture
    (block means, i.e. what the memory actually knows)."""
    if level == 0:
        return a_level.copy()
    f = 2 ** level
    return np.kron(a_level / f, np.ones((f, f)))


def coeff_count(size: int, level: int) -> int:
    n = size >> level
    return n * n
