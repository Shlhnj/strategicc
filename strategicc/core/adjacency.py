"""
strategicc/core/adjacency.py  —  1b. Adjacency multiplier
----------------------------------------------------
Computes the fraction of each cell's 8 neighbours that belong to each class,
then applies this as a multiplier to transition probabilities in the engine.

Design
------
* `compute_neighbor_fractions` is a pure NumPy function with no side-effects.
  It is called once per timestep (on the frozen start-of-step map) and its
  output is consumed by the engine's transition loop.

* The adjacency effect on effective probability is applied in engine.py:

    Strict expansion groups (must touch target class to fire):
        adj_mult = adj_frac * ADJACENCY_STRENGTH          (0 if no neighbours)

    Diffuse groups (can fire anywhere, boosted near target):
        adj_mult = 1.0 + adj_frac * ADJACENCY_STRENGTH   (baseline = 1)
"""

from __future__ import annotations
import numpy as np


def compute_neighbor_fractions(
    lulc_map: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """
    Compute the fraction of valid 8-neighbours belonging to each class.

    Parameters
    ----------
    lulc_map  : uint8 array, shape (rows, cols), values in [1 .. n_classes]
    n_classes : total number of classes (max class id)

    Returns
    -------
    fracs : float32 array, shape (rows, cols, n_classes + 1)
            fracs[r, c, k] = fraction of (r,c)'s neighbours that are class k.
            Index 0 is unused (class IDs start at 1).

    Notes
    -----
    Edge cells have fewer than 8 neighbours; the denominator is adjusted
    automatically so fractions always sum to ≤ 1 across classes.
    """
    rows, cols = lulc_map.shape
    count = np.zeros((rows, cols, n_classes + 1), dtype=np.float32)
    valid = np.zeros((rows, cols),                dtype=np.float32)

    directions = [(-1, -1), (-1, 0), (-1, 1),
                  ( 0, -1),          ( 0, 1),
                  ( 1, -1), ( 1, 0), ( 1, 1)]

    for dr, dc in directions:
        # Source slice (the neighbour)
        r0s = max(0, -dr);  r1s = rows if dr <= 0 else rows - dr
        c0s = max(0, -dc);  c1s = cols if dc <= 0 else cols - dc
        # Destination slice (the centre cell being described)
        r0d = max(0,  dr);  r1d = rows if dr >= 0 else rows + dr
        c0d = max(0,  dc);  c1d = cols if dc >= 0 else cols + dc

        patch = lulc_map[r0s:r1s, c0s:c1s]          # shape (H, W)

        # One-hot encode all classes simultaneously  →  (H, W, n_classes+1)
        one_hot = (
            patch[:, :, np.newaxis]
            == np.arange(n_classes + 1, dtype=np.uint8)
        ).astype(np.float32)

        count[r0d:r1d, c0d:c1d] += one_hot
        valid[r0d:r1d, c0d:c1d] += 1.0

    fracs = count / np.maximum(valid[:, :, np.newaxis], 1.0)
    return fracs
