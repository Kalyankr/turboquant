"""Lloyd-Max codebooks for Gaussian coordinates.

Codebooks are precomputed for N(0, 1/d) at various bit-widths and cached.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch


def _lloyd_max_gaussian_np(n_levels: int, sigma: float, max_iter: int = 300):
    """Compute Lloyd-Max centroids/boundaries for N(0, sigma^2) using math stdlib.

    Returns (centroids, boundaries) as plain Python lists.
    """
    # Use pure-Python normal PDF/CDF via math.erf to avoid scipy dependency
    _INF = float("inf")

    def _phi(z: float) -> float:
        """Standard normal CDF."""
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def _pdf(z: float) -> float:
        """Standard normal PDF."""
        return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

    centroids = [sigma * (-3 + 6 * i / (n_levels - 1)) if n_levels > 1 else 0.0 for i in range(n_levels)]

    for _ in range(max_iter):
        # Boundaries = midpoints
        boundaries = [-_INF] + [(centroids[i] + centroids[i + 1]) / 2 for i in range(n_levels - 1)] + [_INF]
        new_centroids = []
        for i in range(n_levels):
            lo, hi = boundaries[i], boundaries[i + 1]
            lo_z = lo / sigma if lo != -_INF else -20.0
            hi_z = hi / sigma if hi != _INF else 20.0
            num = sigma * (_pdf(lo_z) - _pdf(hi_z))
            den = _phi(hi_z) - _phi(lo_z)
            new_centroids.append(num / den if den > 1e-15 else (lo + hi) / 2)
        if all(abs(a - b) < 1e-12 for a, b in zip(centroids, new_centroids)):
            break
        centroids = new_centroids

    boundaries = [-_INF] + [(centroids[i] + centroids[i + 1]) / 2 for i in range(n_levels - 1)] + [_INF]
    return centroids, boundaries


@lru_cache(maxsize=64)
def get_codebook(d: int, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (centroids, boundaries) tensors for N(0, 1/d) at the given bit-width.

    Results are cached so repeated calls with the same (d, bits) are free.

    Parameters
    ----------
    d : int
        Dimension (determines sigma = 1/sqrt(d)).
    bits : int
        Bit-width (1-8).

    Returns
    -------
    centroids : Tensor of shape (2^bits,)
    boundaries : Tensor of shape (2^bits - 1,)  — inner boundaries only.
    """
    n_levels = 1 << bits
    sigma = 1.0 / math.sqrt(d)
    c_list, b_list = _lloyd_max_gaussian_np(n_levels, sigma)
    centroids = torch.tensor(c_list, dtype=torch.float32)
    # Inner boundaries only (drop -inf and +inf)
    boundaries = torch.tensor(b_list[1:-1], dtype=torch.float32)
    return centroids, boundaries
