"""MSE-optimal TurboQuant quantizer (Algorithm 1).

Quantizes vectors by:
1. Extracting and storing the norm
2. Normalizing to the unit sphere
3. Applying a seeded random rotation
4. Scalar-quantizing each coordinate via Lloyd-Max boundaries
5. Packing indices into compact bit representation
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tqtorch.core.codebook import get_codebook
from tqtorch.core.packed import (
    pack_indices,
    unpack_indices,
    packed_bytes_per_vector,
)
from tqtorch.core.rotation import (
    random_rotation_matrix,
    random_rotate,
    random_rotate_inverse,
)


@dataclass
class QuantizedMSE:
    """Compact representation of MSE-quantized vectors."""

    packed_indices: torch.Tensor  # (n, n_bytes) uint8
    norms: torch.Tensor  # (n,) float16 — original vector norms
    dim: int
    bits: int
    seed: int


class MSEQuantizer:
    """Reusable MSE-optimal TurboQuant quantizer.

    Rotation matrix and codebook are created once and reused for all vectors.

    Parameters
    ----------
    dim : int
        Vector dimensionality.
    bits : int
        Bits per coordinate (1-8).
    seed : int
        Random seed for the rotation matrix.
    device : torch.device or None
        Target device.
    """

    def __init__(
        self, dim: int, bits: int, seed: int = 42, device: torch.device | None = None
    ):
        if not 1 <= bits <= 8:
            raise ValueError("bits must be 1-8")
        self.dim = dim
        self.bits = bits
        self.seed = seed
        self.device = device or torch.device("cpu")

        self.Pi = random_rotation_matrix(dim, seed=seed, device=self.device)
        centroids, boundaries = get_codebook(dim, bits)
        self.centroids = centroids.to(self.device)
        self.boundaries = boundaries.to(self.device)

    def quantize(self, x: torch.Tensor) -> QuantizedMSE:
        """Quantize vectors.

        Parameters
        ----------
        x : Tensor of shape (n, d) or (d,).

        Returns
        -------
        QuantizedMSE
        """
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)
        x = x.to(self.device, dtype=torch.float32)

        # Extract norms and normalize — store as float16
        norms = x.norm(dim=1).to(torch.float16)
        safe_norms = norms.float().clamp(min=1e-12)
        x_unit = x / safe_norms.unsqueeze(1)

        # Rotate
        y = random_rotate(x_unit, self.Pi)

        # Scalar quantize via searchsorted
        indices = torch.searchsorted(self.boundaries, y).to(torch.uint8)

        # Pack
        packed = pack_indices(indices, self.bits)

        return QuantizedMSE(
            packed_indices=packed,
            norms=norms,
            dim=self.dim,
            bits=self.bits,
            seed=self.seed,
        )

    def dequantize(self, qt: QuantizedMSE) -> torch.Tensor:
        """Reconstruct vectors from quantized representation.

        Returns
        -------
        x_hat : Tensor of shape (n, d).
        """
        indices = unpack_indices(qt.packed_indices, qt.bits, qt.dim).to(self.device)
        y_hat = self.centroids[indices.long()]
        x_hat = random_rotate_inverse(y_hat, self.Pi)
        x_hat = x_hat * qt.norms.to(self.device).unsqueeze(1)
        return x_hat

    def bytes_per_vector(self) -> int:
        """Storage bytes per vector (packed indices + float16 norm)."""
        return packed_bytes_per_vector(self.dim, self.bits) + 2


# ── Functional API ────────────────────────────────────────────────────────────


def mse_quantize(x: torch.Tensor, bits: int = 3, seed: int = 42) -> QuantizedMSE:
    """Quantize vectors using MSE-optimal TurboQuant (functional API).

    Parameters
    ----------
    x : Tensor of shape (n, d) or (d,).
    bits : int
        Bits per coordinate.
    seed : int
        Rotation matrix seed.
    """
    d = x.shape[-1]
    q = MSEQuantizer(d, bits, seed=seed, device=x.device)
    return q.quantize(x)


def mse_dequantize(qt: QuantizedMSE) -> torch.Tensor:
    """Reconstruct vectors from QuantizedMSE."""
    q = MSEQuantizer(qt.dim, qt.bits, seed=qt.seed, device=qt.packed_indices.device)
    return q.dequantize(qt)
