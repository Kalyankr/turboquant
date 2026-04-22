"""Random orthogonal rotation matrices via QR decomposition."""

from __future__ import annotations

import torch


def random_rotation_matrix(d: int, seed: int = 0, device: torch.device | None = None) -> torch.Tensor:
    """Generate a d×d random orthogonal matrix (seeded, deterministic).

    Uses QR decomposition of a Gaussian random matrix.

    Parameters
    ----------
    d : int
        Dimension.
    seed : int
        Seed for reproducibility.
    device : torch.device or None
        Target device (default: CPU).

    Returns
    -------
    Pi : Tensor of shape (d, d), orthogonal.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    G = torch.randn(d, d, generator=gen, dtype=torch.float32)
    Q, R = torch.linalg.qr(G)
    # Fix sign ambiguity so output is deterministic across platforms
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)
    if device is not None:
        Q = Q.to(device)
    return Q


def random_rotate(x: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
    """Apply rotation: y = x @ Pi^T (works for batched x of shape (n, d))."""
    return x @ Pi.T


def random_rotate_inverse(y: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
    """Undo rotation: x = y @ Pi (since Pi is orthogonal, Pi^{-1} = Pi^T)."""
    return y @ Pi
