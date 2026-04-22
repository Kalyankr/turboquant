"""1-bit Quantized Johnson-Lindenstrauss (QJL) transform."""

from __future__ import annotations

import torch


def pack_signs(signs: torch.Tensor) -> torch.Tensor:
    """Pack {-1, +1} signs into compact uint8 (1 bit per sign).

    Parameters
    ----------
    signs : Tensor of int8, shape (n, d), values in {-1, 0, +1}.

    Returns
    -------
    packed : Tensor of uint8, shape (n, ceil(d/8)).
    """
    n, d = signs.shape
    # Map: -1 → 0, +1 → 1  (0 → 0)
    bits = (signs > 0).to(torch.uint8)  # (n, d)
    # Pad d to multiple of 8
    pad = (8 - d % 8) % 8
    if pad:
        bits = torch.nn.functional.pad(bits, (0, pad))
    # Reshape to (n, d_padded/8, 8) and pack
    bits = bits.view(n, -1, 8)
    weights = torch.tensor(
        [1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=signs.device
    )
    packed = (bits * weights).sum(dim=2).to(torch.uint8)
    return packed


def unpack_signs(packed: torch.Tensor, d: int) -> torch.Tensor:
    """Unpack uint8-packed bits back to {-1, +1} signs.

    Parameters
    ----------
    packed : Tensor of uint8, shape (n, n_bytes).
    d : int
        Original dimension.

    Returns
    -------
    signs : Tensor of int8, shape (n, d), values in {-1, +1}.
    """
    n = packed.shape[0]
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    # (n, n_bytes, 1) >> (8,) → (n, n_bytes, 8)
    unpacked = (
        (packed.unsqueeze(-1).to(torch.int32) >> shifts.to(torch.int32)) & 1
    ).to(torch.int8)
    unpacked = unpacked.view(n, -1)[:, :d]  # trim padding
    # Map 0 → -1, 1 → +1
    return unpacked * 2 - 1


def packed_sign_bytes(d: int) -> int:
    """Number of bytes needed to store d packed sign bits."""
    return (d + 7) // 8


def qjl_projection_matrix(
    d: int, seed: int = 0, device: torch.device | None = None
) -> torch.Tensor:
    """Generate a d×d random Gaussian projection matrix for QJL.

    Parameters
    ----------
    d : int
        Dimension.
    seed : int
        Seed for reproducibility.
    device : torch.device or None
        Target device.

    Returns
    -------
    S : Tensor of shape (d, d), random Gaussian.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    S = torch.randn(d, d, generator=gen, dtype=torch.float32)
    if device is not None:
        S = S.to(device)
    return S


def qjl_sign(residual: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """Compute 1-bit QJL: sign(residual @ S^T).

    Parameters
    ----------
    residual : Tensor of shape (n, d).
    S : Tensor of shape (d, d).

    Returns
    -------
    signs : Tensor of int8, shape (n, d), values in {-1, +1}.
    """
    return torch.sign(residual @ S.T).to(torch.int8)


def qjl_reconstruct(
    signs: torch.Tensor, gammas: torch.Tensor, S: torch.Tensor, d: int
) -> torch.Tensor:
    """Reconstruct the QJL correction term.

    correction_i = gamma_i * sqrt(pi/2) / d * S^T @ signs_i

    Parameters
    ----------
    signs : Tensor of int8, shape (n, d).
    gammas : Tensor of shape (n,), residual norms.
    S : Tensor of shape (d, d).
    d : int
        Dimension.

    Returns
    -------
    correction : Tensor of shape (n, d).
    """
    import math

    scale = math.sqrt(math.pi / 2) / d
    # (n, d) @ (d, d) = (n, d)
    correction = signs.float() @ S
    correction = correction * (gammas.unsqueeze(1) * scale)
    return correction
