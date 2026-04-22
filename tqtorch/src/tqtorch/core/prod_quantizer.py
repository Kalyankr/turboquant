"""Inner-product-preserving TurboQuant quantizer (Algorithm 2).

Two-stage approach:
  Stage 1: (bits-1)-bit MSE quantization
  Stage 2: 1-bit QJL on the residual for unbiased inner products
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from tqtorch.core.mse_quantizer import MSEQuantizer, QuantizedMSE
from tqtorch.core.packed import (
    pack_indices,
    unpack_indices,
    packed_bytes_per_vector,
)
from tqtorch.core.qjl import (
    qjl_projection_matrix,
    qjl_sign,
    pack_signs,
    unpack_signs,
    packed_sign_bytes,
)


@dataclass
class QuantizedIP:
    """Compact representation of inner-product-quantized vectors."""

    mse_packed: torch.Tensor  # (n, n_bytes) uint8 — MSE stage indices
    mse_norms: torch.Tensor  # (n,) float16
    qjl_packed: torch.Tensor  # (n, ceil(d/8)) uint8 — bit-packed signs
    gammas: torch.Tensor  # (n,) float16 — residual norms
    dim: int
    bits: int
    seed: int


class InnerProductQuantizer:
    """TurboQuant optimized for unbiased inner-product estimation.

    Parameters
    ----------
    dim : int
        Vector dimensionality.
    bits : int
        Total bits per coordinate (>= 2). Stage 1 uses (bits-1), stage 2 uses 1.
    seed : int
        Random seed.
    device : torch.device or None
        Target device.
    """

    def __init__(
        self, dim: int, bits: int, seed: int = 42, device: torch.device | None = None
    ):
        if bits < 2:
            raise ValueError("InnerProductQuantizer requires bits >= 2")
        self.dim = dim
        self.bits = bits
        self.seed = seed
        self.device = device or torch.device("cpu")

        self.mse_q = MSEQuantizer(dim, bits - 1, seed=seed, device=self.device)
        # QJL uses a different seed to ensure independence
        self.S = qjl_projection_matrix(dim, seed=seed + 1_000_000, device=self.device)

    def quantize(self, x: torch.Tensor) -> QuantizedIP:
        """Quantize vectors.

        Parameters
        ----------
        x : Tensor of shape (n, d) or (d,).
        """
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)
        x = x.to(self.device, dtype=torch.float32)

        # Stage 1: MSE at (bits-1) bits
        mse_qt = self.mse_q.quantize(x)
        x_mse = self.mse_q.dequantize(mse_qt)

        # Residual — store gamma as float16
        residual = x - x_mse
        gammas = residual.norm(dim=1).to(torch.float16)

        # Stage 2: QJL sign bits (bit-packed)
        signs = qjl_sign(residual, self.S)
        qjl_packed = pack_signs(signs)

        return QuantizedIP(
            mse_packed=mse_qt.packed_indices,
            mse_norms=mse_qt.norms,
            qjl_packed=qjl_packed,
            gammas=gammas,
            dim=self.dim,
            bits=self.bits,
            seed=self.seed,
        )

    def dequantize(self, qt: QuantizedIP) -> torch.Tensor:
        """Reconstruct vectors (MSE component only).

        The QJL bits are used for inner-product estimation, not
        point-wise reconstruction.  Adding the noisy 1-bit QJL
        correction actually *increases* MSE, so dequantize returns
        only the (bits-1)-bit MSE reconstruction — matching the
        paper's intent for Algorithm 2.
        """
        mse_qt = QuantizedMSE(
            packed_indices=qt.mse_packed,
            norms=qt.mse_norms,
            dim=qt.dim,
            bits=qt.bits - 1,
            seed=qt.seed,
        )
        return self.mse_q.dequantize(mse_qt)

    def bytes_per_vector(self) -> int:
        """Storage bytes per vector."""
        mse_bytes = packed_bytes_per_vector(self.dim, self.bits - 1)
        qjl_bytes = packed_sign_bytes(self.dim)  # 1 bit per sign, packed
        norm_bytes = 2  # float16 norm
        gamma_bytes = 2  # float16 gamma
        return mse_bytes + qjl_bytes + norm_bytes + gamma_bytes

    def estimate_inner_products(
        self, qt: QuantizedIP, queries: torch.Tensor
    ) -> torch.Tensor:
        """Unbiased inner-product estimation using MSE + QJL.

        Computes:  <x, y> ≈ <x̂_mse, y> + sqrt(π/2)/d · ||r|| · Σ_j z_j·(S_j · y)

        Parameters
        ----------
        qt : QuantizedIP
        queries : Tensor of shape (nq, d).

        Returns
        -------
        scores : Tensor of shape (nq, n).
        """
        x_hat = self.dequantize(qt)  # (n, d) — MSE-only
        mse_ip = queries @ x_hat.T  # (nq, n)

        # QJL correction: unbiased residual IP estimator
        signs = unpack_signs(qt.qjl_packed, qt.dim)  # (n, d) in {-1, +1}
        Sy = queries @ self.S.T  # (nq, d)
        qjl_ip = signs.float() @ Sy.T  # (n, nq)
        scale = math.sqrt(math.pi / 2) / self.dim
        gammas = qt.gammas.float()  # (n,)
        qjl_correction = scale * gammas.unsqueeze(0) * qjl_ip.T  # (nq, n)

        return mse_ip + qjl_correction


# ── Functional API ────────────────────────────────────────────────────────────


def ip_quantize(x: torch.Tensor, bits: int = 4, seed: int = 42) -> QuantizedIP:
    """Quantize vectors for unbiased inner-product estimation (functional API)."""
    d = x.shape[-1]
    q = InnerProductQuantizer(d, bits, seed=seed, device=x.device)
    return q.quantize(x)


def estimate_inner_product(qt: QuantizedIP, y: torch.Tensor) -> torch.Tensor:
    """Estimate inner products between quantized vectors and query vectors.

    Uses the unbiased MSE + QJL estimator from Algorithm 2.

    Parameters
    ----------
    qt : QuantizedIP from ip_quantize.
    y : Tensor of shape (n, d) or (d,).

    Returns
    -------
    estimates : Tensor of shape (n,) — one IP estimate per vector pair.
    """
    q = InnerProductQuantizer(
        qt.dim, qt.bits, seed=qt.seed, device=qt.mse_packed.device
    )
    if y.dim() == 1:
        y = y.unsqueeze(0)
    scores = q.estimate_inner_products(qt, y)  # (1, n) or (n, n)
    if scores.shape[0] == 1:
        return scores.squeeze(0)
    # Diagonal: element-wise IP for paired vectors
    return scores.diagonal()
