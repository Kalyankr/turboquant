"""Bit-packing utilities for compact quantized storage."""

from __future__ import annotations

import torch


def pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack integer indices (0..2^bits-1) into a compact uint8 byte tensor.

    Parameters
    ----------
    indices : Tensor of uint8, shape (n, d)
        Values in [0, 2^bits).
    bits : int
        Bit-width per index (1-8).

    Returns
    -------
    packed : Tensor of uint8, shape (n, ceil(d * bits / 8)).
    """
    if bits == 8:
        return indices.to(torch.uint8)

    n, d = indices.shape

    # Fast vectorized paths for the common power-of-two bit-widths that pack
    # an integer number of indices into one byte (bits in {1, 2, 4}).
    if bits in (1, 2, 4):
        per_byte = 8 // bits
        pad = (per_byte - d % per_byte) % per_byte
        idx = indices.to(torch.uint8)
        if pad:
            idx = torch.nn.functional.pad(idx, (0, pad))
        # Group consecutive indices into one byte.
        idx = idx.view(n, -1, per_byte)  # (n, n_bytes, per_byte)
        shifts = (torch.arange(per_byte, device=indices.device, dtype=torch.uint8) * bits)
        # Bit-shift each lane left and OR-reduce across the per_byte axis.
        return ((idx << shifts) & 0xFF).sum(dim=2).to(torch.uint8)

    total_bits = d * bits
    n_bytes = (total_bits + 7) // 8
    packed = torch.zeros(n, n_bytes, dtype=torch.uint8, device=indices.device)

    bit_pos = 0
    for j in range(d):
        byte_idx = bit_pos // 8
        bit_offset = bit_pos % 8
        val = indices[:, j].to(torch.int32)

        # May span two bytes
        packed[:, byte_idx] |= ((val << bit_offset) & 0xFF).to(torch.uint8)
        overflow = bit_offset + bits - 8
        if overflow > 0:
            packed[:, byte_idx + 1] |= ((val >> (bits - overflow)) & 0xFF).to(torch.uint8)
        bit_pos += bits

    return packed


def unpack_indices(packed: torch.Tensor, bits: int, d: int) -> torch.Tensor:
    """Unpack a compact byte tensor back to integer indices.

    Parameters
    ----------
    packed : Tensor of uint8, shape (n, n_bytes).
    bits : int
        Bit-width per index.
    d : int
        Original number of indices per vector.

    Returns
    -------
    indices : Tensor of uint8, shape (n, d).
    """
    if bits == 8:
        return packed

    # Fast vectorized paths matching pack_indices().
    if bits in (1, 2, 4):
        per_byte = 8 // bits
        mask = (1 << bits) - 1
        n = packed.shape[0]
        shifts = (torch.arange(per_byte, device=packed.device, dtype=torch.uint8) * bits)
        # (n, n_bytes, 1) >> (per_byte,) → (n, n_bytes, per_byte)
        unpacked = ((packed.unsqueeze(-1) >> shifts) & mask).to(torch.uint8)
        return unpacked.view(n, -1)[:, :d]

    mask = (1 << bits) - 1
    n = packed.shape[0]
    indices = torch.zeros(n, d, dtype=torch.uint8, device=packed.device)

    bit_pos = 0
    for j in range(d):
        byte_idx = bit_pos // 8
        bit_offset = bit_pos % 8
        val = packed[:, byte_idx].to(torch.int32) >> bit_offset
        overflow = bit_offset + bits - 8
        if overflow > 0:
            val |= packed[:, byte_idx + 1].to(torch.int32) << (bits - overflow)
        indices[:, j] = (val & mask).to(torch.uint8)
        bit_pos += bits

    return indices


def packed_bytes_per_vector(d: int, bits: int) -> int:
    """Number of bytes per vector after packing."""
    return (d * bits + 7) // 8
