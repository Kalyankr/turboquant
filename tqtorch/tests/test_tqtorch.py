"""Tests for tqtorch."""

import math
import tempfile
from pathlib import Path

import pytest
import torch

from tqtorch.core.rotation import (
    random_rotation_matrix,
    random_rotate,
    random_rotate_inverse,
)
from tqtorch.core.codebook import get_codebook
from tqtorch.core.packed import (
    pack_indices,
    unpack_indices,
    packed_bytes_per_vector,
)
from tqtorch.core.mse_quantizer import MSEQuantizer, mse_quantize, mse_dequantize
from tqtorch.core.prod_quantizer import (
    InnerProductQuantizer,
    ip_quantize,
    estimate_inner_product,
)
from tqtorch.core.qjl import qjl_projection_matrix, qjl_sign
from tqtorch.search.index import TurboQuantIndex


# ── Rotation ──────────────────────────────────────────────────────────────────


class TestRotation:
    def test_orthogonality(self):
        Pi = random_rotation_matrix(64, seed=0)
        eye = Pi @ Pi.T
        assert torch.allclose(eye, torch.eye(64), atol=1e-5)

    def test_determinism(self):
        a = random_rotation_matrix(32, seed=7)
        b = random_rotation_matrix(32, seed=7)
        assert torch.equal(a, b)

    def test_roundtrip(self):
        Pi = random_rotation_matrix(16, seed=1)
        x = torch.randn(5, 16)
        y = random_rotate(x, Pi)
        x_back = random_rotate_inverse(y, Pi)
        assert torch.allclose(x, x_back, atol=1e-5)

    def test_norm_preservation(self):
        Pi = random_rotation_matrix(64, seed=3)
        x = torch.randn(10, 64)
        y = random_rotate(x, Pi)
        assert torch.allclose(x.norm(dim=1), y.norm(dim=1), atol=1e-5)


# ── Codebook ──────────────────────────────────────────────────────────────────


class TestCodebook:
    @pytest.mark.parametrize("bits", [1, 2, 3, 4, 8])
    def test_shapes(self, bits):
        centroids, boundaries = get_codebook(64, bits)
        assert centroids.shape == (2**bits,)
        assert boundaries.shape == (2**bits - 1,)

    def test_sorted_centroids(self):
        centroids, _ = get_codebook(128, 4)
        diffs = centroids[1:] - centroids[:-1]
        assert (diffs > 0).all()

    def test_caching(self):
        a = get_codebook(64, 3)
        b = get_codebook(64, 3)
        assert a[0] is b[0]  # Same object from cache


# ── Bit Packing ───────────────────────────────────────────────────────────────


class TestPacking:
    @pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 8])
    def test_roundtrip(self, bits):
        d = 32
        indices = torch.randint(0, 2**bits, (10, d), dtype=torch.uint8)
        packed = pack_indices(indices, bits)
        unpacked = unpack_indices(packed, bits, d)
        assert torch.equal(indices, unpacked)

    def test_compression(self):
        d = 128
        # 2-bit packing: 128 * 2 / 8 = 32 bytes
        assert packed_bytes_per_vector(d, 2) == 32
        # 4-bit: 128 * 4 / 8 = 64 bytes
        assert packed_bytes_per_vector(d, 4) == 64


# ── MSE Quantizer ─────────────────────────────────────────────────────────────


class TestMSEQuantizer:
    def test_shapes(self):
        q = MSEQuantizer(64, bits=3, seed=0)
        x = torch.randn(100, 64)
        qt = q.quantize(x)
        assert qt.packed_indices.shape[0] == 100
        assert qt.norms.shape == (100,)
        x_hat = q.dequantize(qt)
        assert x_hat.shape == (100, 64)

    def test_mse_decreases_with_bits(self):
        torch.manual_seed(0)
        x = torch.randn(200, 32)
        mses = []
        for bits in [1, 2, 3, 4]:
            q = MSEQuantizer(32, bits, seed=42)
            x_hat = q.dequantize(q.quantize(x))
            mse = ((x - x_hat) ** 2).mean().item()
            mses.append(mse)
        # MSE should strictly decrease
        for i in range(len(mses) - 1):
            assert mses[i] > mses[i + 1], f"MSE did not decrease: bits={i + 1}→{i + 2}"

    def test_functional_api(self):
        x = torch.randn(50, 16)
        qt = mse_quantize(x, bits=4, seed=0)
        x_hat = mse_dequantize(qt)
        assert x_hat.shape == x.shape
        mse = ((x - x_hat) ** 2).mean().item()
        assert mse < x.var().item()  # Must be better than zero reconstruction


# ── IP Quantizer ──────────────────────────────────────────────────────────────


class TestIPQuantizer:
    def test_shapes(self):
        q = InnerProductQuantizer(64, bits=4, seed=0)
        x = torch.randn(50, 64)
        qt = q.quantize(x)
        x_hat = q.dequantize(qt)
        assert x_hat.shape == (50, 64)

    def test_ip_unbiased(self):
        """Inner product estimates should be approximately unbiased."""
        torch.manual_seed(42)
        d = 64
        n = 1000
        x = torch.randn(n, d) / math.sqrt(d)
        y = torch.randn(n, d) / math.sqrt(d)

        q = InnerProductQuantizer(d, bits=4, seed=0)
        qt = q.quantize(x)

        true_ips = (x * y).sum(dim=1)
        # Use the proper QJL-based IP estimator
        est_matrix = q.estimate_inner_products(qt, y)  # (n, n)
        est_ips = est_matrix.diagonal()  # paired IPs

        # The means should be close (unbiased => E[estimate] ~ E[true])
        assert abs(true_ips.mean().item() - est_ips.mean().item()) < 0.1

    def test_dequantize_is_mse_only(self):
        """dequantize() should return MSE reconstruction without QJL noise."""
        torch.manual_seed(0)
        d = 64
        x = torch.randn(50, d)
        q_ip = InnerProductQuantizer(d, bits=4, seed=0)
        qt = q_ip.quantize(x)
        x_hat = q_ip.dequantize(qt)

        # MSE of IP dequantize should equal MSE of a standalone (bits-1) MSE quantizer
        q_mse = MSEQuantizer(d, bits=3, seed=0)
        qt_mse = q_mse.quantize(x)
        x_hat_mse = q_mse.dequantize(qt_mse)

        mse_ip = ((x - x_hat) ** 2).mean().item()
        mse_pure = ((x - x_hat_mse) ** 2).mean().item()
        assert abs(mse_ip - mse_pure) < 1e-5, (
            f"Expected equal MSE: {mse_ip} vs {mse_pure}"
        )

    def test_functional_api(self):
        x = torch.randn(30, 32)
        qt = ip_quantize(x, bits=3, seed=0)
        y = torch.randn(30, 32)
        est = estimate_inner_product(qt, y)
        assert est.shape == (30,)


# ── QJL ───────────────────────────────────────────────────────────────────────


class TestQJL:
    def test_sign_output(self):
        S = qjl_projection_matrix(32, seed=0)
        r = torch.randn(10, 32)
        signs = qjl_sign(r, S)
        assert signs.dtype == torch.int8
        assert set(signs.unique().tolist()).issubset({-1, 0, 1})


# ── Search Index ──────────────────────────────────────────────────────────────


class TestTurboQuantIndex:
    def test_add_search(self):
        d, n, nq = 32, 200, 5
        torch.manual_seed(0)
        db = torch.randn(n, d)
        queries = torch.randn(nq, d)

        idx = TurboQuantIndex(dim=d, bits=4, metric="ip", seed=0)
        idx.add(db)
        assert idx.ntotal == n

        scores, indices = idx.search(queries, k=10)
        assert scores.shape == (nq, 10)
        assert indices.shape == (nq, 10)
        # Scores should be sorted descending
        assert (scores[:, :-1] >= scores[:, 1:]).all()

    def test_recall(self):
        """Top-1 recall should be non-trivial."""
        d, n = 32, 500
        torch.manual_seed(42)
        db = torch.randn(n, d)
        queries = db[:20]  # Query vectors from the DB itself

        idx = TurboQuantIndex(dim=d, bits=4, metric="ip", seed=0)
        idx.add(db)
        _, retrieved = idx.search(queries, k=1)

        # At least some should match exactly
        exact = (retrieved[:, 0] == torch.arange(20)).float().mean().item()
        assert exact > 0.3, f"Top-1 recall too low: {exact:.2f}"

    def test_save_load(self):
        d = 16
        db = torch.randn(50, d)
        idx = TurboQuantIndex(dim=d, bits=3, metric="ip", seed=7)
        idx.add(db)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.pt"
            idx.save(path)
            loaded = TurboQuantIndex.load(path)

        assert loaded.ntotal == 50
        assert loaded.dim == d
        assert loaded.bits == 3

        q = torch.randn(3, d)
        s1, i1 = idx.search(q, k=5)
        s2, i2 = loaded.search(q, k=5)
        assert torch.equal(i1, i2)
        assert torch.allclose(s1, s2, atol=1e-4)

    def test_mse_metric(self):
        """MSE metric path should also work."""
        d = 16
        db = torch.randn(30, d)
        idx = TurboQuantIndex(dim=d, bits=3, metric="mse", seed=0)
        idx.add(db)
        scores, indices = idx.search(torch.randn(2, d), k=5)
        assert scores.shape == (2, 5)

    def test_memory_property(self):
        d = 32
        idx = TurboQuantIndex(dim=d, bits=4, metric="ip")
        idx.add(torch.randn(100, d))
        assert idx.memory_usage_mb > 0

    def test_incremental_add(self):
        d = 16
        idx = TurboQuantIndex(dim=d, bits=3, metric="ip")
        idx.add(torch.randn(20, d))
        idx.add(torch.randn(30, d))
        assert idx.ntotal == 50
        scores, _ = idx.search(torch.randn(2, d), k=10)
        assert scores.shape == (2, 10)
