"""TurboQuantIndex — FAISS-like add/search API with bounded search-time memory."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from tqtorch.core.mse_quantizer import MSEQuantizer, QuantizedMSE
from tqtorch.core.prod_quantizer import InnerProductQuantizer, QuantizedIP
from tqtorch.core.packed import unpack_indices


class TurboQuantIndex:
    """Flat brute-force index backed by TurboQuant quantization.

    Supports add/search API similar to FAISS, with:
    - Zero indexing time (no training, no codebook learning)
    - Bounded search-time memory via configurable batch reconstruction
    - Save/load persistence

    Parameters
    ----------
    dim : int
        Vector dimensionality.
    bits : int
        Bits per coordinate.
    metric : str
        ``"ip"`` (inner product) or ``"mse"`` (MSE reconstruction).
    seed : int
        Random seed.
    search_batch_size : int
        Max vectors reconstructed at once during search. Controls peak memory:
        peak_fp32_bytes = search_batch_size * dim * 4.
    device : torch.device or str or None
        Target device.
    """

    def __init__(
        self,
        dim: int,
        bits: int = 4,
        metric: str = "ip",
        seed: int = 42,
        search_batch_size: int = 65_536,
        device: torch.device | str | None = None,
        compute_dtype: torch.dtype = torch.float32,
    ):
        self.dim = dim
        self.bits = bits
        self.metric = metric
        self.seed = seed
        self.search_batch_size = search_batch_size
        self.compute_dtype = compute_dtype
        self.device = (
            torch.device(device)
            if isinstance(device, str)
            else (device or torch.device("cpu"))
        )

        if metric == "ip" and bits >= 2:
            self._quant = InnerProductQuantizer(
                dim, bits, seed=seed, device=self.device
            )
        else:
            self._quant = MSEQuantizer(dim, bits, seed=seed, device=self.device)

        # Storage buffers — grow as vectors are added
        self._mse_packed_chunks: list[torch.Tensor] = []
        self._norms_chunks: list[torch.Tensor] = []
        self._qjl_packed_chunks: list[torch.Tensor] = []
        self._gammas_chunks: list[torch.Tensor] = []
        self._ntotal = 0
        self.last_add_time_ms: float = 0.0

    @property
    def ntotal(self) -> int:
        return self._ntotal

    def add(self, x: torch.Tensor):
        """Add vectors to the index.

        Parameters
        ----------
        x : Tensor of shape (n, d).
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {x.shape[1]}")

        t0 = time.perf_counter()
        x = x.to(self.device, dtype=torch.float32)

        if isinstance(self._quant, InnerProductQuantizer):
            qt = self._quant.quantize(x)
            self._mse_packed_chunks.append(qt.mse_packed)
            self._norms_chunks.append(qt.mse_norms)
            self._qjl_packed_chunks.append(qt.qjl_packed)
            self._gammas_chunks.append(qt.gammas)
        else:
            qt = self._quant.quantize(x)
            self._mse_packed_chunks.append(qt.packed_indices)
            self._norms_chunks.append(qt.norms)

        self._ntotal += x.shape[0]
        self.last_add_time_ms = (time.perf_counter() - t0) * 1000

    def _consolidate(self) -> None:
        """Merge per-add chunk lists into single tensors. Idempotent.

        Called lazily at search time so that repeated ``add()`` calls remain
        O(batch) instead of O(total) per call. After consolidation each
        chunk list has at most one entry.
        """
        if len(self._mse_packed_chunks) > 1:
            self._mse_packed_chunks = [torch.cat(self._mse_packed_chunks, dim=0)]
        if len(self._norms_chunks) > 1:
            self._norms_chunks = [torch.cat(self._norms_chunks, dim=0)]
        if len(self._qjl_packed_chunks) > 1:
            self._qjl_packed_chunks = [torch.cat(self._qjl_packed_chunks, dim=0)]
        if len(self._gammas_chunks) > 1:
            self._gammas_chunks = [torch.cat(self._gammas_chunks, dim=0)]

    def _reconstruct_batch(self, start: int, end: int) -> torch.Tensor:
        """Reconstruct vectors [start, end) from packed storage (MSE-only).

        Assumes ``_consolidate()`` has been called.
        """
        mse_packed = self._mse_packed_chunks[0][start:end]
        norms = self._norms_chunks[0][start:end]

        if isinstance(self._quant, InnerProductQuantizer):
            qt = QuantizedIP(
                mse_packed=mse_packed,
                mse_norms=norms,
                qjl_packed=torch.empty(0),  # not needed for MSE-only dequantize
                gammas=torch.empty(0),
                dim=self.dim,
                bits=self.bits,
                seed=self.seed,
            )
            return self._quant.dequantize(qt)
        else:
            qt = QuantizedMSE(
                packed_indices=mse_packed,
                norms=norms,
                dim=self.dim,
                bits=self.bits,
                seed=self.seed,
            )
            return self._quant.dequantize(qt)

    def _estimate_ip_batch(
        self, start: int, end: int, queries: torch.Tensor, Sy: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate inner products for vectors [start, end) using MSE + QJL.

        Assumes ``_consolidate()`` has been called and that ``Sy`` is the
        precomputed projection ``queries @ self._quant.S.T``.

        Returns scores of shape (nq, end-start).
        """
        mse_packed = self._mse_packed_chunks[0][start:end]
        norms = self._norms_chunks[0][start:end]
        qjl_packed = self._qjl_packed_chunks[0][start:end]
        gammas = self._gammas_chunks[0][start:end]
        qt = QuantizedIP(
            mse_packed=mse_packed,
            mse_norms=norms,
            qjl_packed=qjl_packed,
            gammas=gammas,
            dim=self.dim,
            bits=self.bits,
            seed=self.seed,
        )
        return self._quant.estimate_inner_products(qt, queries, Sy=Sy)

    def reconstruct(self, ids: torch.Tensor | list[int] | int) -> torch.Tensor:
        """Reconstruct individual vectors by id (MSE-only).

        Parameters
        ----------
        ids : int | list[int] | LongTensor of shape (k,)

        Returns
        -------
        x_hat : Tensor of shape (k, dim)  (or (dim,) for a scalar id)
        """
        if self._ntotal == 0:
            raise RuntimeError("Index is empty")
        single = isinstance(ids, int)
        if single:
            ids_t = torch.tensor([ids], dtype=torch.long)
        elif isinstance(ids, torch.Tensor):
            ids_t = ids.to(torch.long).reshape(-1)
        else:
            ids_t = torch.tensor(list(ids), dtype=torch.long)
        if (ids_t < 0).any() or (ids_t >= self._ntotal).any():
            raise IndexError(
                f"ids out of range [0, {self._ntotal}); got min={ids_t.min().item()},"
                f" max={ids_t.max().item()}"
            )
        self._consolidate()
        ids_t = ids_t.to(self.device)

        mse_packed = self._mse_packed_chunks[0].index_select(0, ids_t)
        norms = self._norms_chunks[0].index_select(0, ids_t)

        if isinstance(self._quant, InnerProductQuantizer):
            qt = QuantizedIP(
                mse_packed=mse_packed,
                mse_norms=norms,
                qjl_packed=torch.empty(0),
                gammas=torch.empty(0),
                dim=self.dim,
                bits=self.bits,
                seed=self.seed,
            )
            x_hat = self._quant.dequantize(qt)
        else:
            qt = QuantizedMSE(
                packed_indices=mse_packed,
                norms=norms,
                dim=self.dim,
                bits=self.bits,
                seed=self.seed,
            )
            x_hat = self._quant.dequantize(qt)
        return x_hat[0] if single else x_hat

    def remove(self, ids: torch.Tensor | list[int]) -> None:
        """Remove vectors at the given ids.

        WARNING: this rebuilds storage and renumbers the remaining vectors;
        any cached ids from prior ``search()`` calls become invalid.
        """
        if self._ntotal == 0:
            return
        if isinstance(ids, torch.Tensor):
            ids_t = ids.to(torch.long).reshape(-1)
        else:
            ids_t = torch.tensor(list(ids), dtype=torch.long)
        if ids_t.numel() == 0:
            return
        if (ids_t < 0).any() or (ids_t >= self._ntotal).any():
            raise IndexError("remove: ids out of range")

        self._consolidate()
        keep = torch.ones(self._ntotal, dtype=torch.bool)
        keep[ids_t] = False
        keep_dev = keep.to(self.device)

        self._mse_packed_chunks = [self._mse_packed_chunks[0][keep_dev]]
        self._norms_chunks = [self._norms_chunks[0][keep_dev]]
        if self._qjl_packed_chunks:
            self._qjl_packed_chunks = [self._qjl_packed_chunks[0][keep_dev]]
            self._gammas_chunks = [self._gammas_chunks[0][keep_dev]]
        self._ntotal = int(keep.sum().item())

    def search(
        self, queries: torch.Tensor, k: int = 10
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Search for k nearest neighbors by inner product.

        Uses bounded-memory batch reconstruction: at most
        ``search_batch_size * dim * 4`` bytes of fp32 scratch at any time.

        Parameters
        ----------
        queries : Tensor of shape (nq, d).
        k : int

        Returns
        -------
        scores : Tensor of shape (nq, k), descending.
        indices : Tensor of shape (nq, k), int64.
        """
        if self._ntotal == 0:
            raise RuntimeError("Index is empty; call add() first")
        if queries.dim() == 1:
            queries = queries.unsqueeze(0)
        queries = queries.to(self.device, dtype=torch.float32)
        nq = queries.shape[0]
        k = min(k, self._ntotal)

        # Consolidate per-add chunks once so batch slicing is O(batch).
        self._consolidate()

        # Accumulate top-k across batches
        top_scores = torch.full((nq, k), -float("inf"), device=self.device)
        top_ids = torch.zeros(nq, k, dtype=torch.int64, device=self.device)

        use_ip_estimator = isinstance(self._quant, InnerProductQuantizer)

        # Hoist the QJL projection of queries out of the per-batch loop.
        Sy = queries @ self._quant.S.T if use_ip_estimator else None

        # Cast queries for compute if a reduced precision was requested.
        q_compute = queries.to(self.compute_dtype)

        offset = 0
        while offset < self._ntotal:
            end = min(offset + self.search_batch_size, self._ntotal)

            if use_ip_estimator:
                batch_scores = self._estimate_ip_batch(offset, end, queries, Sy)
            else:
                batch_recon = self._reconstruct_batch(offset, end)  # (batch, d)
                batch_scores = q_compute @ batch_recon.to(self.compute_dtype).T
                batch_scores = batch_scores.to(torch.float32)

            # Merge with running top-k
            combined_scores = torch.cat([top_scores, batch_scores], dim=1)
            combined_ids = torch.cat(
                [
                    top_ids,
                    torch.arange(offset, end, device=self.device)
                    .unsqueeze(0)
                    .expand(nq, -1),
                ],
                dim=1,
            )
            topk_vals, topk_local = combined_scores.topk(
                k, dim=1, largest=True, sorted=True
            )
            top_scores = topk_vals
            top_ids = combined_ids.gather(1, topk_local)

            offset = end

        return top_scores, top_ids

    @property
    def memory_usage_mb(self) -> float:
        """Approximate memory used by quantized storage in MB."""
        total_bytes = 0
        for chunk in self._mse_packed_chunks:
            total_bytes += chunk.nelement() * chunk.element_size()
        for chunk in self._norms_chunks:
            total_bytes += chunk.nelement() * chunk.element_size()
        for chunk in self._qjl_packed_chunks:
            total_bytes += chunk.nelement() * chunk.element_size()
        for chunk in self._gammas_chunks:
            total_bytes += chunk.nelement() * chunk.element_size()
        return total_bytes / 1e6

    def save(self, path: str | Path):
        """Save the index to a .pt file."""
        path = Path(path)
        self._consolidate()
        state = {
            "dim": self.dim,
            "bits": self.bits,
            "metric": self.metric,
            "seed": self.seed,
            "search_batch_size": self.search_batch_size,
            "compute_dtype": str(self.compute_dtype),
            "ntotal": self._ntotal,
            "mse_packed": self._mse_packed_chunks[0]
            if self._mse_packed_chunks
            else torch.empty(0),
            "norms": self._norms_chunks[0]
            if self._norms_chunks
            else torch.empty(0),
        }
        if self._qjl_packed_chunks:
            state["qjl_packed"] = self._qjl_packed_chunks[0]
            state["gammas"] = self._gammas_chunks[0]
        torch.save(state, path)

    @classmethod
    def load(
        cls, path: str | Path, device: torch.device | str | None = None
    ) -> TurboQuantIndex:
        """Load an index from a .pt file."""
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        # Backward-compat: older saves don't have compute_dtype
        compute_dtype = torch.float32
        if "compute_dtype" in state:
            compute_dtype = getattr(torch, state["compute_dtype"].split(".")[-1], torch.float32)
        idx = cls(
            dim=state["dim"],
            bits=state["bits"],
            metric=state["metric"],
            seed=state["seed"],
            search_batch_size=state["search_batch_size"],
            device=device,
            compute_dtype=compute_dtype,
        )
        if state["ntotal"] > 0:
            mse_packed = state["mse_packed"].to(idx.device)
            norms = state["norms"].to(idx.device)
            idx._mse_packed_chunks = [mse_packed]
            idx._norms_chunks = [norms]
            if "qjl_packed" in state:
                idx._qjl_packed_chunks = [state["qjl_packed"].to(idx.device)]
                idx._gammas_chunks = [state["gammas"].to(idx.device)]
            idx._ntotal = state["ntotal"]
        return idx
