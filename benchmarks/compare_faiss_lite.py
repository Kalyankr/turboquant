"""Benchmark utilities — compare TurboQuantLite against FAISS."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from turboquantlite.index import TurboQuantIndex


@dataclass
class BenchmarkResult:
    method: str
    bits_per_vector: float
    memory_mb: float
    recall_at: dict[int, float] = field(default_factory=dict)
    build_time_s: float = 0.0
    query_time_s: float = 0.0
    n_vectors: int = 0
    d: int = 0


def _normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def _recall_at_k(true_ids: np.ndarray, pred_ids: np.ndarray, k: int) -> float:
    """Fraction of queries where the true #1 neighbor appears in the top-k predicted."""
    hits = 0
    for t, p in zip(true_ids, pred_ids):
        if t in p[:k]:
            hits += 1
    return hits / len(true_ids)


# ─── FAISS baselines ──────────────────────────────────────────────────────────

def _build_faiss_flat(d: int, db: np.ndarray):
    """Exact brute-force FAISS index (inner product)."""
    import faiss

    index = faiss.IndexFlatIP(d)
    index.add(db.astype(np.float32))
    return index


def _build_faiss_pq(d: int, db: np.ndarray, m: int, nbits: int = 8):
    """FAISS Product Quantization index."""
    import faiss

    index = faiss.IndexPQ(d, m, nbits, faiss.METRIC_INNER_PRODUCT)
    index.train(db.astype(np.float32))
    index.add(db.astype(np.float32))
    return index


def _build_faiss_ivfpq(d: int, db: np.ndarray, nlist: int, m: int, nbits: int = 8):
    """FAISS IVF + PQ index."""
    import faiss

    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
    index.train(db.astype(np.float32))
    index.add(db.astype(np.float32))
    index.nprobe = min(16, nlist)
    return index


def _build_faiss_sq(d: int, db: np.ndarray, qtype_str: str = "QT_8bit"):
    """FAISS Scalar Quantizer index."""
    import faiss

    qtype_map = {
        "QT_4bit": faiss.ScalarQuantizer.QT_4bit,
        "QT_8bit": faiss.ScalarQuantizer.QT_8bit,
    }
    qtype = qtype_map.get(qtype_str, faiss.ScalarQuantizer.QT_8bit)
    index = faiss.IndexScalarQuantizer(d, qtype, faiss.METRIC_INNER_PRODUCT)
    index.train(db.astype(np.float32))
    index.add(db.astype(np.float32))
    return index


def _faiss_index_memory(index, n: int, d: int, method: str) -> float:
    """Rough memory estimate in bytes for a FAISS index."""
    import faiss

    if isinstance(index, faiss.IndexFlatIP):
        return n * d * 4  # float32
    if isinstance(index, faiss.IndexPQ):
        return n * index.pq.code_size + index.pq.M * (2**index.pq.nbits) * (d // index.pq.M) * 4
    if isinstance(index, faiss.IndexIVFPQ):
        return n * index.pq.code_size + index.pq.M * (2**index.pq.nbits) * (d // index.pq.M) * 4
    if isinstance(index, faiss.IndexScalarQuantizer):
        bits = 8 if "8bit" in method else 4
        return n * d * bits // 8
    return n * d  # fallback


# ─── Main benchmark runner ────────────────────────────────────────────────────

def run_benchmark(
    n_db: int = 50_000,
    n_queries: int = 200,
    d: int = 128,
    ks: list[int] | None = None,
    turbo_bits: list[int] | None = None,
    seed: int = 42,
    include_faiss: bool = True,
) -> list[BenchmarkResult]:
    """Run a full benchmark comparing TurboQuantLite vs FAISS methods.

    Parameters
    ----------
    n_db : int
        Number of database vectors.
    n_queries : int
        Number of query vectors.
    d : int
        Dimensionality.
    ks : list[int]
        Recall@k values to compute (default: [1, 5, 10]).
    turbo_bits : list[int]
        Bit-widths to test for TurboQuant (default: [2, 3, 4]).
    seed : int
        Random seed.
    include_faiss : bool
        Whether to include FAISS baselines (requires ``faiss-cpu``).

    Returns
    -------
    list[BenchmarkResult]
    """
    if ks is None:
        ks = [1, 5, 10]
    if turbo_bits is None:
        turbo_bits = [2, 3, 4]

    rng = np.random.default_rng(seed)
    db = rng.standard_normal((n_db, d)).astype(np.float64)
    db = _normalize(db)
    queries = rng.standard_normal((n_queries, d)).astype(np.float64)
    queries = _normalize(queries)

    # Ground truth — exact inner product
    gt_scores = queries @ db.T
    gt_top1 = np.argmax(gt_scores, axis=1)
    gt_topk = np.argsort(-gt_scores, axis=1)  # full ranking

    results: list[BenchmarkResult] = []

    # ── TurboQuant variants ──────────────────────────────────────────────────
    for b in turbo_bits:
        for variant, label in [("mse", "TurboQuant_MSE"), ("prod", "TurboQuant_Prod")]:
            if variant == "prod" and b < 2:
                continue

            t0 = time.perf_counter()
            idx = TurboQuantIndex(d, b, variant=variant, seed=seed)
            idx.add(db)
            build_t = time.perf_counter() - t0

            t0 = time.perf_counter()
            _, pred_ids = idx.search(queries, k=max(ks))
            query_t = time.perf_counter() - t0

            recall = {}
            for k in ks:
                recall[k] = _recall_at_k(gt_top1, pred_ids, k)

            mem_bytes = idx.memory_bytes()
            bpv = idx.quant.memory_bits_per_vector()

            results.append(
                BenchmarkResult(
                    method=f"{label} {b}-bit",
                    bits_per_vector=bpv,
                    memory_mb=mem_bytes / 1e6,
                    recall_at=recall,
                    build_time_s=build_t,
                    query_time_s=query_t,
                    n_vectors=n_db,
                    d=d,
                )
            )

    # ── FAISS baselines ──────────────────────────────────────────────────────
    if include_faiss:
        try:
            import faiss  # noqa: F401

            db32 = db.astype(np.float32)
            q32 = queries.astype(np.float32)

            faiss_configs = [
                ("FAISS Flat (exact)", lambda: _build_faiss_flat(d, db)),
                ("FAISS PQ (m=8)", lambda: _build_faiss_pq(d, db, m=8)),
                ("FAISS PQ (m=16)", lambda: _build_faiss_pq(d, db, m=16)),
                ("FAISS PQ (m=32)", lambda: _build_faiss_pq(d, db, m=32)),
                ("FAISS IVF+PQ (m=16)", lambda: _build_faiss_ivfpq(d, db, nlist=max(4, int(np.sqrt(n_db))), m=16)),
                ("FAISS SQ 8-bit", lambda: _build_faiss_sq(d, db, "QT_8bit")),
                ("FAISS SQ 4-bit", lambda: _build_faiss_sq(d, db, "QT_4bit")),
            ]

            for name, builder in faiss_configs:
                t0 = time.perf_counter()
                findex = builder()
                build_t = time.perf_counter() - t0

                t0 = time.perf_counter()
                _, I = findex.search(q32, max(ks))
                query_t = time.perf_counter() - t0

                recall = {}
                for k in ks:
                    recall[k] = _recall_at_k(gt_top1, I, k)

                mem = _faiss_index_memory(findex, n_db, d, name)

                # Estimate bits per vector
                if "Flat" in name:
                    bpv = d * 32
                elif "SQ 4" in name:
                    bpv = d * 4
                elif "SQ 8" in name:
                    bpv = d * 8
                elif "PQ" in name:
                    bpv = findex.pq.code_size * 8 if hasattr(findex, "pq") else d * 8

                results.append(
                    BenchmarkResult(
                        method=name,
                        bits_per_vector=bpv,
                        memory_mb=mem / 1e6,
                        recall_at=recall,
                        build_time_s=build_t,
                        query_time_s=query_t,
                        n_vectors=n_db,
                        d=d,
                    )
                )

        except ImportError:
            print("faiss-cpu not installed — skipping FAISS baselines.")
            print("Install with:  pip install faiss-cpu")

    return results


def print_results(results: list[BenchmarkResult]):
    """Pretty-print benchmark results as a table."""
    ks = sorted(next((r.recall_at.keys() for r in results), [1, 5, 10]))
    recall_headers = [f"R@{k}" for k in ks]

    header = f"{'Method':<35} {'Bits/vec':>10} {'Mem (MB)':>10} " + " ".join(
        f"{h:>8}" for h in recall_headers
    ) + f" {'Build(s)':>10} {'Query(s)':>10}"
    print(header)
    print("-" * len(header))

    for r in results:
        recalls = " ".join(f"{r.recall_at.get(k, 0):>8.1%}" for k in ks)
        print(
            f"{r.method:<35} {r.bits_per_vector:>10.0f} {r.memory_mb:>10.2f} "
            f"{recalls} {r.build_time_s:>10.2f} {r.query_time_s:>10.4f}"
        )
