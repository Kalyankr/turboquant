"""Benchmark tqtorch vs FAISS across multiple index types.

Usage:
    uv run --extra bench python benchmarks/compare_faiss.py
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

# ── helpers ───────────────────────────────────────────────────────────────────


def recall_at_k(true_ids: np.ndarray, pred_ids: np.ndarray, k: int) -> float:
    """Fraction of queries whose true top-1 neighbor appears in top-k results."""
    hits = 0
    for i in range(true_ids.shape[0]):
        if true_ids[i, 0] in pred_ids[i, :k]:
            hits += 1
    return hits / true_ids.shape[0]


def faiss_ground_truth(db: np.ndarray, queries: np.ndarray, k: int):
    """Brute-force inner-product ground truth via FAISS Flat."""
    import faiss

    d = db.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(db)
    faiss.normalize_L2(queries)
    index.add(db)
    scores, ids = index.search(queries, k)
    return scores, ids


# ── benchmark runners ─────────────────────────────────────────────────────────


def bench_tqtorch(db_np: np.ndarray, q_np: np.ndarray, k: int, bits: int):
    from tqtorch.search.index import TurboQuantIndex

    d = db_np.shape[1]
    db_t = torch.from_numpy(db_np).float()
    q_t = torch.from_numpy(q_np).float()

    t0 = time.perf_counter()
    idx = TurboQuantIndex(dim=d, bits=bits, metric="ip", seed=42)
    idx.add(db_t)
    add_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    scores, ids = idx.search(q_t, k=k)
    search_ms = (time.perf_counter() - t0) * 1000

    return ids.numpy(), add_ms, search_ms, idx.memory_usage_mb


def bench_faiss_pq(db: np.ndarray, queries: np.ndarray, k: int, m: int):
    import faiss

    d = db.shape[1]
    index = faiss.IndexPQ(d, m, 8, faiss.METRIC_INNER_PRODUCT)
    faiss.normalize_L2(db)
    faiss.normalize_L2(queries)

    t0 = time.perf_counter()
    index.train(db)
    index.add(db)
    add_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    scores, ids = index.search(queries, k)
    search_ms = (time.perf_counter() - t0) * 1000

    mem_mb = (db.shape[0] * m) / 1e6
    return ids, add_ms, search_ms, mem_mb


def bench_faiss_ivfpq(
    db: np.ndarray, queries: np.ndarray, k: int, m: int, nprobe: int | None = None
):
    import faiss

    d = db.shape[1]
    n = db.shape[0]
    nlist = max(4, int(np.sqrt(n)))
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
    faiss.normalize_L2(db)
    faiss.normalize_L2(queries)

    t0 = time.perf_counter()
    index.train(db)
    index.add(db)
    add_ms = (time.perf_counter() - t0) * 1000

    # Use provided nprobe or default to 10% of clusters (good recall baseline)
    index.nprobe = nprobe if nprobe is not None else max(1, nlist // 10)
    t0 = time.perf_counter()
    scores, ids = index.search(queries, k)
    search_ms = (time.perf_counter() - t0) * 1000

    mem_mb = (n * m + nlist * d * 4) / 1e6
    return ids, add_ms, search_ms, mem_mb


def bench_faiss_opq(db: np.ndarray, queries: np.ndarray, k: int, m: int):
    """OPQ: Optimized Product Quantization (learned rotation + PQ)."""
    import faiss

    d = db.shape[1]
    opq = faiss.OPQMatrix(d, m)
    pq_index = faiss.IndexPQ(d, m, 8, faiss.METRIC_INNER_PRODUCT)
    index = faiss.IndexPreTransform(opq, pq_index)
    faiss.normalize_L2(db)
    faiss.normalize_L2(queries)

    t0 = time.perf_counter()
    index.train(db)
    index.add(db)
    add_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    scores, ids = index.search(queries, k)
    search_ms = (time.perf_counter() - t0) * 1000

    mem_mb = (db.shape[0] * m) / 1e6
    return ids, add_ms, search_ms, mem_mb


def bench_faiss_sq(db: np.ndarray, queries: np.ndarray, k: int, sq_bits: int):
    import faiss

    d = db.shape[1]
    sq_type = {4: faiss.ScalarQuantizer.QT_4bit, 8: faiss.ScalarQuantizer.QT_8bit}[
        sq_bits
    ]
    index = faiss.IndexScalarQuantizer(d, sq_type, faiss.METRIC_INNER_PRODUCT)
    faiss.normalize_L2(db)
    faiss.normalize_L2(queries)

    t0 = time.perf_counter()
    index.train(db)
    index.add(db)
    add_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    scores, ids = index.search(queries, k)
    search_ms = (time.perf_counter() - t0) * 1000

    bytes_per = {4: d // 2, 8: d}[sq_bits]
    mem_mb = (db.shape[0] * bytes_per) / 1e6
    return ids, add_ms, search_ms, mem_mb


# ── main ──────────────────────────────────────────────────────────────────────


def run_benchmark(
    n_db: int = 10_000,
    n_queries: int = 200,
    d: int = 128,
    k: int = 10,
    seed: int = 42,
):
    print(f"\n{'=' * 78}")
    print(f"  tqtorch vs FAISS Benchmark")
    print(f"  n_db={n_db:,}  n_queries={n_queries}  dim={d}  k={k}")
    print(f"{'=' * 78}\n")

    np.random.seed(seed)
    # NOTE: synthetic Gaussian data — best-case for rotation-based quantizers.
    # See multi_dataset_benchmark.ipynb for real-data (GloVe, 20 Newsgroups) results.
    db = np.random.randn(n_db, d).astype(np.float32)
    queries = np.random.randn(n_queries, d).astype(np.float32)

    # Ground truth
    print("Computing ground truth (FAISS Flat IP)...")
    db_gt, q_gt = db.copy(), queries.copy()
    _, gt_ids = faiss_ground_truth(db_gt, q_gt, k)

    results = []

    # tqtorch at various bit widths
    for bits in [2, 3, 4, 5]:
        db_c, q_c = db.copy(), queries.copy()
        # Normalize like FAISS does
        db_c /= np.linalg.norm(db_c, axis=1, keepdims=True) + 1e-12
        q_c /= np.linalg.norm(q_c, axis=1, keepdims=True) + 1e-12
        ids, add_ms, search_ms, mem_mb = bench_tqtorch(db_c, q_c, k, bits)
        r1 = recall_at_k(gt_ids, ids, 1)
        r10 = recall_at_k(gt_ids, ids, k)
        results.append((f"TurboQuant-{bits}bit", add_ms, search_ms, mem_mb, r1, r10))

    # FAISS PQ
    for m in [8, 16, 32]:
        if d % m != 0:
            continue
        db_c, q_c = db.copy(), queries.copy()
        ids, add_ms, search_ms, mem_mb = bench_faiss_pq(db_c, q_c, k, m)
        r1 = recall_at_k(gt_ids, ids, 1)
        r10 = recall_at_k(gt_ids, ids, k)
        results.append((f"FAISS-PQ(m={m})", add_ms, search_ms, mem_mb, r1, r10))

    # FAISS IVF+PQ (sweep nprobe for fair comparison)
    nlist = max(4, int(np.sqrt(n_db)))
    for m in [8, 16]:
        if d % m != 0:
            continue
        for nprobe in [nlist // 10, nlist // 4, nlist]:
            nprobe = max(1, nprobe)
            db_c, q_c = db.copy(), queries.copy()
            ids, add_ms, search_ms, mem_mb = bench_faiss_ivfpq(
                db_c, q_c, k, m, nprobe=nprobe
            )
            r1 = recall_at_k(gt_ids, ids, 1)
            r10 = recall_at_k(gt_ids, ids, k)
            results.append(
                (f"FAISS-IVF+PQ(m={m},np={nprobe})", add_ms, search_ms, mem_mb, r1, r10)
            )

    # FAISS OPQ (Optimized PQ — learned rotation + PQ)
    for m in [8, 16, 32]:
        if d % m != 0:
            continue
        db_c, q_c = db.copy(), queries.copy()
        ids, add_ms, search_ms, mem_mb = bench_faiss_opq(db_c, q_c, k, m)
        r1 = recall_at_k(gt_ids, ids, 1)
        r10 = recall_at_k(gt_ids, ids, k)
        results.append((f"FAISS-OPQ(m={m})", add_ms, search_ms, mem_mb, r1, r10))

    # FAISS SQ
    for sq_bits in [4, 8]:
        db_c, q_c = db.copy(), queries.copy()
        ids, add_ms, search_ms, mem_mb = bench_faiss_sq(db_c, q_c, k, sq_bits)
        r1 = recall_at_k(gt_ids, ids, 1)
        r10 = recall_at_k(gt_ids, ids, k)
        results.append((f"FAISS-SQ({sq_bits}bit)", add_ms, search_ms, mem_mb, r1, r10))

    # Print table
    header = f"{'Method':<25} {'Add(ms)':>9} {'Search(ms)':>11} {'Mem(MB)':>9} {'R@1':>7} {'R@10':>7}"
    print(header)
    print("-" * len(header))
    for name, add_ms, search_ms, mem_mb, r1, r10 in results:
        print(
            f"{name:<25} {add_ms:>9.1f} {search_ms:>11.1f} {mem_mb:>9.2f} {r1:>7.3f} {r10:>7.3f}"
        )

    return results


if __name__ == "__main__":
    # Allow quick/full mode from CLI
    if "--large" in sys.argv:
        run_benchmark(n_db=100_000, n_queries=500, d=256, k=10)
    else:
        run_benchmark(n_db=10_000, n_queries=200, d=128, k=10)
