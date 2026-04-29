"""Real-embeddings benchmark for tqtorch.

Verifies the paper's "OPQ-level recall at zero training cost" claim on
*real* (concentrated) embeddings — the regime where TurboQuant should shine.

Datasets (auto-detected, in priority order):

1. **GloVe-100d** — `benchmarks/data/glove.6B.100d.txt`. Always preferred:
   it ships with the repo, no internet required, 400k vectors.
2. **sentence-transformers (all-MiniLM-L6-v2)** — only if the package is
   installed. Encodes a small built-in text corpus (the script's own docstrings
   plus a handful of canned sentences) — purely a smoke test for the
   ST → TurboQuantIndex pipeline.

Usage:

    uv run --extra bench python benchmarks/real_embeddings.py
    uv run --extra bench python benchmarks/real_embeddings.py --large
    uv run --extra bench python benchmarks/real_embeddings.py --st  # add ST smoke test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
GLOVE_PATH = REPO_ROOT / "benchmarks" / "data" / "glove.6B.100d.txt"


# ── helpers ───────────────────────────────────────────────────────────────────


def _load_glove(path: Path, max_vectors: int) -> np.ndarray:
    """Load up to `max_vectors` GloVe vectors, return float32 (n, d)."""
    if not path.exists():
        raise FileNotFoundError(
            f"GloVe file not found at {path}. "
            "Download from https://nlp.stanford.edu/data/glove.6B.zip and "
            f"extract glove.6B.100d.txt into {path.parent}."
        )
    vecs = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= max_vectors:
                break
            parts = line.rstrip().split(" ")
            vecs.append([float(x) for x in parts[1:]])
    return np.asarray(vecs, dtype=np.float32)


def _normalize(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def recall_at_k(true_top_k: np.ndarray, pred_top_k: np.ndarray, k: int) -> float:
    """Mean fraction of the true top-k neighbours present in the predicted top-k."""
    nq = true_top_k.shape[0]
    hits = 0.0
    for i in range(nq):
        hits += len(set(true_top_k[i, :k].tolist()) & set(pred_top_k[i, :k].tolist()))
    return hits / (nq * k)


# ── Index runners ─────────────────────────────────────────────────────────────


def _bench_tqtorch(db: np.ndarray, queries: np.ndarray, k: int, bits: int, metric: str):
    from tqtorch.search.index import TurboQuantIndex

    d = db.shape[1]
    db_t = torch.from_numpy(db)
    q_t = torch.from_numpy(queries)

    t0 = time.perf_counter()
    idx = TurboQuantIndex(dim=d, bits=bits, metric=metric, seed=42)
    idx.add(db_t)
    add_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    _, ids = idx.search(q_t, k=k)
    search_ms = (time.perf_counter() - t0) * 1000

    return ids.numpy(), add_ms, search_ms, idx.memory_usage_mb


def _bench_faiss(name: str, builder, db: np.ndarray, queries: np.ndarray, k: int):
    """Run an arbitrary FAISS index. ``builder(d)`` returns an *untrained* index.

    Returns (ids, add_ms, search_ms, mem_mb).
    """
    import faiss

    d = db.shape[1]
    index = builder(d)
    t0 = time.perf_counter()
    if not index.is_trained:
        index.train(db)
    index.add(db)
    add_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    _, ids = index.search(queries, k)
    search_ms = (time.perf_counter() - t0) * 1000

    # rough memory model
    if isinstance(index, faiss.IndexFlatIP):
        mem_mb = db.shape[0] * d * 4 / 1e6
    elif "SQ4" in name:
        mem_mb = db.shape[0] * d * 0.5 / 1e6
    elif "SQ8" in name:
        mem_mb = db.shape[0] * d / 1e6
    elif "PQ" in name or "OPQ" in name:
        # bytes per code = m (each subspace uses 8-bit codes by default)
        m = int(name.split("m=")[1].rstrip(")"))
        mem_mb = db.shape[0] * m / 1e6
    else:
        mem_mb = float("nan")
    return ids, add_ms, search_ms, mem_mb


# ── Main ──────────────────────────────────────────────────────────────────────


def run(n_db: int, n_queries: int, k: int = 10, st_smoke: bool = False) -> None:
    print(f"\n{'=' * 78}")
    print("  tqtorch real-embeddings benchmark")
    print(f"{'=' * 78}\n")

    # ---- Load GloVe ----------------------------------------------------------
    total = n_db + n_queries
    print(f"Loading {total:,} GloVe-100d vectors from {GLOVE_PATH.name}...")
    raw = _load_glove(GLOVE_PATH, max_vectors=total)
    raw = _normalize(raw)
    db, queries = raw[:n_db].copy(), raw[n_db : n_db + n_queries].copy()
    d = db.shape[1]
    print(f"  db={db.shape}  queries={queries.shape}  dim={d}\n")

    # ---- Ground truth (exact IP top-k) ---------------------------------------
    print("Computing exact ground truth (numpy IP top-k)...")
    t0 = time.perf_counter()
    gt_scores = queries @ db.T
    # Top-k by IP — argpartition for speed
    gt_ids = np.argpartition(-gt_scores, k, axis=1)[:, :k]
    rows = np.arange(n_queries)[:, None]
    gt_top_scores = gt_scores[rows, gt_ids]
    gt_order = np.argsort(-gt_top_scores, axis=1)
    gt_top_k = gt_ids[rows, gt_order]
    print(f"  done in {time.perf_counter() - t0:.2f}s\n")

    rows_table: list[tuple] = []

    # ---- TurboQuant ----------------------------------------------------------
    for bits in (2, 3, 4, 5, 6):
        for metric in ("ip", "mse"):
            if metric == "ip" and bits < 2:
                continue
            ids, add_ms, search_ms, mem_mb = _bench_tqtorch(db, queries, k, bits, metric)
            r1 = recall_at_k(gt_top_k, ids, 1)
            rk = recall_at_k(gt_top_k, ids, k)
            rows_table.append(
                (f"TurboQuant-{bits}b-{metric}", add_ms, search_ms, mem_mb, r1, rk)
            )

    # ---- FAISS baselines (optional) ------------------------------------------
    try:
        import faiss

        print("Running FAISS baselines...\n")
        # IndexFlatIP — exact baseline
        rows_table.append(
            (
                "FAISS-FlatIP",
                *_bench_faiss(
                    "FAISS-FlatIP", lambda d: faiss.IndexFlatIP(d), db, queries, k
                )[1:],
                1.0,
                1.0,
            )
        )

        for m in (4, 5, 10, 20, 25, 50, 100):
            if d % m != 0:
                continue
            for fam, ctor in [
                (
                    "PQ",
                    lambda d, m=m: faiss.IndexPQ(
                        d, m, 8, faiss.METRIC_INNER_PRODUCT
                    ),
                ),
                (
                    "OPQ",
                    lambda d, m=m: faiss.IndexPreTransform(
                        faiss.OPQMatrix(d, m),
                        faiss.IndexPQ(d, m, 8, faiss.METRIC_INNER_PRODUCT),
                    ),
                ),
            ]:
                name = f"FAISS-{fam}(m={m})"
                ids, add_ms, search_ms, mem_mb = _bench_faiss(name, ctor, db, queries, k)
                rows_table.append(
                    (
                        name,
                        add_ms,
                        search_ms,
                        mem_mb,
                        recall_at_k(gt_top_k, ids, 1),
                        recall_at_k(gt_top_k, ids, k),
                    )
                )

        for sq_bits, qt_enum, label in (
            (4, faiss.ScalarQuantizer.QT_4bit, "FAISS-SQ4"),
            (8, faiss.ScalarQuantizer.QT_8bit, "FAISS-SQ8"),
        ):
            ctor = lambda d, qt=qt_enum: faiss.IndexScalarQuantizer(
                d, qt, faiss.METRIC_INNER_PRODUCT
            )
            ids, add_ms, search_ms, mem_mb = _bench_faiss(label, ctor, db, queries, k)
            rows_table.append(
                (
                    label,
                    add_ms,
                    search_ms,
                    mem_mb,
                    recall_at_k(gt_top_k, ids, 1),
                    recall_at_k(gt_top_k, ids, k),
                )
            )
    except ImportError:
        print("(faiss-cpu not installed — skipping FAISS baselines)\n")

    # ---- Print table ---------------------------------------------------------
    header = f"{'Method':<24} {'Add(ms)':>9} {'Search(ms)':>11} {'Mem(MB)':>9} {'R@1':>7} {'R@10':>7}"
    print(header)
    print("-" * len(header))
    for name, add_ms, search_ms, mem_mb, r1, rk in rows_table:
        print(
            f"{name:<24} {add_ms:>9.1f} {search_ms:>11.1f} {mem_mb:>9.2f} {r1:>7.3f} {rk:>7.3f}"
        )

    # ---- Optional: sentence-transformers smoke test --------------------------
    if st_smoke:
        _run_st_smoke()


def _run_st_smoke() -> None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("\n(sentence-transformers not installed — skipping ST smoke test)")
        return
    from tqtorch.search.index import TurboQuantIndex

    print("\n--- sentence-transformers smoke test ---")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    corpus = [
        "TurboQuant is a vector quantization algorithm.",
        "Lloyd-Max codebooks are optimal for Gaussian sources.",
        "Random rotations spread information across coordinates.",
        "Product quantization learns a data-dependent codebook.",
        "GloVe vectors capture word co-occurrence statistics.",
        "BGE and OpenAI embeddings concentrate in a few directions.",
        "FAISS is a library for efficient similarity search.",
        "RAG pipelines depend on dense retrieval quality.",
    ]
    queries = [
        "quantization for embeddings",
        "approximate nearest neighbor search",
    ]
    emb = model.encode(corpus, normalize_embeddings=True, convert_to_numpy=True)
    q_emb = model.encode(queries, normalize_embeddings=True, convert_to_numpy=True)

    idx = TurboQuantIndex(dim=emb.shape[1], bits=4, metric="ip", seed=0)
    idx.add(torch.from_numpy(emb))
    scores, ids = idx.search(torch.from_numpy(q_emb), k=3)
    for q, qids, qscores in zip(queries, ids.tolist(), scores.tolist()):
        print(f"\nQ: {q}")
        for rank, (i, s) in enumerate(zip(qids, qscores), start=1):
            print(f"  {rank}. ({s:+.3f})  {corpus[i]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--large", action="store_true", help="50k db / 1k queries")
    parser.add_argument("--st", action="store_true", help="run ST smoke test too")
    args = parser.parse_args()
    if args.large:
        run(n_db=50_000, n_queries=1_000, st_smoke=args.st)
    else:
        run(n_db=10_000, n_queries=200, st_smoke=args.st)


if __name__ == "__main__":
    sys.exit(main() or 0)
