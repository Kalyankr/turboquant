<div align="center">

# TurboQuant

**Compress vectors to 4 bits with OPQ-level recall — zero training time.**

A PyTorch (`tqtorch`) and minimal NumPy (`turboquantlite`) implementation of Google's TurboQuant vector quantization algorithm
([Zandieh et al., 2025](https://arxiv.org/abs/2504.19874)).

</div>

---

## Table of Contents

- [Why TurboQuant?](#why-turboquant)
- [Benchmark Results](#benchmark-results)
- [Repository Structure](#repository-structure)
- [Install](#install)
- [Usage](#usage)
- [LangChain Integration](#langchain-integration)
- [Citation](#citation)
- [License](#license)

---

## Why TurboQuant?

Real embeddings (sentence-transformers, GloVe, BGE) concentrate information into a few principal components. Naïve scalar quantization loses too much accuracy. Product Quantization (PQ / OPQ) fixes this by learning a data-dependent codebook via k-means — a heavy step that must be repeated whenever the embedding distribution shifts.

**TurboQuant eliminates the training step entirely:**

1. **Random Rotation** — multiply by a seeded, data-independent orthogonal matrix $\Pi$ (via QR decomposition). Information is spread uniformly across all dimensions.
2. **Lloyd-Max Codebooks** — after rotation, coordinates are approximately i.i.d. Gaussian, so each one is scalar-quantized into $b$ bits using precomputed Lloyd-Max centroids for $\mathcal{N}(0, 1/d)$.
3. **QJL Correction** *(inner-product variant)* — allocate $(b{-}1)$ bits to MSE quantization and 1 bit to a Quantized Johnson-Lindenstrauss sign on the residual, yielding an **unbiased** inner-product estimator.

$$\hat{x} = \Pi^\top \, Q_b(\Pi \, x)$$

**Result:** OPQ-level retrieval accuracy with zero training cost and no dataset dependency.

---

## Benchmark Results

GloVe-100d, 10,000 database vectors, 200 queries, recall@10 vs exact inner-product ground truth.
Reproduce with [`benchmarks/real_embeddings.py`](./benchmarks/real_embeddings.py).

| Method                       |     Build |   Memory |  R@10 |
| :--------------------------- | --------: | -------: | ----: |
| **TurboQuant 4-bit (mse)**   |  **38 ms** | **0.52 MB** | **0.889** |
| **TurboQuant 6-bit (mse)**   | **123 ms** | **0.77 MB** | **0.973** |
| FAISS-PQ&nbsp;&nbsp;(m=50)   |  1,187 ms |  0.50 MB | 0.913 |
| FAISS-OPQ (m=50)             | 20,518 ms |  0.50 MB | 0.922 |
| FAISS-PQ&nbsp;&nbsp;(m=100)  |  3,288 ms |  1.00 MB | 0.976 |
| FAISS-SQ8                    |      8 ms |  1.00 MB | 0.990 |
| FAISS-FlatIP (exact)         |      4 ms |  4.00 MB | 1.000 |

> TurboQuant matches OPQ-level recall at the **same memory footprint** while building **~150–500× faster** — no codebook training, no dataset dependency.

```bash
uv run --extra bench python benchmarks/real_embeddings.py
```

---

## Repository Structure

| Directory                              | Description                                                                                                  |
| :------------------------------------- | :----------------------------------------------------------------------------------------------------------- |
| [`tqtorch/`](./tqtorch/)               | Production PyTorch package — GPU-accelerated, 1-to-8-bit packing, save/load, fp16 search, LangChain support. |
| [`turboquantlite/`](./turboquantlite/) | Minimal NumPy reference implementation — readable, dependency-light.                                         |
| [`notebooks/`](./notebooks/)           | Interactive walkthroughs of the math and intuition.                                                          |
| [`benchmarks/`](./benchmarks/)         | Reproducible scripts vs FAISS PQ / OPQ / IVF / SQ on GloVe, MNIST, and 20 Newsgroups.                         |

---

## Install

```bash
uv add tqtorch
# or
pip install ./tqtorch
```

---

## Usage

### Build an index

```python
import torch
from tqtorch import TurboQuantIndex

# 4-bit, unbiased inner-product index — no training step.
index = TurboQuantIndex(dim=384, bits=4, metric="ip")

embeddings = torch.randn(10_000, 384)
index.add(embeddings)

queries = torch.randn(5, 384)
scores, ids = index.search(queries, k=10)
```

### Inspect, edit, and accelerate

```python
# Reconstruct stored vectors (MSE-only, lossy)
v   = index.reconstruct(7)             # single id   -> Tensor (dim,)
V   = index.reconstruct([0, 5, 99])    # batch ids   -> Tensor (k, dim)

# Remove vectors (renumbers remaining ids)
index.remove([0, 1, 2])

# fp16 search-time matmul for ~2× throughput on GPUs that support it
index = TurboQuantIndex(dim=384, bits=4, metric="ip",
                        compute_dtype=torch.float16)

# Persist / restore
index.save("my.index.pt")
index = TurboQuantIndex.load("my.index.pt")
```

---

## LangChain Integration

```python
from tqtorch.search.langchain import TurboQuantVectorStore
from langchain_openai import OpenAIEmbeddings

store = TurboQuantVectorStore.from_texts(
    texts=["Document 1 text", "Document 2 text"],
    embedding=OpenAIEmbeddings(),
    bits=4,
    metric="ip",
)

docs = store.similarity_search("Query string", k=3)
```

---

## Citation

```bibtex
@article{zandieh2025turboquant,
  title   = {TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author  = {Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  journal = {arXiv preprint arXiv:2504.19874},
  year    = {2025}
}
```

---

## License

MIT
