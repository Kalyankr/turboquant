# TurboQuant

A PyTorch (`tqtorch`) and minimal NumPy (`turboquantlite`) implementation of Google's 2025 vector quantization algorithms: [TurboQuant](https://arxiv.org/abs/2501.XXXXX).

> **Compress vectors to 4-bits with 99.4% Recall@10. Zero training time.**

This repo provides drop-in vector indexes and LangChain VectorStores, enabling **~8x memory compression** for RAG and dense retrieval pipelines without the cost, complexity, or dataset-dependency of k-means (FAISS PQ).

## Why TurboQuant?

Real embeddings (sentence-transformers, GloVe, BGE) cluster information into a few principal components. If you try to scalar-quantize them directly, you lose too much accuracy.FAISS Product Quantization (PQ/OPQ) fixes this, but it requires learning a data-dependent codebook via k-means — a heavy, expensive step whenever distributions shift.

**TurboQuant fixes this in 2 steps:**
1. **Random Rotation:** Multiply your vectors by a seeded, data-independent orthogonal matrix (via QR decomposition). This spreads information perfectly across all dimensions.
2. **Lloyd-Max Codebooks:** Because dimensions are independent and Gaussian, scalar quantize coordinates optimally into 4 bits (or fewer) using standard precomputed Gaussian codebooks. QJL bit-flipping preserves exact inner products.

**Result:** OPQ-level retrieval accuracy without the training cost.

### Benchmark vs FAISS

*Comparison on GloVe-100d (400,000 vectors) using flat exhaustive inner-product search (4-bit encoding).*

| Method | Index Memory | R@10 | Codebook Training Time |
|--------|--------------|------|------------------------|
| **TurboQuant (tqtorch)** | **21 MB** | **0.994** | **0.00s (None)** |
| FAISS SQ-4bit | ~21 MB | 0.970 | 0.05s (Range bounds) |
| FAISS OPQ-4bit | ~21 MB | 0.148 | 12.4s (k-means + PCA) |
| Float32 (Uncompressed) | 160 MB | 1.000 | - |

*(Run these yourself in `benchmarks/multi_dataset_benchmark.ipynb`)*

## Repository Structure

- [`tqtorch/`](./tqtorch/): The main production-ready Python package. **GPU accelerated** via PyTorch, features 1-to-8 bit packing, save/load, and LangChain integration.
- [`turboquantlite/`](./turboquantlite/): A lightweight NumPy/SciPy reference implementation.
- [`notebooks/`](./notebooks/): Interactive notebooks that walk through the math and intuition behind the algorithm. Start with `turboquant_small_dim_walkthrough.ipynb`.
- [`benchmarks/`](./benchmarks/): Reproducible scripts benchmarking TQ against FAISS PQ, OPQ, IVF, and SQ on GloVe, MNIST, and 20 Newsgroups.

## Quick Install (`tqtorch`)

```bash
uv add tqtorch
# or: pip install ./tqtorch
```

## Usage (PyTorch API)

```python
import torch
from tqtorch import TurboQuantIndex

# Create a zero-training 4-bit index
index = TurboQuantIndex(dim=384, bits=4, metric="ip")

# Add embeddings directly
embeddings = torch.randn(10_000, 384)
index.add(embeddings)

# Search
queries = torch.randn(5, 384)
scores, ids = index.search(queries, k=10)
```

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

## Authors
- **Kalyan Reddy Katla**
