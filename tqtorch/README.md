# tqtorch

**GPU-accelerated TurboQuant vector quantization for embeddings & RAG.**

A PyTorch-native implementation of Google's [TurboQuant](https://arxiv.org/abs/2504.19874) algorithm (Zandieh et al., 2025) — scalar quantization via random rotation + Lloyd-Max codebooks — with bit-packing, save/load, and LangChain integration.

## Features

- **Zero training** — rotation matrix is data-independent, no k-means needed
- **Configurable precision** — 1-8 bits per coordinate
- **Two quantizers** — MSE-optimal (Algorithm 1) and inner-product-preserving (Algorithm 2 with QJL)
- **Compact storage** — bit-packing into uint8 tensors
- **Bounded search memory** — configurable batch reconstruction during search
- **Save / Load** — persist and reload indexes via `torch.save`
- **GPU ready** — all operations use PyTorch tensors (CPU or CUDA)
- **LangChain VectorStore** — drop-in integration for RAG pipelines

## Installation

```bash
# With uv (recommended)
uv add tqtorch

# From source
git clone <repo-url> && cd tqtorch
uv sync                       # core only (torch)
uv sync --extra dev           # + pytest, scipy
uv sync --extra bench         # + faiss-cpu, matplotlib
uv sync --extra all           # everything
```

## Quick Start

### Vector Search Index

```python
import torch
from tqtorch import TurboQuantIndex

# Create an index
index = TurboQuantIndex(dim=128, bits=4, metric="ip")

# Add vectors
db = torch.randn(10_000, 128)
index.add(db)

# Search
queries = torch.randn(5, 128)
scores, ids = index.search(queries, k=10)

print(f"Top result for query 0: id={ids[0, 0]}, score={scores[0, 0]:.4f}")
```

### Low-Level Quantizers

```python
from tqtorch import MSEQuantizer, InnerProductQuantizer

# MSE-optimal quantizer (Algorithm 1)
mse_q = MSEQuantizer(dim=128, bits=3, seed=42)
qt = mse_q.quantize(torch.randn(100, 128))
x_hat = mse_q.dequantize(qt)
print(f"Bytes per vector: {mse_q.bytes_per_vector()}")

# Inner-product quantizer (Algorithm 2: MSE + QJL)
ip_q = InnerProductQuantizer(dim=128, bits=4, seed=42)
qt = ip_q.quantize(torch.randn(100, 128))
x_hat = ip_q.dequantize(qt)
```

### Functional API

```python
from tqtorch import mse_quantize, mse_dequantize, ip_quantize, estimate_inner_product

x = torch.randn(100, 64)
qt = mse_quantize(x, bits=4)
x_hat = mse_dequantize(qt)

qt_ip = ip_quantize(x, bits=4)
y = torch.randn(100, 64)
estimates = estimate_inner_product(qt_ip, y)
```

### Save & Load

```python
index.save("my_index.pt")
loaded = TurboQuantIndex.load("my_index.pt")
```

### LangChain Integration

```python
from tqtorch.search.langchain import TurboQuantVectorStore

store = TurboQuantVectorStore.from_texts(
    texts=["hello world", "vector search is cool"],
    embedding=my_embedding_model,  # any LangChain Embeddings
    bits=4,
    metric="ip",
)
docs = store.similarity_search("hello", k=1)
```

## Architecture

```
tqtorch/
├── core/
│   ├── rotation.py       # Random orthogonal rotation (QR)
│   ├── codebook.py       # Lloyd-Max Gaussian codebooks (cached)
│   ├── packed.py          # Bit-packing / unpacking
│   ├── qjl.py            # 1-bit Quantized JL transform
│   ├── mse_quantizer.py  # MSE-optimal quantizer (Algo 1)
│   └── prod_quantizer.py # Inner-product quantizer (Algo 2)
└── search/
    ├── index.py          # TurboQuantIndex (add/search/save/load)
    └── langchain.py      # LangChain VectorStore wrapper
```

## Algorithm Overview

**TurboQuant** quantizes vectors in three steps:

1. **Random rotation** — multiply by a fixed orthogonal matrix Π (QR-based, seeded)
2. **Scalar quantization** — each rotated coordinate is independently quantized using Lloyd-Max centroids optimal for N(0, 1/d)
3. **Bit-packing** — store indices in compact uint8 tensors

The **inner-product variant** (Algorithm 2) splits the bit budget:
- (b-1) bits for MSE quantization
- 1 bit for a QJL (Quantized Johnson-Lindenstrauss) sign on the residual, ensuring unbiased inner-product estimation

## Benchmarks

Run the benchmark scripts from the root repository:

```bash
uv run --extra bench python ../benchmarks/compare_faiss.py
```

Or open the interactive notebook: `../benchmarks/benchmark_vs_faiss.ipynb`

| Method | Mem (MB) | R@1 | R@10 | Training |
|--------|----------|-----|------|----------|
| TurboQuant-4bit | ~0.2 | ★ | ★ | **None** |
| FAISS-PQ(m=16) | ~0.2 | ★ | ★ | k-means |
| FAISS-SQ(8bit) | ~1.3 | ★★ | ★★ | Range scan |

*(★ = data-dependent — run benchmarks for exact numbers on your workload)*

## Testing

```bash
uv run --extra dev pytest tests/ -v
```

## Requirements

- Python ≥ 3.12
- PyTorch ≥ 2.4
- Optional: `langchain-core`, `faiss-cpu`, `matplotlib`, `scipy`

## License

[MIT License](../LICENSE)

MIT