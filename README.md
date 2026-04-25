# TurboQuant

A PyTorch (`tqtorch`) and minimal NumPy (`turboquantlite`) implementation of Google's TurboQuant vector quantization algorithm ([Zandieh et al., 2025](https://arxiv.org/abs/2504.19874)).

> **Compress vectors to 4-bits with 99.4% Recall@10. Zero training time.**

This repo provides drop-in vector indexes and LangChain VectorStores, enabling **~8x memory compression** for RAG and dense retrieval pipelines without the cost, complexity, or dataset-dependency of k-means codebook learning.

## Why TurboQuant?

Real embeddings (sentence-transformers, GloVe, BGE) concentrate information into a few principal components. Naïve scalar quantization loses too much accuracy. Product Quantization (PQ/OPQ) fixes this by learning a data-dependent codebook via k-means — a heavy, expensive step whenever distributions shift.

**TurboQuant eliminates the training step entirely:**

1. **Random Rotation** — Multiply vectors by a seeded, data-independent orthogonal matrix ($\Pi$, via QR decomposition). This spreads information uniformly across all dimensions.
2. **Lloyd-Max Codebooks** — After rotation, coordinates are approximately i.i.d. Gaussian. Scalar-quantize each coordinate optimally into $b$ bits using precomputed Lloyd-Max centroids for $\mathcal{N}(0, 1/d)$.
3. **QJL Correction** *(inner-product variant)* — Allocate $(b{-}1)$ bits for MSE quantization and 1 bit for a Quantized Johnson-Lindenstrauss sign on the residual, yielding an unbiased inner-product estimator.

$$\hat{x} = \Pi^\top \, Q_b(\Pi \, x)$$

**Result:** OPQ-level retrieval accuracy with zero training cost and no dataset dependency.

## Repository Structure

| Directory | Description |
|-----------|-------------|
| [`tqtorch/`](./tqtorch/) | Production PyTorch package — GPU-accelerated, 1-to-8 bit packing, save/load, LangChain integration |
| [`notebooks/`](./notebooks/) | Interactive walkthroughs of the math and intuition (start with `turboquant_small_dim_walkthrough.ipynb`) |
| [`benchmarks/`](./benchmarks/) | Reproducible scripts comparing TQ against FAISS PQ, OPQ, IVF, and SQ on GloVe, MNIST, and 20 Newsgroups |

## Quick Install

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

## References

```bibtex
@article{zandieh2025turboquant,
  title   = {TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author  = {Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  journal = {arXiv preprint arXiv:2504.19874},
  year    = {2025}
}
```

## License

MIT
