"""LangChain VectorStore integration for TurboQuant."""

from __future__ import annotations

from typing import Any

import torch

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore

    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False


def _require_langchain():
    if not _HAS_LANGCHAIN:
        raise ImportError(
            "langchain-core is required for TurboQuantVectorStore. "
            "Install it with: pip install langchain-core"
        )


if _HAS_LANGCHAIN:

    class TurboQuantVectorStore(VectorStore):
        """LangChain VectorStore backed by TurboQuantIndex.

        Parameters
        ----------
        embedding : Embeddings
            Embedding model (must implement ``embed_query`` and ``embed_documents``).
        dim : int
            Embedding dimensionality.
        bits : int
            Bits per coordinate.
        metric : str
            ``"ip"`` or ``"mse"``.
        seed : int
            Random seed.
        device : str or torch.device or None
            Target device.
        """

        def __init__(
            self,
            embedding: Embeddings,
            dim: int,
            bits: int = 4,
            metric: str = "ip",
            seed: int = 42,
            device: str | torch.device | None = None,
        ):
            _require_langchain()
            from tqtorch.search.index import TurboQuantIndex

            self._embedding = embedding
            self._index = TurboQuantIndex(
                dim=dim, bits=bits, metric=metric, seed=seed, device=device,
            )
            self._documents: list[Document] = []

        @property
        def embeddings(self) -> Embeddings:
            return self._embedding

        def add_texts(
            self,
            texts: list[str],
            metadatas: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> list[str]:
            """Embed texts and add to the index."""
            vectors = self._embedding.embed_documents(list(texts))
            x = torch.tensor(vectors, dtype=torch.float32)
            self._index.add(x)

            ids = []
            for i, text in enumerate(texts):
                meta = metadatas[i] if metadatas else {}
                doc_id = str(self._index.ntotal - len(texts) + i)
                self._documents.append(Document(page_content=text, metadata=meta))
                ids.append(doc_id)
            return ids

        def similarity_search(
            self,
            query: str,
            k: int = 4,
            **kwargs: Any,
        ) -> list[Document]:
            """Return the k most similar documents."""
            results = self.similarity_search_with_score(query, k=k, **kwargs)
            return [doc for doc, _score in results]

        def similarity_search_with_score(
            self,
            query: str,
            k: int = 4,
            **kwargs: Any,
        ) -> list[tuple[Document, float]]:
            """Return (document, score) pairs for the k most similar documents."""
            q_vec = self._embedding.embed_query(query)
            q_tensor = torch.tensor([q_vec], dtype=torch.float32)
            scores, indices = self._index.search(q_tensor, k=k)

            results = []
            for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
                if idx < len(self._documents):
                    results.append((self._documents[idx], score))
            return results

        @classmethod
        def from_texts(
            cls,
            texts: list[str],
            embedding: Embeddings,
            metadatas: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> TurboQuantVectorStore:
            """Create a TurboQuantVectorStore from a list of texts."""
            # Need dim from a sample embedding
            sample = embedding.embed_query(texts[0])
            dim = len(sample)
            store = cls(embedding=embedding, dim=dim, **kwargs)
            store.add_texts(texts, metadatas=metadatas)
            return store
