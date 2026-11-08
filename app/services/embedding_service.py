import hashlib
import math
from typing import List, Sequence

from app.core.config import settings

SUPPORTED_EMBEDDING_PROVIDERS = ("local",)


class UnsupportedEmbeddingProviderError(ValueError):
    """Raised when EMBEDDING_PROVIDER is set to anything other than ``local``."""


class EmbeddingService:
    """Creates embeddings with a deterministic, local hashing embedder.

    Embeddings are computed in-process: no external embedding API, network access or key is
    used. ``local`` is the only supported provider; any other configured value is rejected
    before embedding anything.
    """

    def __init__(self, provider: str | None = None, dimensions: int | None = None) -> None:
        self.provider = (provider if provider is not None else settings.EMBEDDING_PROVIDER).strip().lower()
        self.dimensions = dimensions if dimensions is not None else settings.LOCAL_EMBEDDING_DIMENSIONS

    def _local_embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        tokens = text.lower().split()
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if self.provider not in SUPPORTED_EMBEDDING_PROVIDERS:
            raise UnsupportedEmbeddingProviderError(
                "Unsupported embedding provider; only 'local' embeddings are supported"
            )
        return [self._local_embed(text) for text in texts]

    def embed_query(self, query: str) -> List[float]:
        return self.embed([query])[0]


embedding_service = EmbeddingService()
