import hashlib
import math
import os
from typing import List, Sequence

from app.core.config import settings


class EmbeddingService:
    """Creates embeddings using OpenAI or a deterministic local test-friendly model."""

    def __init__(self) -> None:
        self.provider = settings.EMBEDDING_PROVIDER.lower()
        self.model = settings.EMBEDDING_MODEL
        self.dimensions = settings.LOCAL_EMBEDDING_DIMENSIONS
        self.api_key = settings.OPENAI_API_KEY or os.getenv("openai_api_key")
        self._client = None

    def _openai_client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

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
        if self.provider == "local":
            return [self._local_embed(text) for text in texts]
        if self.provider != "openai":
            raise ValueError(f"Unsupported embedding provider: {self.provider}")
        response = self._openai_client().embeddings.create(input=list(texts), model=self.model)
        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> List[float]:
        return self.embed([query])[0]


embedding_service = EmbeddingService()
