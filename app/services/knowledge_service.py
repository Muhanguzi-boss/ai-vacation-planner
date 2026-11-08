import os
from pathlib import Path
from typing import Any, Dict, List

from app.core.config import settings
from app.services.chunking_service import chunk_document
from app.services.embedding_service import embedding_service
from app.services.vector_store import SQLiteVectorStore, vector_store


class KnowledgeService:
    def __init__(self, store: SQLiteVectorStore | None = None) -> None:
        self.store = store or vector_store
        self.knowledge_path = Path(settings.KNOWLEDGE_BASE_PATH)
        self.chunk_size = settings.KNOWLEDGE_CHUNK_SIZE
        self.chunk_overlap = settings.KNOWLEDGE_CHUNK_OVERLAP
        self.top_k = settings.KNOWLEDGE_TOP_K
        self.min_score = settings.KNOWLEDGE_MIN_SCORE

    def ingest(self, path: Path | None = None) -> int:
        source_path = path or self.knowledge_path
        records: List[Dict[str, Any]] = []
        for document_path in sorted(source_path.glob("*.md")):
            text = document_path.read_text(encoding="utf-8")
            destination = document_path.stem.replace("-guide", "").replace("-", " ").title()
            chunks = chunk_document(
                text,
                {
                    "destination": destination,
                    "source": document_path.name,
                    "category": "travel-guide",
                },
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            embeddings = embedding_service.embed([chunk.text for chunk in chunks])
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                records.append({
                    "text": chunk.text,
                    "metadata": {**chunk.metadata, "chunk_index": index},
                    "embedding": embedding,
                })
        return self.store.upsert(records)

    def search(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        results = self.store.search(
            embedding_service.embed_query(query),
            top_k or self.top_k,
        )
        return [result for result in results if result["score"] >= self.min_score]


knowledge_service = KnowledgeService()
