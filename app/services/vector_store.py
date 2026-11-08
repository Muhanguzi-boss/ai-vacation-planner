import json
import math
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from app.core.config import settings


class SQLiteVectorStore:
    def __init__(self, path: str | None = None) -> None:
        configured_path = path or settings.VECTOR_STORE_PATH
        self.path = Path(configured_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    UNIQUE(source, chunk_index)
                )
                """
            )

    def upsert(self, chunks: Iterable[Dict[str, Any]]) -> int:
        records = list(chunks)
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO knowledge_chunks(source, chunk_index, text, metadata, embedding)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(source, chunk_index) DO UPDATE SET
                    text = excluded.text,
                    metadata = excluded.metadata,
                    embedding = excluded.embedding
                """ ,
                [
                    (
                        record["metadata"]["source"],
                        record["metadata"]["chunk_index"],
                        record["text"],
                        json.dumps(record["metadata"]),
                        json.dumps(record["embedding"]),
                    )
                    for record in records
                ],
            )
            connection.commit()
        return len(records)

    def search(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        if top_k <= 0:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT text, metadata, embedding FROM knowledge_chunks"
            ).fetchall()
        scored = []
        for row in rows:
            embedding = json.loads(row["embedding"])
            score = self._cosine_similarity(query_embedding, embedding)
            scored.append({
                "text": row["text"],
                "metadata": json.loads(row["metadata"]),
                "score": round(score, 6),
            })
        return sorted(scored, key=lambda result: result["score"], reverse=True)[:top_k]

    @staticmethod
    def _cosine_similarity(left: List[float], right: List[float]) -> float:
        if len(left) != len(right):
            return 0.0
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


vector_store = SQLiteVectorStore()
