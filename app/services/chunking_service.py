from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    metadata: Dict[str, str]


def chunk_document(
    text: str,
    metadata: Dict[str, str],
    chunk_size: int = 900,
    chunk_overlap: int = 120,
) -> List[DocumentChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between 0 and chunk_size")

    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: List[DocumentChunk] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary
        chunk_text = normalized[start:end].strip()
        if chunk_text:
            chunks.append(DocumentChunk(text=chunk_text, metadata=dict(metadata)))
        if end >= len(normalized):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks
