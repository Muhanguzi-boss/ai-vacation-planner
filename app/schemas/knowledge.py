from typing import Any, Dict, List

from pydantic import BaseModel, Field


class KnowledgeResult(BaseModel):
    text: str
    metadata: Dict[str, Any]
    score: float = Field(ge=0, le=1)


class KnowledgeSearchResponse(BaseModel):
    query: str
    results: List[KnowledgeResult]
