from fastapi import APIRouter, Query

from app.schemas.knowledge import KnowledgeSearchResponse
from app.services.knowledge_service import knowledge_service

router = APIRouter(prefix="/knowledge", tags=["Knowledge"])


@router.get("/search", response_model=KnowledgeSearchResponse)
def search_knowledge(
    query: str = Query(..., min_length=2),
    top_k: int | None = Query(default=None, ge=1, le=20),
):
    return KnowledgeSearchResponse(
        query=query,
        results=knowledge_service.search(query, top_k=top_k),
    )
