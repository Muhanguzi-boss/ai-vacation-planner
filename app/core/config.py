from pydantic import ConfigDict
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    KNOWLEDGE_BASE_PATH: str = "data/knowledge"
    VECTOR_STORE_PATH: str = "data/vector_store.db"
    EMBEDDING_PROVIDER: str = "local"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LOCAL_EMBEDDING_DIMENSIONS: int = 2048
    OPENAI_API_KEY: Optional[str] = None
    KNOWLEDGE_CHUNK_SIZE: int = 900
    KNOWLEDGE_CHUNK_OVERLAP: int = 120
    KNOWLEDGE_TOP_K: int = 5
    KNOWLEDGE_MIN_SCORE: float = 0.2


settings = Settings()