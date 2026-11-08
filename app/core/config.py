from pydantic import ConfigDict
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
    LOCAL_EMBEDDING_DIMENSIONS: int = 2048
    KNOWLEDGE_CHUNK_SIZE: int = 900
    KNOWLEDGE_CHUNK_OVERLAP: int = 120
    KNOWLEDGE_TOP_K: int = 5
    KNOWLEDGE_MIN_SCORE: float = 0.2
    MAX_IMAGE_UPLOAD_MB: int = 5
    MCP_TIMEOUT_SECONDS: int = 15
    MCP_WEATHER_ENABLED: bool = False
    MCP_WEATHER_FALLBACK_ENABLED: bool = True
    MAX_TTS_TEXT_CHARS: int = 3000
    WHISPER_MODEL: str = "base"
    WHISPER_COMPUTE_TYPE: str = "int8"
    MAX_AUDIO_UPLOAD_MB: int = 10
    MAX_AUDIO_DURATION_SECONDS: int = 300
    TELEMETRY_ENABLED: bool = True
    TELEMETRY_LOG_ENABLED: bool = True
    DD_METRICS_ENABLED: bool = False
    DD_AGENT_HOST: str = "localhost"
    DD_DOGSTATSD_PORT: int = 8125
    DD_SERVICE: str = "ai-vacation-planner"
    DD_ENV: str = "local"
    DD_VERSION: str = "1.0.0"


settings = Settings()