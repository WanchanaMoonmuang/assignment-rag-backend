from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gemini_provider: str = "developer_api"
    gemini_api_key: str | None = None
    mongodb_uri: str | None = None
    mongodb_database: str = "poc_rag"
    mongodb_chunk_collection: str = "document_chunks"
    mongodb_document_collection: str = "documents"
    mongodb_conversation_collection: str = "conversations"
    mongodb_ingestion_job_collection: str = "ingestion_jobs"
    mongodb_vector_index: str | None = None
    mongodb_search_index: str = "document_chunks_text_index"
    gemini_model: str = "gemini-3.5-flash"
    gemini_embedding_model: str = "gemini-embedding-2"
    gemini_embedding_dimensions: int = 768
    google_cloud_project: str | None = None
    gcp_project_id: str | None = None
    google_cloud_location: str = "us-central1"
    backend_cors_origins: str = "http://localhost:5173"
    auth_username: str = "admin"
    auth_password: str = ""
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    rag_top_k: int = Field(default=5, ge=0, le=20)
    history_context_window: int = Field(default=8, ge=0, le=100)
    gemini_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    generation_context_token_budget: int = Field(default=32_000, ge=1)
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, ge=1, le=20 * 1024 * 1024)
    gcs_bucket_name: str | None = None
    gcs_project_id: str | None = None
    ingestion_job_lease_seconds: int = Field(default=300, ge=30, le=3600)
    ingestion_job_max_attempts: int = Field(default=3, ge=1, le=10)
    ingestion_processing_timeout_seconds: int = Field(default=600, ge=60, le=3600)
    max_document_bytes: int = Field(default=1_048_576, ge=1)
    rag_chunk_size: int = Field(default=900, ge=1)
    rag_chunk_overlap: int = Field(default=150, ge=0)

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.backend_cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
