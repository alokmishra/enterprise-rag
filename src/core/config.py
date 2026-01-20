"""
Enterprise RAG System - Configuration Management
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


class Settings(BaseSettings):
    """Application settings loaded from environment variables and config files."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )
    
    # -------------------------------------------------------------------------
    # Application
    # -------------------------------------------------------------------------
    RAG_ENV: str = Field(default="development", description="Environment name")
    DEBUG: bool = Field(default=False, description="Debug mode")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    SECRET_KEY: str = Field(default="change-me-in-production", description="Secret key")

    @field_validator("SECRET_KEY")
    @classmethod
    def validate_secret_key(cls, v, info):
        """Ensure SECRET_KEY is changed in production."""
        rag_env = info.data.get("RAG_ENV", "development")
        if rag_env == "production" and v == "change-me-in-production":
            raise ValueError("SECRET_KEY must be changed in production environment")
        return v
    
    # -------------------------------------------------------------------------
    # Server
    # -------------------------------------------------------------------------
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    WORKERS: int = Field(default=4, description="Number of workers")
    CORS_ORIGINS: str = Field(
        default="http://localhost:3000",
        description="Allowed CORS origins (comma-separated)"
    )
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Get CORS origins as a list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
    
    # -------------------------------------------------------------------------
    # Database
    # -------------------------------------------------------------------------
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://rag:rag@localhost:5432/rag",
        description="Database connection URL"
    )
    DATABASE_POOL_SIZE: int = Field(default=20, description="Connection pool size")
    DATABASE_MAX_OVERFLOW: int = Field(default=10, description="Max overflow connections")
    
    # -------------------------------------------------------------------------
    # Redis
    # -------------------------------------------------------------------------
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    REDIS_CACHE_TTL: int = Field(default=3600, description="Default cache TTL in seconds")
    
    # -------------------------------------------------------------------------
    # Vector Store
    # -------------------------------------------------------------------------
    QDRANT_URL: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL"
    )
    QDRANT_API_KEY: Optional[str] = Field(default=None, description="Qdrant API key")
    QDRANT_COLLECTION_NAME: str = Field(default="documents", description="Default collection")
    
    # -------------------------------------------------------------------------
    # Graph Database
    # -------------------------------------------------------------------------
    NEO4J_URI: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j connection URI"
    )
    NEO4J_USER: str = Field(default="neo4j", description="Neo4j username")
    NEO4J_PASSWORD: str = Field(default="password", description="Neo4j password")
    
    # -------------------------------------------------------------------------
    # LLM Providers
    # -------------------------------------------------------------------------
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, description="Anthropic API key")
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key")
    
    DEFAULT_LLM_PROVIDER: str = Field(default="anthropic", description="Default LLM provider")
    DEFAULT_LLM_MODEL: str = Field(default="claude-sonnet-4-20250514", description="Default model")
    LLM_TEMPERATURE: float = Field(default=0.0, description="LLM temperature")
    LLM_MAX_TOKENS: int = Field(default=4096, description="Max tokens for generation")
    
    # -------------------------------------------------------------------------
    # Embeddings
    # -------------------------------------------------------------------------
    EMBEDDING_PROVIDER: str = Field(default="openai", description="Embedding provider")
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-large", description="Embedding model")
    EMBEDDING_DIMENSIONS: int = Field(default=3072, description="Embedding dimensions")
    
    # -------------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------------
    DEFAULT_TOP_K: int = Field(default=10, description="Default number of results")
    RERANK_TOP_K: int = Field(default=5, description="Results after reranking")
    HYBRID_SEARCH_ALPHA: float = Field(default=0.5, description="Hybrid search weight")
    ENABLE_GRAPH_RETRIEVAL: bool = Field(default=True, description="Enable graph retrieval")
    
    # -------------------------------------------------------------------------
    # Agents
    # -------------------------------------------------------------------------
    AGENT_MAX_ITERATIONS: int = Field(default=3, description="Max agent iterations")
    AGENT_TIMEOUT_SECONDS: int = Field(default=120, description="Agent timeout")
    ENABLE_FACT_VERIFICATION: bool = Field(default=True, description="Enable fact checking")
    ENABLE_CRITIC_AGENT: bool = Field(default=True, description="Enable critic agent")
    
    # -------------------------------------------------------------------------
    # Document Processing
    # -------------------------------------------------------------------------
    MAX_FILE_SIZE_MB: int = Field(default=50, description="Max file size in MB")
    CHUNK_SIZE: int = Field(default=512, description="Default chunk size")
    CHUNK_OVERLAP: int = Field(default=50, description="Chunk overlap")
    
    # -------------------------------------------------------------------------
    # Feature Flags
    # -------------------------------------------------------------------------
    FEATURE_MULTI_MODAL: bool = Field(default=False, description="Enable multi-modal")
    FEATURE_KNOWLEDGE_GRAPH: bool = Field(default=True, description="Enable knowledge graph")
    FEATURE_MULTI_AGENT: bool = Field(default=True, description="Enable multi-agent")
    FEATURE_STREAMING: bool = Field(default=True, description="Enable streaming responses")

    # -------------------------------------------------------------------------
    # Deployment Configuration
    # -------------------------------------------------------------------------
    DEPLOYMENT_MODE: str = Field(
        default="saas_multi_tenant",
        description="Deployment mode: saas_multi_tenant, saas_dedicated, on_premise, air_gapped"
    )
    TENANT_ISOLATION: str = Field(
        default="shared",
        description="Tenant isolation: shared, schema, database, instance"
    )

    # -------------------------------------------------------------------------
    # Local LLM Configuration
    # -------------------------------------------------------------------------
    LOCAL_LLM_ENABLED: bool = Field(default=False, description="Enable local LLM")
    LOCAL_LLM_MODEL_PATH: Optional[str] = Field(default=None, description="Path to local LLM model")
    LOCAL_LLM_MODEL_TYPE: str = Field(default="llama", description="Local LLM type: llama, mistral")
    LOCAL_LLM_CONTEXT_LENGTH: int = Field(default=4096, description="Local LLM context length")
    LOCAL_LLM_GPU_LAYERS: int = Field(default=0, description="GPU layers for local LLM")

    # -------------------------------------------------------------------------
    # Telemetry Configuration
    # -------------------------------------------------------------------------
    TELEMETRY_ENABLED: bool = Field(default=True, description="Enable telemetry")
    
    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    @property
    def BASE_DIR(self) -> Path:
        """Get the base directory of the project."""
        return Path(__file__).parent.parent.parent
    
    @property
    def CONFIG_DIR(self) -> Path:
        """Get the config directory."""
        return self.BASE_DIR / "configs"
    
    @property
    def DATA_DIR(self) -> Path:
        """Get the data directory."""
        return self.BASE_DIR / "data"
    
    def load_yaml_config(self, name: str) -> dict:
        """Load a YAML configuration file."""
        config_path = self.CONFIG_DIR / f"{name}.yaml"
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {}


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
