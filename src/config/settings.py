"""
Configuration settings for WaveMaker Docs Agent.
Uses pydantic-settings for type-safe environment variable loading.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # === AI Provider Selection ===
    ai_provider: str = Field(
        default="anthropic",
        description="AI provider: 'anthropic', 'openai', or 'ollama'",
    )

    # === Provider-Specific API Keys ===
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key for Claude",
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for GPT models",
    )

    # === Provider-Specific Models ===
    anthropic_model: str = Field(
        default="claude-sonnet-4-5-20250929",
        description="Anthropic Claude model to use",
    )
    openai_model: str = Field(
        default="gpt-4o",
        description="OpenAI model to use",
    )
    ollama_model: str = Field(
        default="llama3.2",
        description="Ollama model to use",
    )

    # === Ollama Configuration ===
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server base URL",
    )

    # === Redis Configuration ===
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )

    # === Qdrant Configuration ===
    qdrant_url: str = Field(..., description="Qdrant Cloud cluster URL")
    qdrant_api_key: str = Field(..., description="Qdrant API key")
    qdrant_collection_name: str = Field(
        default="wavemaker_docs",
        description="Name of the Qdrant collection",
    )

    # === WaveMaker Docs Configuration ===
    docs_repo_url: str = Field(
        default="https://github.com/wavemaker/docs.git",
        description="WaveMaker docs repository URL",
    )
    docs_branch: str = Field(
        default="release-12",
        description="Git branch to index",
    )
    docs_base_url: str = Field(
        default="https://docs.wavemaker.com/learn",
        description="Base URL for documentation links",
    )

    # === Academy MCP Configuration ===
    academy_mcp_url: Optional[str] = Field(
        default="https://dev-academyservices.wavemaker.com/mcp",
        description="Academy MCP server URL (optional)",
    )

    # === Model Configuration ===
    embedding_model: str = Field(
        default="BAAI/bge-base-en-v1.5",
        description="Sentence transformer model for embeddings",
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for local reranking",
    )
    
    # === Reranker Provider ===
    # Options: "jina" (recommended for Apple Silicon), "local" (cross-encoder)
    reranker_provider: str = Field(
        default="jina",
        description="Reranker provider: 'jina' (API) or 'local' (cross-encoder)",
    )
    jina_api_key: Optional[str] = Field(
        default=None,
        description="Jina AI API key for reranking",
    )

    # === LLM Configuration (shared across providers) ===
    llm_temperature: float = Field(
        default=0.2,
        description="Temperature for LLM generation",
    )
    llm_max_tokens: int = Field(
        default=2048,
        description="Maximum tokens for LLM response",
    )

    # === Cache Configuration ===
    cache_ttl_hours: int = Field(
        default=1,
        description="Cache TTL in hours",
    )
    semantic_cache_threshold: float = Field(
        default=0.95,
        description="Similarity threshold for semantic cache hits",
    )
    semantic_cache_max_candidates: int = Field(
        default=200,
        description="Number of semantic cache entries to compare per lookup",
    )

    # === Retrieval Configuration ===
    retrieval_top_k: int = Field(
        default=30,
        description="Number of documents to retrieve per search method",
    )
    rerank_top_k: int = Field(
        default=5,
        description="Number of documents to pass to LLM after reranking",
    )
    reranker_candidates: int = Field(
        default=12,
        description="Maximum documents to send to reranker provider",
    )
    reranker_enabled: bool = Field(
        default=True,
        description="Enable reranking (only used when provider is 'local')",
    )

    # === Server Configuration ===
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")

    # === Chunking Configuration ===
    chunk_min_tokens: int = Field(default=100, description="Minimum chunk size")
    chunk_max_tokens: int = Field(default=512, description="Maximum chunk size")
    chunk_target_tokens: int = Field(default=350, description="Target chunk size")

    # === Analytics Configuration ===
    analytics_enabled: bool = Field(
        default=True,
        description="Enable analytics tracking",
    )
    analytics_db_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/docs_agent",
        description="Postgres connection URL for analytics",
    )
    analytics_batch_size: int = Field(
        default=100,
        description="Number of events to batch before writing to DB",
    )
    analytics_flush_interval: int = Field(
        default=5,
        description="Seconds between forced flushes to DB",
    )
    analytics_redis_key: str = Field(
        default="docs_agent:analytics:queue",
        description="Redis key for analytics event queue",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to avoid re-reading .env file on every call.
    """
    return Settings()
