"""
backend/core/config.py
─────────────────────
Central configuration using pydantic-settings.
All secrets loaded from environment / .env file.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    APP_NAME: str = "Intelligent Hiring Engine"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENV: Literal["development", "staging", "production"] = "development"
    SECRET_KEY: str = Field(..., description="JWT / API signing key")
    API_KEY: str = Field(..., description="Header-based API key for external clients")

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://user:password@localhost:5432/hiring_engine"
    )
    MONGODB_URL: str = Field(default="mongodb://localhost:27017")
    MONGODB_DB: str = "hiring_engine"
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── LLM ───────────────────────────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    LLM_PROVIDER: Literal["anthropic", "openai"] = "anthropic"
    LLM_MODEL: str = "claude-3-haiku-20240307"
    LLM_MAX_TOKENS: int = 2048
    LLM_TEMPERATURE: float = 0.0

    # ── Embedding ─────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_CACHE_TTL: int = 3600  # seconds

    # ── ML Model ─────────────────────────────────────────────────────────────
    MODEL_DIR: str = "ml/artifacts"
    RANKER_MODEL_PATH: str = "ml/artifacts/xgb_ranker.json"
    LABEL_ENCODER_PATH: str = "ml/artifacts/label_encoder.pkl"
    SCALER_PATH: str = "ml/artifacts/feature_scaler.pkl"

    # ── Ranking Weights ───────────────────────────────────────────────────────
    WEIGHT_SEMANTIC: float = 0.25
    WEIGHT_ML_SCORE: float = 0.45
    WEIGHT_SKILL_MATCH: float = 0.20
    WEIGHT_EXPERIENCE: float = 0.10

    # ── Upload ────────────────────────────────────────────────────────────────
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: list[str] = [".pdf", ".docx", ".doc"]
    UPLOAD_DIR: str = "data/raw/uploads"

    # ── Duplicate Detection ───────────────────────────────────────────────────
    LSH_THRESHOLD: float = 0.85   # Jaccard similarity threshold
    LSH_NUM_PERM: int = 128

    # ── Keyword Stuffing ─────────────────────────────────────────────────────
    STUFFING_ZSCORE_THRESHOLD: float = 2.5

    # ── CORS ─────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: list[str] = ["http://localhost:8501", "http://localhost:3000"]

    @field_validator("WEIGHT_SEMANTIC", "WEIGHT_ML_SCORE", "WEIGHT_SKILL_MATCH", "WEIGHT_EXPERIENCE")
    @classmethod
    def weights_positive(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Weights must be in [0, 1]")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton — call this everywhere."""
    return Settings()
