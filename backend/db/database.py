"""
backend/db/database.py
───────────────────────
Async SQLAlchemy engine + session factory.
Uses asyncpg driver for PostgreSQL.

Tables:
  - resumes         : parsed resume records
  - job_descriptions: structured JDs
  - ranking_jobs    : ranking run history + results
"""

from __future__ import annotations

from datetime import datetime
from typing import AsyncGenerator
from uuid import uuid4

from loguru import logger
from sqlalchemy import (
    JSON, Boolean, Column, DateTime, Float, Integer,
    String, Text, text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from backend.core.config import get_settings

settings = get_settings()

# ── Engine ─────────────────────────────────────────────────────────────────────
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,       # reconnect on stale connections
    pool_recycle=3600,        # recycle connections every hour
)

AsyncSessionFactory = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# ── Base Model ─────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ── ORM Models ─────────────────────────────────────────────────────────────────

class ResumeRecord(Base):
    __tablename__ = "resumes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    candidate_name = Column(String(200), nullable=False, default="")
    email = Column(String(200), nullable=True, index=True)
    phone = Column(String(50), nullable=True)
    linkedin = Column(String(500), nullable=True)
    github = Column(String(500), nullable=True)
    location = Column(String(200), nullable=True)

    # Structured data stored as JSON
    skills = Column(JSON, nullable=False, default=list)
    education = Column(JSON, nullable=False, default=list)
    experience = Column(JSON, nullable=False, default=list)
    projects = Column(JSON, nullable=False, default=list)
    certifications = Column(JSON, nullable=False, default=list)

    # Aggregates
    total_experience_months = Column(Integer, default=0)
    highest_education_level = Column(String(50), default="other")

    # Quality signals
    is_duplicate = Column(Boolean, default=False)
    duplicate_of = Column(UUID(as_uuid=True), nullable=True)
    has_keyword_stuffing = Column(Boolean, default=False)
    keyword_stuffing_score = Column(Float, default=0.0)

    # Embedding (stored as JSON list for portability)
    embedding = Column(JSON, nullable=True)

    raw_text = Column(Text, nullable=False, default="")
    parsed_at = Column(DateTime, default=datetime.utcnow)
    parse_confidence = Column(Float, default=1.0)


class JobDescriptionRecord(Base):
    __tablename__ = "job_descriptions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    title = Column(String(300), nullable=False)
    company = Column(String(300), nullable=True)
    location = Column(String(300), nullable=True)

    required_skills = Column(JSON, nullable=False, default=list)
    preferred_skills = Column(JSON, nullable=False, default=list)
    min_experience_years = Column(Float, default=0.0)
    max_experience_years = Column(Float, nullable=True)
    education_requirement = Column(Text, default="")
    responsibilities = Column(JSON, nullable=False, default=list)

    embedding_text = Column(Text, default="")
    raw_text = Column(Text, default="")
    embedding = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class RankingRunRecord(Base):
    __tablename__ = "ranking_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    jd_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    jd_title = Column(String(300), nullable=False)
    total_candidates = Column(Integer, default=0)
    results = Column(JSON, nullable=False, default=list)   # RankedCandidate list
    metadata_ = Column("metadata", JSON, nullable=False, default=dict)
    ranked_at = Column(DateTime, default=datetime.utcnow)


# ── Session Dependency ─────────────────────────────────────────────────────────

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions."""
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ── Startup / Init ─────────────────────────────────────────────────────────────

async def init_db() -> None:
    """Create all tables. Called during app startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized")


async def check_db_connection() -> bool:
    """Health check for database connectivity."""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"DB connection failed: {e}")
        return False
