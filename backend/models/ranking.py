"""
backend/models/job.py + ranking.py
───────────────────────────────────
Schemas for Job Descriptions and Ranking results.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Job Description Models
# ──────────────────────────────────────────────────────────────────────────────

class RequiredSkill(BaseModel):
    name: str
    importance: float = 1.0   # 0-1, weight for scoring
    is_mandatory: bool = False


class ParsedJobDescription(BaseModel):
    jd_id: UUID = Field(default_factory=uuid4)
    title: str
    company: Optional[str] = None
    location: Optional[str] = None

    # LLM-extracted structured fields
    required_skills: list[RequiredSkill] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    min_experience_years: float = 0.0
    max_experience_years: Optional[float] = None
    education_requirement: str = ""
    responsibilities: list[str] = Field(default_factory=list)
    benefits: list[str] = Field(default_factory=list)

    # For embedding
    embedding_text: str = ""     # cleaned text used for embedding
    raw_text: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ──────────────────────────────────────────────────────────────────────────────
# Ranking Models
# ──────────────────────────────────────────────────────────────────────────────

class SkillGapResult(BaseModel):
    matched_skills: list[str]
    missing_mandatory: list[str]
    missing_preferred: list[str]
    extra_skills: list[str]          # candidate has but JD doesn't need
    skill_match_ratio: float         # matched / total_required


class FeatureVector(BaseModel):
    """All features fed into the XGBoost ranker."""
    resume_id: UUID
    jd_id: UUID

    # Similarity features
    semantic_similarity: float        # cosine sim of embeddings
    skill_match_ratio: float
    mandatory_skill_coverage: float   # ratio of mandatory skills covered
    preferred_skill_coverage: float

    # Experience features
    experience_years: float
    experience_match_score: float     # how well experience aligns with JD
    role_title_similarity: float      # cosine of job title embeddings

    # Education features
    education_level_score: float      # 0-5 scale
    education_field_match: float      # 0-1

    # Quality signals (anti-fraud)
    keyword_stuffing_score: float
    is_duplicate: float               # 0 or 1

    # Project / certification signals
    project_count: int
    has_relevant_projects: float      # 0-1
    certification_count: int


class RankedCandidate(BaseModel):
    rank: int
    resume_id: UUID
    candidate_name: str
    email: Optional[str] = None

    # Scores
    final_score: float               # 0-100
    semantic_score: float
    ml_score: float
    skill_match_score: float
    experience_score: float

    # Explainability
    skill_gap: SkillGapResult
    shap_explanation: dict           # feature -> shap_value
    top_positive_factors: list[str]  # human-readable explanations
    top_negative_factors: list[str]

    # Flags
    is_duplicate: bool = False
    has_keyword_stuffing: bool = False
    recommendation: str              # "Strong Hire" | "Consider" | "Reject"


class RankingResponse(BaseModel):
    jd_id: UUID
    jd_title: str
    total_candidates: int
    ranked_candidates: list[RankedCandidate]
    ranking_metadata: dict
    ranked_at: datetime = Field(default_factory=datetime.utcnow)


class RankingRequest(BaseModel):
    jd_id: UUID
    resume_ids: Optional[list[UUID]] = None   # None = rank all uploaded
    top_k: int = Field(default=10, ge=1, le=100)
    include_rejected: bool = False
