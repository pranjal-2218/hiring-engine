"""
backend/models/resume.py
────────────────────────
Pydantic v2 schemas for resume data flowing through the pipeline.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class EducationLevel(str, Enum):
    HIGH_SCHOOL = "high_school"
    DIPLOMA = "diploma"
    BACHELOR = "bachelor"
    MASTER = "master"
    PHD = "phd"
    OTHER = "other"


class Education(BaseModel):
    degree: str
    field_of_study: str
    institution: str
    graduation_year: Optional[int] = None
    cgpa: Optional[float] = None
    level: EducationLevel = EducationLevel.OTHER

    @field_validator("cgpa")
    @classmethod
    def validate_cgpa(cls, v):
        if v is not None and not (0.0 <= v <= 10.0):
            raise ValueError("CGPA must be between 0 and 10")
        return v


class WorkExperience(BaseModel):
    company: str
    role: str
    start_date: Optional[str] = None    # "Jan 2021"
    end_date: Optional[str] = None      # "Dec 2023" or "Present"
    duration_months: int = 0
    description: str = ""
    technologies: list[str] = Field(default_factory=list)
    is_current: bool = False


class Project(BaseModel):
    name: str
    description: str
    technologies: list[str] = Field(default_factory=list)
    url: Optional[str] = None
    impact: Optional[str] = None  # quantified impact if mentioned


class ParsedResume(BaseModel):
    resume_id: UUID = Field(default_factory=uuid4)
    candidate_name: str = ""
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    location: Optional[str] = None
    summary: str = ""

    # Core sections
    skills: list[str] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    experience: list[WorkExperience] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)

    # Computed during parsing
    total_experience_months: int = 0
    highest_education_level: EducationLevel = EducationLevel.OTHER
    raw_text: str = ""
    raw_text_word_count: int = 0

    # Quality signals
    is_duplicate: bool = False
    duplicate_of: Optional[UUID] = None
    has_keyword_stuffing: bool = False
    keyword_stuffing_score: float = 0.0

    parsed_at: datetime = Field(default_factory=datetime.utcnow)
    parse_confidence: float = 1.0  # 0-1, how confident the parser is


class ResumeUploadResponse(BaseModel):
    resume_id: UUID
    message: str
    parse_status: str
    warnings: list[str] = Field(default_factory=list)
