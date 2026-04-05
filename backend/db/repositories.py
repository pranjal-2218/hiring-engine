"""
backend/db/repositories.py
───────────────────────────
Data access layer using Repository pattern.
Abstracts all DB operations behind clean interfaces.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import (
    JobDescriptionRecord,
    RankingRunRecord,
    ResumeRecord,
)
from backend.models.resume import ParsedResume
from backend.models.ranking import ParsedJobDescription, RankingResponse


class ResumeRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, resume: ParsedResume) -> ResumeRecord:
        record = ResumeRecord(
            id=resume.resume_id,
            candidate_name=resume.candidate_name,
            email=resume.email,
            phone=resume.phone,
            linkedin=resume.linkedin,
            github=resume.github,
            location=resume.location,
            skills=resume.skills,
            education=[e.model_dump() for e in resume.education],
            experience=[e.model_dump() for e in resume.experience],
            projects=[p.model_dump() for p in resume.projects],
            certifications=resume.certifications,
            total_experience_months=resume.total_experience_months,
            highest_education_level=resume.highest_education_level.value,
            is_duplicate=resume.is_duplicate,
            duplicate_of=resume.duplicate_of,
            has_keyword_stuffing=resume.has_keyword_stuffing,
            keyword_stuffing_score=resume.keyword_stuffing_score,
            raw_text=resume.raw_text[:50000],   # cap at 50KB
            parse_confidence=resume.parse_confidence,
        )
        self.session.add(record)
        await self.session.flush()
        return record

    async def get_by_id(self, resume_id: UUID) -> Optional[ResumeRecord]:
        result = await self.session.execute(
            select(ResumeRecord).where(ResumeRecord.id == resume_id)
        )
        return result.scalar_one_or_none()

    async def get_all(self, skip: int = 0, limit: int = 100) -> list[ResumeRecord]:
        result = await self.session.execute(
            select(ResumeRecord).offset(skip).limit(limit)
        )
        return list(result.scalars().all())

    async def count(self) -> int:
        from sqlalchemy import func
        result = await self.session.execute(
            select(func.count()).select_from(ResumeRecord)
        )
        return result.scalar() or 0

    async def delete(self, resume_id: UUID) -> bool:
        result = await self.session.execute(
            delete(ResumeRecord).where(ResumeRecord.id == resume_id)
        )
        return result.rowcount > 0

    async def mark_duplicate(self, resume_id: UUID, original_id: UUID) -> None:
        await self.session.execute(
            update(ResumeRecord)
            .where(ResumeRecord.id == resume_id)
            .values(is_duplicate=True, duplicate_of=original_id)
        )


class JobDescriptionRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, jd: ParsedJobDescription) -> JobDescriptionRecord:
        record = JobDescriptionRecord(
            id=jd.jd_id,
            title=jd.title,
            company=jd.company,
            location=jd.location,
            required_skills=[s.model_dump() for s in jd.required_skills],
            preferred_skills=jd.preferred_skills,
            min_experience_years=jd.min_experience_years,
            max_experience_years=jd.max_experience_years,
            education_requirement=jd.education_requirement,
            responsibilities=jd.responsibilities,
            embedding_text=jd.embedding_text,
            raw_text=jd.raw_text[:50000],
        )
        self.session.add(record)
        await self.session.flush()
        return record

    async def get_by_id(self, jd_id: UUID) -> Optional[JobDescriptionRecord]:
        result = await self.session.execute(
            select(JobDescriptionRecord).where(JobDescriptionRecord.id == jd_id)
        )
        return result.scalar_one_or_none()

    async def get_all(self) -> list[JobDescriptionRecord]:
        result = await self.session.execute(select(JobDescriptionRecord))
        return list(result.scalars().all())

    async def delete(self, jd_id: UUID) -> bool:
        result = await self.session.execute(
            delete(JobDescriptionRecord).where(JobDescriptionRecord.id == jd_id)
        )
        return result.rowcount > 0


class RankingRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save_run(self, response: RankingResponse) -> RankingRunRecord:
        record = RankingRunRecord(
            jd_id=response.jd_id,
            jd_title=response.jd_title,
            total_candidates=response.total_candidates,
            results=[c.model_dump(mode="json") for c in response.ranked_candidates],
            metadata_=response.ranking_metadata,
        )
        self.session.add(record)
        await self.session.flush()
        return record

    async def get_runs_for_jd(self, jd_id: UUID) -> list[RankingRunRecord]:
        result = await self.session.execute(
            select(RankingRunRecord)
            .where(RankingRunRecord.jd_id == jd_id)
            .order_by(RankingRunRecord.ranked_at.desc())
        )
        return list(result.scalars().all())
