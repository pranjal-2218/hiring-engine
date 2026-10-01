"""
backend/services/ml/feature_eng.py
────────────────────────────────────
Feature engineering pipeline.

Converts (ParsedResume, ParsedJobDescription, semantic_score) → FeatureVector
for the XGBoost ranker.

Feature groups:
  1. Similarity features  (semantic, skill match ratios)
  2. Experience features  (years, role alignment)
  3. Education features   (level score, field match)
  4. Quality signals      (stuffing, duplicates)
  5. Richness signals     (project count, certifications)
"""

from __future__ import annotations

import re

import numpy as np
from fastembed import TextEmbedding

from backend.models.resume import Education, EducationLevel, ParsedResume
from backend.models.ranking import (
    FeatureVector,
    ParsedJobDescription,
    SkillGapResult,
)

# Education level → numeric score (0–5)
EDU_LEVEL_SCORE = {
    EducationLevel.OTHER: 0,
    EducationLevel.HIGH_SCHOOL: 1,
    EducationLevel.DIPLOMA: 2,
    EducationLevel.BACHELOR: 3,
    EducationLevel.MASTER: 4,
    EducationLevel.PHD: 5,
}

CS_FIELDS = {
    "computer science", "cs", "information technology", "it",
    "software engineering", "computer engineering", "data science",
    "artificial intelligence", "ai", "electronics", "ece",
}


class FeatureEngineer:
    """
    Converts raw parsed objects into a flat feature vector for the ML ranker.

    Usage:
        fe = FeatureEngineer(embedding_model)
        features = fe.compute(resume, jd, semantic_sim=0.78)
    """

    def __init__(self, embedding_model: TextEmbedding):
        self.emb_model = embedding_model

    def compute(
        self,
        resume: ParsedResume,
        jd: ParsedJobDescription,
        semantic_sim: float,
    ) -> FeatureVector:
        skill_gap = self._compute_skill_gap(resume, jd)

        return FeatureVector(
            resume_id=resume.resume_id,
            jd_id=jd.jd_id,
            # Similarity
            semantic_similarity=round(semantic_sim, 6),
            skill_match_ratio=round(skill_gap.skill_match_ratio, 6),
            mandatory_skill_coverage=round(
                self._mandatory_coverage(resume, jd), 6
            ),
            preferred_skill_coverage=round(
                self._preferred_coverage(resume, jd), 6
            ),
            # Experience
            experience_years=round(resume.total_experience_months / 12, 2),
            experience_match_score=round(
                self._experience_match(resume, jd), 6
            ),
            role_title_similarity=round(
                self._role_title_sim(resume, jd), 6
            ),
            # Education
            education_level_score=round(
                EDU_LEVEL_SCORE.get(resume.highest_education_level, 0) / 5.0, 4
            ),
            education_field_match=round(
                self._education_field_match(resume), 4
            ),
            # Quality
            keyword_stuffing_score=round(resume.keyword_stuffing_score, 6),
            is_duplicate=float(resume.is_duplicate),
            # Richness
            project_count=len(resume.projects),
            has_relevant_projects=round(
                self._project_relevance(resume, jd), 6
            ),
            certification_count=len(resume.certifications),
        )

    def compute_batch(
        self,
        resumes: list[ParsedResume],
        jd: ParsedJobDescription,
        semantic_scores: list[float],
    ) -> list[FeatureVector]:
        return [
            self.compute(r, jd, s)
            for r, s in zip(resumes, semantic_scores)
        ]

    def to_numpy(self, feature_vecs: list[FeatureVector]) -> np.ndarray:
        """Convert list of FeatureVectors to numpy matrix for XGBoost."""
        rows = []
        for fv in feature_vecs:
            rows.append([
                fv.semantic_similarity,
                fv.skill_match_ratio,
                fv.mandatory_skill_coverage,
                fv.preferred_skill_coverage,
                fv.experience_years,
                fv.experience_match_score,
                fv.role_title_similarity,
                fv.education_level_score,
                fv.education_field_match,
                fv.keyword_stuffing_score,
                fv.is_duplicate,
                float(fv.project_count),
                fv.has_relevant_projects,
                float(fv.certification_count),
            ])
        return np.array(rows, dtype=np.float32)

    FEATURE_NAMES = [
        "semantic_similarity",
        "skill_match_ratio",
        "mandatory_skill_coverage",
        "preferred_skill_coverage",
        "experience_years",
        "experience_match_score",
        "role_title_similarity",
        "education_level_score",
        "education_field_match",
        "keyword_stuffing_score",
        "is_duplicate",
        "project_count",
        "has_relevant_projects",
        "certification_count",
    ]

    # ── Skill Gap ──────────────────────────────────────────────────────────────

    def _compute_skill_gap(
        self,
        resume: ParsedResume,
        jd: ParsedJobDescription,
    ) -> SkillGapResult:
        resume_skills_lower = {s.lower() for s in resume.skills}

        # Also include skills from project/experience tech stacks
        for exp in resume.experience:
            resume_skills_lower.update(t.lower() for t in exp.technologies)
        for proj in resume.projects:
            resume_skills_lower.update(t.lower() for t in proj.technologies)

        required_names = [s.name.lower() for s in jd.required_skills]
        mandatory_names = [s.name.lower() for s in jd.required_skills if s.is_mandatory]
        preferred_names = [s.lower() for s in jd.preferred_skills]

        matched = [s for s in required_names if self._skill_match(s, resume_skills_lower)]
        missing_mand = [s for s in mandatory_names if not self._skill_match(s, resume_skills_lower)]
        missing_pref = [s for s in preferred_names if not self._skill_match(s, resume_skills_lower)]
        extra = list(resume_skills_lower - set(required_names) - set(preferred_names))

        total_required = len(required_names) or 1
        ratio = len(matched) / total_required

        return SkillGapResult(
            matched_skills=matched,
            missing_mandatory=missing_mand,
            missing_preferred=missing_pref,
            extra_skills=extra[:10],
            skill_match_ratio=ratio,
        )

    def _skill_match(self, skill: str, resume_skills: set[str]) -> bool:
        """Fuzzy match: exact OR acronym expansion OR substring."""
        if skill in resume_skills:
            return True
        # Substring match for compound skills (e.g. "machine learning" ↔ "ml")
        for rs in resume_skills:
            if skill in rs or rs in skill:
                return True
        return False

    # ── Skill Coverages ───────────────────────────────────────────────────────

    def _mandatory_coverage(self, resume: ParsedResume, jd: ParsedJobDescription) -> float:
        mandatory = [s for s in jd.required_skills if s.is_mandatory]
        if not mandatory:
            return 1.0
        resume_skills = {s.lower() for s in resume.skills}
        matched = sum(1 for s in mandatory if self._skill_match(s.name.lower(), resume_skills))
        return matched / len(mandatory)

    def _preferred_coverage(self, resume: ParsedResume, jd: ParsedJobDescription) -> float:
        if not jd.preferred_skills:
            return 0.5  # neutral if no preferred skills defined
        resume_skills = {s.lower() for s in resume.skills}
        matched = sum(1 for s in jd.preferred_skills if self._skill_match(s.lower(), resume_skills))
        return matched / len(jd.preferred_skills)

    # ── Experience Features ───────────────────────────────────────────────────

    def _experience_match(self, resume: ParsedResume, jd: ParsedJobDescription) -> float:
        """
        Score how well candidate's experience years match JD requirements.
        Penalizes under-experience more than over-experience.
        """
        candidate_years = resume.total_experience_months / 12
        min_req = jd.min_experience_years
        max_req = jd.max_experience_years

        if candidate_years < min_req:
            # Under-experienced: linear decay
            return max(0.0, candidate_years / max(min_req, 1))
        elif max_req and candidate_years > max_req * 1.5:
            # Severely over-experienced (may be too senior)
            return 0.7
        else:
            return 1.0

    def _role_title_sim(self, resume: ParsedResume, jd: ParsedJobDescription) -> float:
        """Semantic similarity between most recent role and JD title."""
        if not resume.experience:
            return 0.0
        last_role = resume.experience[0].role
        if not last_role or not jd.title:
            return 0.0
        embeddings = list(self.emb_model.embed([last_role, jd.title]))
        emb_arr = np.array(embeddings)
        norms = np.linalg.norm(emb_arr, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embs = emb_arr / norms
        return float(max(0.0, np.dot(embs[0], embs[1])))

    # ── Education Features ────────────────────────────────────────────────────

    def _education_field_match(self, resume: ParsedResume) -> float:
        """Check if highest education is in a CS/tech-related field."""
        for edu in resume.education:
            field_lower = edu.field_of_study.lower()
            if any(cs_field in field_lower for cs_field in CS_FIELDS):
                return 1.0
            # Partial match
            for cs_field in CS_FIELDS:
                if any(word in field_lower for word in cs_field.split()):
                    return 0.6
        return 0.0

    # ── Project Relevance ─────────────────────────────────────────────────────

    def _project_relevance(self, resume: ParsedResume, jd: ParsedJobDescription) -> float:
        """Check if projects use JD-required technologies."""
        if not resume.projects:
            return 0.0
        jd_tech = {s.name.lower() for s in jd.required_skills}
        jd_tech.update(s.lower() for s in jd.preferred_skills)

        relevant_count = 0
        for proj in resume.projects:
            proj_tech = {t.lower() for t in proj.technologies}
            if proj_tech & jd_tech:  # intersection
                relevant_count += 1

        return min(1.0, relevant_count / max(len(resume.projects), 1))
