"""
backend/services/explainer/skill_gap.py
─────────────────────────────────────────
Skill Gap Analysis service.

Provides detailed analysis of what skills a candidate lacks
and recommends learning resources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from backend.models.resume import ParsedResume
from backend.models.ranking import ParsedJobDescription, SkillGapResult


# Learning path recommendations by skill category
LEARNING_RESOURCES = {
    "python": "https://docs.python.org/3/tutorial/",
    "machine learning": "https://www.coursera.org/learn/machine-learning",
    "deep learning": "https://www.coursera.org/specializations/deep-learning",
    "sql": "https://mode.com/sql-tutorial/",
    "aws": "https://aws.amazon.com/training/",
    "docker": "https://docs.docker.com/get-started/",
    "kubernetes": "https://kubernetes.io/docs/tutorials/",
    "tensorflow": "https://www.tensorflow.org/tutorials",
    "pytorch": "https://pytorch.org/tutorials/",
    "spark": "https://spark.apache.org/docs/latest/",
    "kafka": "https://kafka.apache.org/quickstart",
    "react": "https://react.dev/learn",
    "fastapi": "https://fastapi.tiangolo.com/tutorial/",
}

ESTIMATED_LEARNING_WEEKS = {
    "python": 8,
    "sql": 4,
    "machine learning": 12,
    "deep learning": 16,
    "docker": 3,
    "kubernetes": 6,
    "aws": 8,
    "react": 8,
    "fastapi": 2,
    "tensorflow": 10,
    "pytorch": 10,
}


@dataclass
class SkillRecommendation:
    skill: str
    priority: str          # "Critical" | "High" | "Medium" | "Low"
    reason: str
    learning_resource: Optional[str] = None
    estimated_weeks: Optional[int] = None


@dataclass
class FullSkillGapReport:
    resume_id: str
    jd_title: str
    skill_gap: SkillGapResult
    recommendations: list[SkillRecommendation] = field(default_factory=list)
    overall_readiness_pct: float = 0.0
    summary: str = ""
    time_to_ready_weeks: Optional[int] = None


class SkillGapAnalyzer:
    """
    Generates detailed skill gap reports with prioritized recommendations.
    """

    def analyze(
        self,
        resume: ParsedResume,
        jd: ParsedJobDescription,
        skill_gap: SkillGapResult,
    ) -> FullSkillGapReport:
        recommendations = []

        # Priority 1: Missing mandatory skills → Critical
        for skill in skill_gap.missing_mandatory:
            recommendations.append(
                SkillRecommendation(
                    skill=skill,
                    priority="Critical",
                    reason=f"'{skill}' is listed as a mandatory requirement for this role",
                    learning_resource=LEARNING_RESOURCES.get(skill.lower()),
                    estimated_weeks=ESTIMATED_LEARNING_WEEKS.get(skill.lower()),
                )
            )

        # Priority 2: Missing preferred skills → High/Medium
        for i, skill in enumerate(skill_gap.missing_preferred[:10]):
            priority = "High" if i < 3 else "Medium"
            recommendations.append(
                SkillRecommendation(
                    skill=skill,
                    priority=priority,
                    reason=f"'{skill}' is preferred and would strengthen the application",
                    learning_resource=LEARNING_RESOURCES.get(skill.lower()),
                    estimated_weeks=ESTIMATED_LEARNING_WEEKS.get(skill.lower()),
                )
            )

        # Sort: Critical → High → Medium → Low
        priority_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        recommendations.sort(key=lambda r: priority_order[r.priority])

        # Compute readiness %
        total_req = len(jd.required_skills) or 1
        readiness = (
            len(skill_gap.matched_skills) / total_req * 100
        )

        # Estimate time to ready
        total_weeks = sum(
            r.estimated_weeks or 6
            for r in recommendations
            if r.priority in ("Critical", "High")
        )
        time_to_ready = total_weeks if total_weeks > 0 else None

        summary = self._generate_summary(
            resume, jd, skill_gap, readiness, recommendations
        )

        return FullSkillGapReport(
            resume_id=str(resume.resume_id),
            jd_title=jd.title,
            skill_gap=skill_gap,
            recommendations=recommendations,
            overall_readiness_pct=round(readiness, 1),
            summary=summary,
            time_to_ready_weeks=time_to_ready,
        )

    def _generate_summary(
        self,
        resume: ParsedResume,
        jd: ParsedJobDescription,
        skill_gap: SkillGapResult,
        readiness: float,
        recommendations: list[SkillRecommendation],
    ) -> str:
        n_matched = len(skill_gap.matched_skills)
        n_missing_crit = len(skill_gap.missing_mandatory)
        n_missing_pref = len(skill_gap.missing_preferred)

        if readiness >= 80 and n_missing_crit == 0:
            verdict = "Strong fit"
            action = "This candidate meets the core requirements and should be fast-tracked."
        elif readiness >= 60 and n_missing_crit <= 1:
            verdict = "Moderate fit"
            action = "Candidate meets most requirements with minor gaps that can be addressed."
        elif n_missing_crit > 2:
            verdict = "Significant gaps"
            action = f"Missing {n_missing_crit} mandatory skills. Consider only if timeline allows upskilling."
        else:
            verdict = "Partial fit"
            action = "Candidate has foundational skills but needs significant upskilling."

        lines = [
            f"**{verdict}** — {readiness:.0f}% skill readiness for '{jd.title}'",
            f"✅ Matched {n_matched} required skills: {', '.join(skill_gap.matched_skills[:5])}",
        ]
        if skill_gap.missing_mandatory:
            lines.append(
                f"❌ Missing mandatory: {', '.join(skill_gap.missing_mandatory)}"
            )
        if skill_gap.missing_preferred:
            lines.append(
                f"⚠️ Missing preferred: {', '.join(skill_gap.missing_preferred[:4])}"
            )
        lines.append(f"\n💡 {action}")

        return "\n".join(lines)
