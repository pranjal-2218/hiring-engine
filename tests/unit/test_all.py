"""
tests/unit/test_parser.py
──────────────────────────
Unit tests for resume parsing pipeline.
"""

from __future__ import annotations

import pytest

from backend.models.resume import EducationLevel
from backend.services.parser.resume_parser import ResumeParser


@pytest.fixture
def parser():
    return ResumeParser()


SAMPLE_RESUME_TEXT = """
John Doe
john.doe@gmail.com | +91-9876543210 | linkedin.com/in/johndoe | github.com/johndoe
Bangalore, India

SUMMARY
Senior ML Engineer with 5+ years of experience building production ML systems.

SKILLS
Python, TensorFlow, PyTorch, scikit-learn, XGBoost, SQL, PostgreSQL, Docker, Kubernetes, AWS

EXPERIENCE
Senior ML Engineer | Google India | Jan 2021 - Present
• Built recommendation system serving 10M users
• Designed MLOps pipeline with Kubeflow and Airflow
• Python, TensorFlow, BigQuery, Kubernetes

ML Engineer | Flipkart | Jun 2019 - Dec 2020
• Developed NLP-based search ranking
• Python, BERT, Elasticsearch

EDUCATION
M.Tech in Computer Science | IIT Bangalore | 2019
CGPA: 8.9/10

B.Tech in Computer Science | NIT Trichy | 2017

PROJECTS
Real-time Fraud Detection System
• Built streaming fraud detection using Kafka and Spark
• Python, Kafka, Spark, XGBoost, Docker
"""


def test_extract_email(parser):
    result = parser._parse_text(SAMPLE_RESUME_TEXT)
    assert result.email == "john.doe@gmail.com"


def test_extract_phone(parser):
    result = parser._parse_text(SAMPLE_RESUME_TEXT)
    assert result.phone is not None
    assert "9876543210" in result.phone


def test_extract_linkedin(parser):
    result = parser._parse_text(SAMPLE_RESUME_TEXT)
    assert result.linkedin is not None
    assert "johndoe" in result.linkedin


def test_extract_skills(parser):
    result = parser._parse_text(SAMPLE_RESUME_TEXT)
    skills_lower = [s.lower() for s in result.skills]
    assert "python" in skills_lower
    assert "tensorflow" in skills_lower
    assert "docker" in skills_lower


def test_extract_education(parser):
    result = parser._parse_text(SAMPLE_RESUME_TEXT)
    assert len(result.education) >= 1
    # Check highest level is at least bachelor
    assert result.highest_education_level in [EducationLevel.MASTER, EducationLevel.BACHELOR]


def test_extract_experience(parser):
    result = parser._parse_text(SAMPLE_RESUME_TEXT)
    assert result.total_experience_months > 0


def test_extract_projects(parser):
    result = parser._parse_text(SAMPLE_RESUME_TEXT)
    assert len(result.projects) >= 1


def test_word_count(parser):
    result = parser._parse_text(SAMPLE_RESUME_TEXT)
    assert result.raw_text_word_count > 50


# ──────────────────────────────────────────────────────────────────────────────

"""
tests/unit/test_metrics.py
"""

from ml.evaluation.metrics import (
    ndcg_at_k, precision_at_k, mean_reciprocal_rank, average_precision,
)


def test_ndcg_perfect():
    """Perfect ranking: NDCG = 1.0"""
    labels = [3, 2, 1, 0]
    ideal = [3, 2, 1, 0]
    assert ndcg_at_k(labels, ideal, k=4) == pytest.approx(1.0, abs=1e-6)


def test_ndcg_worst():
    """Worst ranking (reversed): NDCG < 0.5"""
    labels = [0, 1, 2, 3]
    ideal = [3, 2, 1, 0]
    result = ndcg_at_k(labels, ideal, k=4)
    assert result < 0.9   # should be significantly less than perfect


def test_ndcg_all_zero():
    """All-zero labels: trivially perfect (no relevant items to rank)."""
    labels = [0, 0, 0]
    ideal = [0, 0, 0]
    assert ndcg_at_k(labels, ideal, k=3) == 1.0


def test_precision_at_k():
    labels = [3, 2, 0, 1, 0]  # relevant (≥2): 3, 2
    assert precision_at_k(labels, k=3, threshold=2) == pytest.approx(2/3, abs=1e-6)
    assert precision_at_k(labels, k=5, threshold=2) == pytest.approx(2/5, abs=1e-6)


def test_mrr_first_item():
    labels = [2, 0, 0, 0]
    assert mean_reciprocal_rank(labels, threshold=2) == pytest.approx(1.0)


def test_mrr_third_item():
    labels = [0, 0, 2, 0]
    assert mean_reciprocal_rank(labels, threshold=2) == pytest.approx(1/3, abs=1e-6)


def test_mrr_no_relevant():
    labels = [0, 0, 0]
    assert mean_reciprocal_rank(labels, threshold=2) == 0.0


def test_average_precision():
    labels = [1, 0, 1, 0, 1]
    ap = average_precision(labels, threshold=1)
    assert 0.0 < ap <= 1.0


# ──────────────────────────────────────────────────────────────────────────────

"""
tests/unit/test_feature_eng.py
"""

import uuid
from unittest.mock import MagicMock

from backend.models.resume import ParsedResume, WorkExperience
from backend.models.ranking import ParsedJobDescription, RequiredSkill


def make_resume(skills, exp_months=24):
    r = ParsedResume()
    r.resume_id = uuid.uuid4()
    r.skills = skills
    r.total_experience_months = exp_months
    r.raw_text = " ".join(skills)
    return r


def make_jd(required, mandatory_count=3):
    jd = ParsedJobDescription(title="ML Engineer")
    jd.jd_id = uuid.uuid4()
    jd.required_skills = [
        RequiredSkill(name=s, importance=0.9, is_mandatory=(i < mandatory_count))
        for i, s in enumerate(required)
    ]
    jd.min_experience_years = 2.0
    return jd


def test_skill_match_perfect():
    from backend.services.ml.feature_eng import FeatureEngineer
    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=__import__("numpy").array([[0.5, 0.5]]))
    fe = FeatureEngineer(mock_model)
    resume = make_resume(["Python", "TensorFlow", "Docker", "Kubernetes"])
    jd = make_jd(["Python", "TensorFlow", "Docker"])
    gap = fe._compute_skill_gap(resume, jd)
    assert gap.skill_match_ratio == pytest.approx(1.0, abs=0.01)


def test_skill_match_partial():
    from backend.services.ml.feature_eng import FeatureEngineer
    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=__import__("numpy").array([[0.5, 0.5]]))
    fe = FeatureEngineer(mock_model)
    resume = make_resume(["Python"])   # only 1 of 3
    jd = make_jd(["Python", "TensorFlow", "Docker"])
    gap = fe._compute_skill_gap(resume, jd)
    assert gap.skill_match_ratio < 0.5
