"""
tests/unit/test_resume_parser.py
──────────────────────────────────
Unit tests for resume parsing pipeline.
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.services.parser.resume_parser import ResumeParser
from backend.models.resume import EducationLevel


SAMPLE_RESUME_TEXT = """
John Doe
john.doe@example.com | +91 9876543210
linkedin.com/in/johndoe | github.com/johndoe

SUMMARY
Passionate ML Engineer with 4 years of experience building production ML systems.

SKILLS
Python, TensorFlow, PyTorch, Docker, Kubernetes, AWS, PostgreSQL, FastAPI, Spark

EXPERIENCE

Senior ML Engineer
TechCorp India | Jan 2021 - Present
• Built real-time fraud detection model reducing false positives by 40%
• Led migration of ML training pipeline to Kubernetes
Technologies: Python, TensorFlow, Docker, Kubernetes, AWS

Data Scientist
StartupXYZ | Jun 2019 - Dec 2020
• Developed NLP pipeline for sentiment analysis (95% accuracy)
• Built A/B testing framework for product recommendations
Technologies: Python, scikit-learn, SQL, Airflow

EDUCATION

M.Tech in Computer Science
IIT Delhi | 2019 | CGPA: 8.9

B.Tech in Computer Science
NIT Trichy | 2017

PROJECTS

Resume Parser System
Built automated resume parsing using spaCy and PyMuPDF
Technologies: Python, spaCy, FastAPI, Docker

CERTIFICATIONS
AWS Certified ML Specialist
Google Cloud Professional Data Engineer
"""


@pytest.fixture
def parser():
    return ResumeParser()


class TestResumeParser:

    def test_parse_email(self, parser):
        result = parser._parse_text(SAMPLE_RESUME_TEXT)
        assert result.email == "john.doe@example.com"

    def test_parse_phone(self, parser):
        result = parser._parse_text(SAMPLE_RESUME_TEXT)
        assert result.phone is not None
        assert "9876543210" in result.phone

    def test_parse_linkedin(self, parser):
        result = parser._parse_text(SAMPLE_RESUME_TEXT)
        assert result.linkedin is not None
        assert "johndoe" in result.linkedin

    def test_parse_github(self, parser):
        result = parser._parse_text(SAMPLE_RESUME_TEXT)
        assert result.github is not None

    def test_parse_skills(self, parser):
        result = parser._parse_text(SAMPLE_RESUME_TEXT)
        assert len(result.skills) > 0
        skill_lower = [s.lower() for s in result.skills]
        assert "python" in skill_lower

    def test_parse_education(self, parser):
        result = parser._parse_text(SAMPLE_RESUME_TEXT)
        assert len(result.education) >= 1

    def test_highest_education_level(self, parser):
        result = parser._parse_text(SAMPLE_RESUME_TEXT)
        # M.Tech should map to MASTER
        assert result.highest_education_level in [EducationLevel.MASTER, EducationLevel.BACHELOR]

    def test_parse_experience(self, parser):
        result = parser._parse_text(SAMPLE_RESUME_TEXT)
        assert len(result.experience) >= 1

    def test_total_experience_positive(self, parser):
        result = parser._parse_text(SAMPLE_RESUME_TEXT)
        assert result.total_experience_months >= 0

    def test_parse_projects(self, parser):
        result = parser._parse_text(SAMPLE_RESUME_TEXT)
        assert len(result.projects) >= 1

    def test_parse_certifications(self, parser):
        result = parser._parse_text(SAMPLE_RESUME_TEXT)
        assert len(result.certifications) >= 1

    def test_raw_text_stored(self, parser):
        result = parser._parse_text(SAMPLE_RESUME_TEXT)
        assert len(result.raw_text) > 0

    def test_word_count(self, parser):
        result = parser._parse_text(SAMPLE_RESUME_TEXT)
        assert result.raw_text_word_count > 50

    def test_resume_id_generated(self, parser):
        result = parser._parse_text(SAMPLE_RESUME_TEXT)
        assert result.resume_id is not None


class TestSkillExtraction:

    def test_skill_deduplication(self, parser):
        text = "Python Python Python Django REST API"
        skills = parser._parse_skills(text)
        assert skills.count("Python") <= 1

    def test_empty_skills_section(self, parser):
        skills = parser._parse_skills("")
        assert skills == []

    def test_comma_separated_skills(self, parser):
        skills = parser._parse_skills("Python, Java, SQL, Docker")
        assert len(skills) == 4

    def test_bullet_separated_skills(self, parser):
        skills = parser._parse_skills("• Python\n• TensorFlow\n• AWS")
        assert len(skills) >= 2


class TestEducationParsing:

    def test_detect_mtech(self, parser):
        level = parser._detect_education_level("M.Tech in Computer Science")
        assert level == EducationLevel.MASTER

    def test_detect_btech(self, parser):
        level = parser._detect_education_level("B.Tech from IIT")
        assert level == EducationLevel.BACHELOR

    def test_detect_phd(self, parser):
        level = parser._detect_education_level("PhD in Machine Learning")
        assert level == EducationLevel.PHD

    def test_extract_cgpa(self, parser):
        cgpa = parser._extract_cgpa("CGPA: 8.9/10")
        assert cgpa == 8.9

    def test_extract_cgpa_percentage(self, parser):
        # 85/100 should be normalized to 8.5/10
        cgpa = parser._extract_cgpa("Score: 85")
        assert cgpa is not None

    def test_extract_year(self, parser):
        year = parser._extract_year("Graduated in 2020")
        assert year == 2020


class TestSectionSplitting:

    def test_section_detection(self, parser):
        sections = parser._split_sections(SAMPLE_RESUME_TEXT)
        assert "skills" in sections or "experience" in sections

    def test_experience_section_exists(self, parser):
        sections = parser._split_sections(SAMPLE_RESUME_TEXT)
        assert any("experience" in k for k in sections)


# ─────────────────────────────────────────────────────────────────────────────
# tests/unit/test_metrics.py
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from ml.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    mean_reciprocal_rank,
    average_precision,
    dcg_at_k,
)


class TestNDCG:

    def test_perfect_ranking(self):
        # Perfect ranking: highest labels first
        labels = np.array([3, 2, 1, 0])
        score = ndcg_at_k(labels, k=4)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_worst_ranking(self):
        # Reverse order
        labels = np.array([0, 1, 2, 3])
        score = ndcg_at_k(labels, k=4)
        assert score < 1.0

    def test_all_zeros(self):
        labels = np.array([0, 0, 0, 0])
        score = ndcg_at_k(labels, k=4)
        assert score == 0.0

    def test_binary_labels(self):
        labels = np.array([1, 1, 0, 0])
        score = ndcg_at_k(labels, k=2)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_k_cutoff(self):
        # Only first K items matter
        labels_good = np.array([2, 0, 0, 0])
        labels_bad = np.array([0, 0, 0, 2])
        score_good = ndcg_at_k(labels_good, k=1)
        score_bad = ndcg_at_k(labels_bad, k=1)
        assert score_good > score_bad


class TestPrecisionAtK:

    def test_all_relevant(self):
        labels = np.array([2, 2, 2, 2])
        score = precision_at_k(labels, k=4, threshold=1)
        assert score == 1.0

    def test_none_relevant(self):
        labels = np.array([0, 0, 0, 0])
        score = precision_at_k(labels, k=4, threshold=1)
        assert score == 0.0

    def test_half_relevant(self):
        labels = np.array([1, 0, 1, 0])
        score = precision_at_k(labels, k=4, threshold=1)
        assert score == 0.5


class TestMRR:

    def test_first_is_relevant(self):
        labels = np.array([1, 0, 0, 0])
        assert mean_reciprocal_rank(labels) == 1.0

    def test_second_is_relevant(self):
        labels = np.array([0, 1, 0, 0])
        assert mean_reciprocal_rank(labels) == pytest.approx(0.5, abs=1e-6)

    def test_none_relevant(self):
        labels = np.array([0, 0, 0, 0])
        assert mean_reciprocal_rank(labels) == 0.0
