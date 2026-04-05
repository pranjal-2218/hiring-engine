"""
backend/services/parser/jd_parser.py
──────────────────────────────────────
LLM-powered Job Description understanding.

Uses structured prompting to extract:
 - Required / preferred skills with importance weights
 - Experience requirements
 - Education requirements
 - Responsibilities
Retries on failure with exponential backoff.
"""

from __future__ import annotations

import json
import re
from typing import Any

import anthropic
import openai
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.core.config import get_settings
from backend.models.ranking import ParsedJobDescription, RequiredSkill

settings = get_settings()


JD_EXTRACTION_PROMPT = """You are an expert HR analyst. Extract structured information from this job description.

Return ONLY valid JSON matching this exact schema (no markdown, no explanation):
{
  "title": "exact job title",
  "company": "company name or null",
  "location": "location or null",
  "required_skills": [
    {"name": "skill name", "importance": 0.0-1.0, "is_mandatory": true/false}
  ],
  "preferred_skills": ["skill1", "skill2"],
  "min_experience_years": number,
  "max_experience_years": number or null,
  "education_requirement": "e.g. B.Tech in CS or equivalent",
  "responsibilities": ["responsibility 1", "responsibility 2"],
  "benefits": ["benefit 1", "benefit 2"]
}

Rules:
- importance=1.0 means absolutely critical
- Mark is_mandatory=true for hard requirements
- Extract ALL technical skills explicitly mentioned
- min_experience_years should be 0 if not specified
- Keep responsibilities concise (max 10)

JOB DESCRIPTION:
{jd_text}
"""


class JDParser:
    """
    Parses raw JD text into a structured ParsedJobDescription using an LLM.
    Supports both Anthropic Claude and OpenAI as providers.
    """

    def __init__(self):
        self._anthropic_client = None
        self._openai_client = None
        self._init_clients()

    def _init_clients(self):
        if settings.ANTHROPIC_API_KEY:
            self._anthropic_client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        if settings.OPENAI_API_KEY:
            self._openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

    # ── Public API ─────────────────────────────────────────────────────────────

    def parse(self, jd_text: str) -> ParsedJobDescription:
        """Parse JD text into structured format. Falls back to rule-based if LLM fails."""
        try:
            extracted = self._llm_extract(jd_text)
            return self._build_parsed_jd(extracted, jd_text)
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}. Falling back to rule-based.")
            extracted = self._rule_based_extract(jd_text)
            return self._build_parsed_jd(extracted, jd_text)

    # ── LLM Extraction ─────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _llm_extract(self, jd_text: str) -> dict:
        prompt = JD_EXTRACTION_PROMPT.format(jd_text=jd_text[:4000])  # token limit

        if settings.LLM_PROVIDER == "anthropic" and self._anthropic_client:
            return self._call_anthropic(prompt)
        elif self._openai_client:
            return self._call_openai(prompt)
        else:
            raise RuntimeError("No LLM client initialized")

    def _call_anthropic(self, prompt: str) -> dict:
        response = self._anthropic_client.messages.create(
            model=settings.LLM_MODEL,
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
        return self._safe_json_parse(raw)

    def _call_openai(self, prompt: str) -> dict:
        response = self._openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.LLM_TEMPERATURE,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        return self._safe_json_parse(raw)

    def _safe_json_parse(self, raw: str) -> dict:
        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip("`").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}\nRaw: {raw[:200]}")
            raise

    # ── Rule-Based Fallback ────────────────────────────────────────────────────

    def _rule_based_extract(self, jd_text: str) -> dict:
        """
        Regex + keyword-based extraction as fallback.
        Covers common JD patterns without needing an LLM.
        """
        text_lower = jd_text.lower()

        # Extract years of experience
        exp_match = re.search(
            r"(\d+)\+?\s*(?:to\s*(\d+))?\s*years?\s*(?:of\s*)?experience", text_lower
        )
        min_exp = int(exp_match.group(1)) if exp_match else 0
        max_exp = int(exp_match.group(2)) if exp_match and exp_match.group(2) else None

        # Extract skills from tech keyword list
        tech_skills = self._extract_tech_skills(jd_text)

        # Extract title from first non-empty line
        title = next(
            (l.strip() for l in jd_text.split("\n") if l.strip() and len(l.strip()) > 5),
            "Software Engineer",
        )

        return {
            "title": title,
            "company": None,
            "location": None,
            "required_skills": [{"name": s, "importance": 0.8, "is_mandatory": True} for s in tech_skills[:10]],
            "preferred_skills": tech_skills[10:20],
            "min_experience_years": min_exp,
            "max_experience_years": max_exp,
            "education_requirement": self._extract_education_req(jd_text),
            "responsibilities": self._extract_bullets(jd_text),
            "benefits": [],
        }

    def _extract_tech_skills(self, text: str) -> list[str]:
        # Comprehensive tech keyword list
        known_tech = [
            "Python", "Java", "JavaScript", "TypeScript", "Go", "Rust", "C++", "C#",
            "SQL", "NoSQL", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
            "AWS", "GCP", "Azure", "Docker", "Kubernetes", "Terraform", "Ansible",
            "TensorFlow", "PyTorch", "scikit-learn", "Pandas", "NumPy", "Spark",
            "Kafka", "Airflow", "dbt", "FastAPI", "Django", "Flask", "Spring Boot",
            "React", "Angular", "Vue", "Node.js", "GraphQL", "REST", "gRPC",
            "Git", "CI/CD", "Jenkins", "GitHub Actions", "Linux", "Bash",
            "Machine Learning", "Deep Learning", "NLP", "Computer Vision", "MLOps",
            "XGBoost", "LightGBM", "BERT", "Transformers", "LLM", "RAG",
            "Microservices", "DevOps", "Agile", "Scrum",
        ]
        found = []
        for tech in known_tech:
            if re.search(r"\b" + re.escape(tech) + r"\b", text, re.I):
                found.append(tech)
        return found

    def _extract_education_req(self, text: str) -> str:
        patterns = [
            r"(?:b\.?tech|be|bachelor|master|m\.?tech|msc)[^.]*(?:computer science|cs|it|engineering)[^.]*",
            r"(?:degree|qualification)[^.]{0,80}",
        ]
        for p in patterns:
            m = re.search(p, text, re.I)
            if m:
                return m.group(0).strip()
        return ""

    def _extract_bullets(self, text: str) -> list[str]:
        bullets = re.findall(r"[•\-–*]\s*(.+)", text)
        return [b.strip() for b in bullets[:10]]

    # ── Object Construction ────────────────────────────────────────────────────

    def _build_parsed_jd(self, data: dict[str, Any], raw_text: str) -> ParsedJobDescription:
        required_skills = [
            RequiredSkill(
                name=s["name"],
                importance=float(s.get("importance", 0.8)),
                is_mandatory=bool(s.get("is_mandatory", False)),
            )
            for s in data.get("required_skills", [])
            if isinstance(s, dict) and s.get("name")
        ]

        # Build embedding text: concatenate key fields for semantic matching
        embedding_parts = [
            data.get("title", ""),
            " ".join(s["name"] for s in data.get("required_skills", []) if isinstance(s, dict)),
            " ".join(data.get("preferred_skills", [])),
            data.get("education_requirement", ""),
            " ".join(data.get("responsibilities", [])[:5]),
        ]
        embedding_text = " ".join(p for p in embedding_parts if p)

        return ParsedJobDescription(
            title=data.get("title", ""),
            company=data.get("company"),
            location=data.get("location"),
            required_skills=required_skills,
            preferred_skills=data.get("preferred_skills", []),
            min_experience_years=float(data.get("min_experience_years", 0) or 0),
            max_experience_years=data.get("max_experience_years"),
            education_requirement=data.get("education_requirement", ""),
            responsibilities=data.get("responsibilities", []),
            benefits=data.get("benefits", []),
            embedding_text=embedding_text,
            raw_text=raw_text,
        )
