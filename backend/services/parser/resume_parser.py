"""
backend/services/parser/resume_parser.py
─────────────────────────────────────────
Production resume parsing pipeline:
  PDF → raw text → spaCy NER → structured ParsedResume

Handles: PDFs, DOCX, multi-column layouts, scanned PDFs (fallback).
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import spacy
from docx import Document
from loguru import logger

from backend.models.resume import (
    Education,
    EducationLevel,
    ParsedResume,
    Project,
    WorkExperience,
)

# ── Load spaCy model once at import time ──────────────────────────────────────
try:
    NLP = spacy.load("en_core_web_lg")
except OSError:
    logger.warning("en_core_web_lg not found, falling back to en_core_web_sm")
    NLP = spacy.load("en_core_web_sm")

# ── Regex patterns ─────────────────────────────────────────────────────────────
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(?:\+91[-\s]?)?(?:\d{10}|\d{3}[-.\s]\d{3}[-.\s]\d{4})")
LINKEDIN_RE = re.compile(r"linkedin\.com/in/[\w-]+", re.I)
GITHUB_RE = re.compile(r"github\.com/[\w-]+", re.I)
URL_RE = re.compile(r"https?://\S+")

EDUCATION_KEYWORDS = {
    "phd": EducationLevel.PHD,
    "ph.d": EducationLevel.PHD,
    "doctorate": EducationLevel.PHD,
    "master": EducationLevel.MASTER,
    "mtech": EducationLevel.MASTER,
    "m.tech": EducationLevel.MASTER,
    "mba": EducationLevel.MASTER,
    "msc": EducationLevel.MASTER,
    "m.sc": EducationLevel.MASTER,
    "bachelor": EducationLevel.BACHELOR,
    "btech": EducationLevel.BACHELOR,
    "b.tech": EducationLevel.BACHELOR,
    "be": EducationLevel.BACHELOR,
    "b.e": EducationLevel.BACHELOR,
    "bsc": EducationLevel.BACHELOR,
    "b.sc": EducationLevel.BACHELOR,
    "diploma": EducationLevel.DIPLOMA,
    "12th": EducationLevel.HIGH_SCHOOL,
    "hsc": EducationLevel.HIGH_SCHOOL,
}

SECTION_HEADERS = {
    "experience": ["experience", "work experience", "employment", "work history", "career"],
    "education": ["education", "academic", "qualification", "degree"],
    "skills": ["skills", "technical skills", "core competencies", "technologies", "tools"],
    "projects": ["projects", "personal projects", "academic projects", "key projects"],
    "certifications": ["certifications", "certificates", "courses", "training"],
    "summary": ["summary", "objective", "profile", "about me", "overview"],
}

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


class ResumeParser:
    """
    End-to-end resume parsing pipeline.

    Usage:
        parser = ResumeParser()
        result: ParsedResume = parser.parse(pdf_bytes, filename="cv.pdf")
    """

    def __init__(self, nlp=None):
        self.nlp = nlp or NLP

    # ── Public API ─────────────────────────────────────────────────────────────

    def parse(self, file_bytes: bytes, filename: str = "resume.pdf") -> ParsedResume:
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            raw_text = self._extract_pdf_text(file_bytes)
        elif ext in (".docx", ".doc"):
            raw_text = self._extract_docx_text(file_bytes)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        if len(raw_text.strip()) < 100:
            logger.warning("Very short extracted text — possible scanned PDF")

        return self._parse_text(raw_text)

    # ── Text Extraction ────────────────────────────────────────────────────────

    def _extract_pdf_text(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF, handling multi-column layouts."""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []
        for page in doc:
            # sort blocks top-to-bottom, left-to-right for multi-column
            blocks = page.get_text("blocks", sort=True)
            page_text = "\n".join(b[4] for b in blocks if b[6] == 0)  # text blocks only
            pages.append(page_text)
        doc.close()
        return "\n\n".join(pages)

    def _extract_docx_text(self, docx_bytes: bytes) -> str:
        doc = Document(io.BytesIO(docx_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    # ── Core Parsing ───────────────────────────────────────────────────────────

    def _parse_text(self, raw_text: str) -> ParsedResume:
        cleaned = self._clean_text(raw_text)
        sections = self._split_sections(cleaned)

        resume = ParsedResume(
            raw_text=raw_text,
            raw_text_word_count=len(raw_text.split()),
        )

        # Extract contact info from full text
        resume.email = self._extract_email(cleaned)
        resume.phone = self._extract_phone(cleaned)
        resume.linkedin = self._extract_pattern(LINKEDIN_RE, cleaned)
        resume.github = self._extract_pattern(GITHUB_RE, cleaned)
        resume.candidate_name = self._extract_name(cleaned)

        # Parse each section
        resume.summary = sections.get("summary", "")
        resume.skills = self._parse_skills(sections.get("skills", ""))
        resume.education = self._parse_education(sections.get("education", ""))
        resume.experience = self._parse_experience(sections.get("experience", ""))
        resume.projects = self._parse_projects(sections.get("projects", ""))
        resume.certifications = self._parse_certifications(
            sections.get("certifications", "")
        )

        # Compute aggregates
        resume.total_experience_months = sum(
            e.duration_months for e in resume.experience
        )
        resume.highest_education_level = self._highest_education(resume.education)

        return resume

    # ── Section Splitting ──────────────────────────────────────────────────────

    def _split_sections(self, text: str) -> dict[str, str]:
        """
        Identify section boundaries by matching header keywords.
        Returns dict of section_name -> section_content.
        """
        lines = text.split("\n")
        sections: dict[str, list[str]] = {}
        current_section = "header"
        sections[current_section] = []

        for line in lines:
            lower_line = line.lower().strip()
            matched_section = self._match_section_header(lower_line)
            if matched_section:
                current_section = matched_section
                sections.setdefault(current_section, [])
            else:
                sections.setdefault(current_section, []).append(line)

        return {k: "\n".join(v).strip() for k, v in sections.items()}

    def _match_section_header(self, line: str) -> Optional[str]:
        for section, keywords in SECTION_HEADERS.items():
            for kw in keywords:
                # Match full-line headers (allow trailing colons, numbers)
                pattern = rf"^{re.escape(kw)}[\s:]*$"
                if re.match(pattern, line.strip()):
                    return section
        return None

    # ── Contact Extraction ─────────────────────────────────────────────────────

    def _extract_email(self, text: str) -> Optional[str]:
        match = EMAIL_RE.search(text)
        return match.group(0).lower() if match else None

    def _extract_phone(self, text: str) -> Optional[str]:
        match = PHONE_RE.search(text)
        return match.group(0) if match else None

    def _extract_pattern(self, pattern: re.Pattern, text: str) -> Optional[str]:
        match = pattern.search(text)
        return match.group(0) if match else None

    def _extract_name(self, text: str) -> str:
        """Use spaCy PERSON entity from the first ~200 chars (header section)."""
        header = text[:300]
        doc = self.nlp(header)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text.strip()
        # Fallback: first non-empty line that's not contact info
        for line in text.split("\n")[:5]:
            line = line.strip()
            if line and not EMAIL_RE.search(line) and not PHONE_RE.search(line):
                if len(line.split()) <= 5:   # names are short
                    return line
        return ""

    # ── Skills Parsing ─────────────────────────────────────────────────────────

    def _parse_skills(self, skills_text: str) -> list[str]:
        if not skills_text:
            return []
        # Split on common delimiters: comma, bullet, pipe, newline
        raw = re.split(r"[,•|·\n\t]+", skills_text)
        skills = []
        for s in raw:
            s = s.strip(" -–•·[]():")
            if s and 1 < len(s) < 60:
                skills.append(s)
        return list(dict.fromkeys(skills))  # deduplicate, preserve order

    # ── Education Parsing ──────────────────────────────────────────────────────

    def _parse_education(self, edu_text: str) -> list[Education]:
        if not edu_text:
            return []

        educations = []
        # Split into blocks by blank lines or bullet points
        blocks = re.split(r"\n{2,}|(?<=\n)(?=•|-|\d\.)", edu_text)

        for block in blocks:
            block = block.strip()
            if not block or len(block) < 10:
                continue

            level = self._detect_education_level(block)
            cgpa = self._extract_cgpa(block)
            year = self._extract_year(block)
            degree, field, institution = self._extract_degree_info(block)

            if degree or institution:
                educations.append(
                    Education(
                        degree=degree,
                        field_of_study=field,
                        institution=institution,
                        graduation_year=year,
                        cgpa=cgpa,
                        level=level,
                    )
                )

        return educations

    def _detect_education_level(self, text: str) -> EducationLevel:
        lower = text.lower()
        for kw, level in EDUCATION_KEYWORDS.items():
            if kw in lower:
                return level
        return EducationLevel.OTHER

    def _extract_cgpa(self, text: str) -> Optional[float]:
        match = re.search(r"(?:cgpa|gpa|score)[:\s]*(\d+\.?\d*)", text, re.I)
        if match:
            val = float(match.group(1))
            # Normalize 100-point scale to 10-point
            if val > 10:
                val = val / 10
            return round(val, 2)
        return None

    def _extract_year(self, text: str) -> Optional[int]:
        matches = re.findall(r"\b(?:19|20)\d{2}\b", text)
        if matches:
            return int(matches[-1])  # graduation year = last year mentioned
        return None

    def _extract_degree_info(self, block: str) -> tuple[str, str, str]:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        degree, field, institution = "", "", ""

        for line in lines:
            lower = line.lower()
            if any(kw in lower for kw in EDUCATION_KEYWORDS):
                degree = line
            elif any(word in lower for word in ["university", "institute", "college", "iit", "nit", "bits"]):
                institution = line
            elif "in" in lower or "of" in lower:
                field = line

        return degree, field, institution

    # ── Experience Parsing ─────────────────────────────────────────────────────

    def _parse_experience(self, exp_text: str) -> list[WorkExperience]:
        if not exp_text:
            return []

        experiences = []
        # Split blocks by double newlines
        blocks = re.split(r"\n{2,}", exp_text)

        for block in blocks:
            block = block.strip()
            if not block or len(block) < 20:
                continue

            company, role, start, end, months, is_current = self._extract_job_meta(block)
            description = self._extract_job_description(block)
            techs = self._extract_technologies(block)

            if company or role:
                experiences.append(
                    WorkExperience(
                        company=company,
                        role=role,
                        start_date=start,
                        end_date=end,
                        duration_months=months,
                        description=description,
                        technologies=techs,
                        is_current=is_current,
                    )
                )

        return experiences

    def _extract_job_meta(self, block: str) -> tuple:
        lines = block.split("\n")
        company, role = "", ""
        start_date, end_date = None, None
        months, is_current = 0, False

        # Date range patterns: "Jan 2021 – Dec 2023" or "2021 - Present"
        date_pattern = re.compile(
            r"([A-Za-z]{3}\.?\s*)?\s*(\d{4})\s*[-–—to]+\s*([A-Za-z]{3}\.?\s*)?\s*(\d{4}|[Pp]resent|[Cc]urrent)",
            re.I,
        )

        for line in lines:
            dm = date_pattern.search(line)
            if dm:
                start_date = f"{dm.group(1) or ''} {dm.group(2)}".strip()
                end_raw = f"{dm.group(3) or ''} {dm.group(4)}".strip()
                if re.search(r"present|current", end_raw, re.I):
                    end_date = "Present"
                    is_current = True
                else:
                    end_date = end_raw
                months = self._calc_duration_months(dm.group(1), dm.group(2), dm.group(3), dm.group(4))
                continue

            # Heuristic: first short line without dates = role / company
            stripped = line.strip("•-–| ")
            if stripped and len(stripped.split()) <= 8:
                if not role and not any(c.isdigit() for c in stripped):
                    role = stripped
                elif not company and not any(c.isdigit() for c in stripped):
                    company = stripped

        return company, role, start_date, end_date, months, is_current

    def _calc_duration_months(self, s_month, s_year, e_month, e_year) -> int:
        try:
            sy = int(s_year)
            ey = int(e_year) if e_year and not re.search(r"present|current", e_year, re.I) else 2024
            sm = MONTH_MAP.get((s_month or "jan").lower()[:3].strip("."), 1)
            em = MONTH_MAP.get((e_month or "dec").lower()[:3].strip("."), 12)
            return max(0, (ey - sy) * 12 + (em - sm))
        except Exception:
            return 0

    def _extract_job_description(self, block: str) -> str:
        # Bullet points typically describe responsibilities
        bullets = re.findall(r"[•\-–*]\s*(.+)", block)
        return " ".join(bullets[:5])  # keep top 5 bullets

    def _extract_technologies(self, text: str) -> list[str]:
        # Look for tech words (CamelCase, abbreviations, version numbers)
        tech_pattern = re.compile(
            r"\b(?:[A-Z][a-zA-Z]+|[A-Z]{2,}|[a-z]+\d+|"
            r"Python|Java|Go|Rust|SQL|NoSQL|AWS|GCP|Azure|"
            r"Docker|Kubernetes|TensorFlow|PyTorch|React|Angular|"
            r"FastAPI|Django|Flask|Spring|Node|Kafka|Spark|Airflow)\b"
        )
        return list(set(tech_pattern.findall(text)))[:15]

    # ── Projects Parsing ───────────────────────────────────────────────────────

    def _parse_projects(self, proj_text: str) -> list[Project]:
        if not proj_text:
            return []

        projects = []
        blocks = re.split(r"\n{2,}|(?<=\n)(?=[A-Z])", proj_text)

        for block in blocks:
            block = block.strip()
            if not block or len(block) < 15:
                continue

            lines = block.split("\n")
            name = lines[0].strip("•-–| *")
            description = " ".join(l.strip("•-| ") for l in lines[1:6])
            techs = self._extract_technologies(block)
            url = self._extract_pattern(URL_RE, block)

            if name:
                projects.append(
                    Project(
                        name=name,
                        description=description,
                        technologies=techs,
                        url=url,
                    )
                )

        return projects[:10]  # cap at 10 projects

    # ── Certifications ─────────────────────────────────────────────────────────

    def _parse_certifications(self, cert_text: str) -> list[str]:
        if not cert_text:
            return []
        raw = re.split(r"[•\n\t,]+", cert_text)
        return [c.strip() for c in raw if c.strip() and len(c.strip()) > 5]

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        # Remove excessive whitespace while preserving structure
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        # Remove null bytes and non-printable chars
        text = "".join(c for c in text if c.isprintable() or c in "\n\t")
        return text.strip()

    def _highest_education(self, educations: list[Education]) -> EducationLevel:
        order = [
            EducationLevel.HIGH_SCHOOL,
            EducationLevel.DIPLOMA,
            EducationLevel.BACHELOR,
            EducationLevel.MASTER,
            EducationLevel.PHD,
        ]
        if not educations:
            return EducationLevel.OTHER
        levels = [e.level for e in educations]
        for level in reversed(order):
            if level in levels:
                return level
        return EducationLevel.OTHER
