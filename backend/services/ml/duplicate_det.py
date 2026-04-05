"""
backend/services/ml/duplicate_det.py
──────────────────────────────────────
MinHash LSH-based duplicate resume detection.
Detects near-duplicate resumes (same candidate submitted multiple times
or slightly modified versions).
"""

from __future__ import annotations

from uuid import UUID

from datasketch import MinHash, MinHashLSH
from loguru import logger

from backend.core.config import get_settings
from backend.models.resume import ParsedResume

settings = get_settings()


class DuplicateDetector:
    """
    Uses MinHash + Locality Sensitive Hashing to find near-duplicate resumes.

    Process:
      1. Shingling: convert text to character n-grams (shingles)
      2. MinHash: approximate Jaccard similarity via min-hash signatures
      3. LSH: index signatures for sub-linear similarity search

    Jaccard(A, B) = |shingles(A) ∩ shingles(B)| / |shingles(A) ∪ shingles(B)|
    """

    def __init__(self):
        self._lsh = MinHashLSH(
            threshold=settings.LSH_THRESHOLD,
            num_perm=settings.LSH_NUM_PERM,
        )
        self._registered: dict[UUID, MinHash] = {}

    def add_resume(self, resume: ParsedResume) -> bool:
        """
        Add resume to the index.
        Returns True if resume is a duplicate of an existing one.
        """
        key = str(resume.resume_id)
        minhash = self._compute_minhash(resume.raw_text)

        try:
            duplicates = self._lsh.query(minhash)
            if duplicates:
                logger.info(
                    f"Duplicate detected: {resume.resume_id} ≈ {duplicates}"
                )
                resume.is_duplicate = True
                resume.duplicate_of = UUID(duplicates[0])
                return True

            self._lsh.insert(key, minhash)
            self._registered[resume.resume_id] = minhash
            return False
        except Exception as e:
            logger.error(f"LSH error: {e}")
            return False

    def bulk_check(self, resumes: list[ParsedResume]) -> list[ParsedResume]:
        """Check all resumes for duplicates and mark them."""
        for resume in resumes:
            self.add_resume(resume)
        return resumes

    def get_similarity(self, resume_a: ParsedResume, resume_b: ParsedResume) -> float:
        """Compute approximate Jaccard similarity between two resumes."""
        mh_a = self._compute_minhash(resume_a.raw_text)
        mh_b = self._compute_minhash(resume_b.raw_text)
        return mh_a.jaccard(mh_b)

    def _compute_minhash(self, text: str) -> MinHash:
        """Generate MinHash signature from character 3-grams of text."""
        minhash = MinHash(num_perm=settings.LSH_NUM_PERM)
        text_lower = text.lower()
        # Character 3-grams (shingles)
        shingles = {text_lower[i:i+3] for i in range(len(text_lower) - 2)}
        for shingle in shingles:
            minhash.update(shingle.encode("utf-8"))
        return minhash

    def clear(self):
        self._lsh = MinHashLSH(
            threshold=settings.LSH_THRESHOLD,
            num_perm=settings.LSH_NUM_PERM,
        )
        self._registered.clear()


# ─────────────────────────────────────────────────────────────────────────────
# backend/utils/keyword_stuffing.py
# ─────────────────────────────────────────────────────────────────────────────
"""
Keyword stuffing detection using statistical anomaly detection.

Method: Compare keyword density of resume against a reference corpus.
A resume is flagged if its keyword density z-score exceeds a threshold.

Also detects: hidden text, tiny fonts, white-on-white text (via PDF metadata).
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from typing import Optional

import fitz  # PyMuPDF
from loguru import logger


# Reference keyword density statistics (from a corpus of ~10k normal resumes)
# These should ideally be computed from your actual corpus
REFERENCE_MEAN_KEYWORD_DENSITY = 0.08   # 8% keywords on average
REFERENCE_STD_KEYWORD_DENSITY = 0.04    # std dev

COMMON_TECH_KEYWORDS = {
    "python", "java", "javascript", "sql", "aws", "docker", "kubernetes",
    "machine learning", "deep learning", "tensorflow", "pytorch", "react",
    "angular", "node", "mongodb", "postgresql", "redis", "kafka", "spark",
    "git", "ci/cd", "agile", "scrum", "rest", "api", "microservices",
}


class KeywordStuffingDetector:
    """
    Detects keyword stuffing in resumes.

    Returns a score in [0, 1] where:
      0.0 = definitely clean
      1.0 = almost certainly stuffed
    """

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def score(self, raw_text: str, pdf_bytes: Optional[bytes] = None) -> float:
        """
        Compute stuffing score. Combines:
          1. Keyword density anomaly (z-score based)
          2. Repetition score (same phrase appearing many times)
          3. Hidden text detection (if PDF provided)
        """
        scores = []

        density_score = self._keyword_density_score(raw_text)
        scores.append(density_score)

        repetition_score = self._repetition_score(raw_text)
        scores.append(repetition_score)

        if pdf_bytes:
            hidden_score = self._hidden_text_score(pdf_bytes)
            scores.append(hidden_score * 2)  # weight hidden text higher

        # Weighted average
        combined = sum(scores) / len(scores) if scores else 0.0
        return round(min(1.0, combined), 4)

    def is_stuffed(self, raw_text: str, pdf_bytes: Optional[bytes] = None) -> bool:
        return self.score(raw_text, pdf_bytes) >= self.threshold

    # ── Detection Methods ──────────────────────────────────────────────────────

    def _keyword_density_score(self, text: str) -> float:
        """
        Compute z-score of keyword density vs reference corpus.
        High z-score → anomalous keyword density → possible stuffing.
        """
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return 0.0

        total_words = len(words)
        keyword_count = sum(1 for w in words if w in COMMON_TECH_KEYWORDS)

        # Check bigrams too
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if bigram in COMMON_TECH_KEYWORDS:
                keyword_count += 1

        density = keyword_count / total_words
        z_score = (density - REFERENCE_MEAN_KEYWORD_DENSITY) / REFERENCE_STD_KEYWORD_DENSITY

        # Sigmoid transform to [0, 1]
        import math
        return round(1 / (1 + math.exp(-0.5 * (z_score - 2))), 4)

    def _repetition_score(self, text: str) -> float:
        """
        Detect unusual repetition of the same skill/keyword.
        Legitimate resumes don't repeat the same keyword >3-4 times.
        """
        words = re.findall(r"\b\w{4,}\b", text.lower())
        if not words:
            return 0.0

        counter = Counter(words)
        tech_counts = {
            word: count
            for word, count in counter.items()
            if word in COMMON_TECH_KEYWORDS and count > 3
        }

        if not tech_counts:
            return 0.0

        # Max repetition of any keyword
        max_rep = max(tech_counts.values())
        # Score: 0 for 3 reps, 1 for 15+ reps
        score = min(1.0, (max_rep - 3) / 12)
        return round(score, 4)

    def _hidden_text_score(self, pdf_bytes: bytes) -> float:
        """
        Detect white/invisible text in PDF (common stuffing technique).
        Checks for text with very small font size or white color.
        """
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_chars = 0
            suspicious_chars = 0

            for page in doc:
                for block in page.get_text("rawdict")["blocks"]:
                    if block.get("type") != 0:  # text blocks only
                        continue
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "")
                            size = span.get("size", 12)
                            color = span.get("color", 0)  # 0 = black

                            total_chars += len(text)

                            # Suspicious: tiny font (<4pt) or white text (0xFFFFFF)
                            if size < 4 or color == 0xFFFFFF:
                                suspicious_chars += len(text)

            doc.close()
            if total_chars == 0:
                return 0.0
            return round(suspicious_chars / total_chars, 4)

        except Exception as e:
            logger.debug(f"PDF analysis error: {e}")
            return 0.0
