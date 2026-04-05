"""
backend/services/nlp/embedder.py
──────────────────────────────────
Embedding generation using SentenceTransformers.
Includes FAISS index for fast similarity search + LRU caching.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Optional
from uuid import UUID

import faiss
import numpy as np
from cachetools import LRUCache
from loguru import logger
from sentence_transformers import SentenceTransformer

from backend.core.config import get_settings
from backend.models.resume import ParsedResume
from backend.models.ranking import ParsedJobDescription

settings = get_settings()

# Cache embeddings to avoid re-computing same texts
_embedding_cache: LRUCache = LRUCache(maxsize=1000)


class EmbeddingService:
    """
    Wraps SentenceTransformer for:
      - Resume embedding generation
      - JD embedding generation
      - Semantic similarity scoring
      - FAISS-based nearest-neighbor search
    """

    def __init__(self, model_name: Optional[str] = None):
        model_name = model_name or settings.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self._faiss_index: Optional[faiss.IndexFlatIP] = None
        self._index_id_map: list[UUID] = []   # position -> resume_id

    # ── Core Embedding ─────────────────────────────────────────────────────────

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string with caching."""
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in _embedding_cache:
            return _embedding_cache[cache_key]

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,  # L2 normalize → cosine sim = dot product
            show_progress_bar=False,
        )
        _embedding_cache[cache_key] = embedding
        return embedding

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Batch embedding with progress bar for large sets."""
        return self.model.encode(
            texts,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 10,
        )

    # ── Resume / JD Text Builders ──────────────────────────────────────────────

    def build_resume_text(self, resume: ParsedResume) -> str:
        """
        Construct a rich text representation for embedding.
        Weights important sections by repetition.
        """
        parts = []

        if resume.candidate_name:
            parts.append(resume.candidate_name)

        # Skills — most important, repeat for weight
        if resume.skills:
            skill_str = " ".join(resume.skills)
            parts.extend([skill_str, skill_str])  # repeat 2x for emphasis

        # Experience titles
        for exp in resume.experience[:5]:
            parts.append(f"{exp.role} at {exp.company}")
            if exp.technologies:
                parts.append(" ".join(exp.technologies))
            if exp.description:
                parts.append(exp.description[:200])

        # Education
        for edu in resume.education:
            parts.append(f"{edu.degree} {edu.field_of_study} {edu.institution}")

        # Projects
        for proj in resume.projects[:3]:
            parts.append(f"{proj.name} {proj.description[:150]}")
            if proj.technologies:
                parts.append(" ".join(proj.technologies))

        # Summary
        if resume.summary:
            parts.append(resume.summary[:300])

        return " ".join(p for p in parts if p)

    def build_jd_text(self, jd: ParsedJobDescription) -> str:
        """
        Construct JD embedding text.
        Uses pre-built embedding_text if available.
        """
        if jd.embedding_text:
            return jd.embedding_text

        parts = [jd.title]
        parts.extend(s.name for s in jd.required_skills)
        parts.extend(jd.preferred_skills)
        if jd.education_requirement:
            parts.append(jd.education_requirement)
        parts.extend(jd.responsibilities[:5])
        return " ".join(parts)

    # ── Similarity Computation ─────────────────────────────────────────────────

    def cosine_similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """
        Cosine similarity between two normalized embeddings.
        Since we L2-normalize at encode time, this is just dot product.
        Returns float in [-1, 1], clipped to [0, 1].
        """
        sim = float(np.dot(emb_a, emb_b))
        return max(0.0, min(1.0, sim))

    def score_resume_against_jd(
        self,
        resume: ParsedResume,
        jd: ParsedJobDescription,
    ) -> float:
        """End-to-end: build texts → embed → cosine sim."""
        resume_text = self.build_resume_text(resume)
        jd_text = self.build_jd_text(jd)
        resume_emb = self.embed(resume_text)
        jd_emb = self.embed(jd_text)
        return self.cosine_similarity(resume_emb, jd_emb)

    # ── FAISS Index (for bulk candidate search) ────────────────────────────────

    def build_index(self, resumes: list[ParsedResume]) -> None:
        """Build FAISS inner-product index from all resume embeddings."""
        texts = [self.build_resume_text(r) for r in resumes]
        embeddings = self.embed_batch(texts).astype(np.float32)

        self._faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self._faiss_index.add(embeddings)
        self._index_id_map = [r.resume_id for r in resumes]
        logger.info(f"FAISS index built with {len(resumes)} candidates")

    def search_top_k(
        self,
        jd: ParsedJobDescription,
        k: int = 20,
    ) -> list[tuple[UUID, float]]:
        """
        Return top-k (resume_id, similarity_score) pairs for a JD.
        Fast O(n) approximate search via FAISS.
        """
        if self._faiss_index is None:
            raise RuntimeError("FAISS index not built. Call build_index() first.")

        jd_text = self.build_jd_text(jd)
        jd_emb = self.embed(jd_text).astype(np.float32).reshape(1, -1)

        k = min(k, self._faiss_index.ntotal)
        scores, indices = self._faiss_index.search(jd_emb, k)

        return [
            (self._index_id_map[idx], float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx >= 0
        ]

    def save_index(self, path: str) -> None:
        faiss.write_index(self._faiss_index, path)
        with open(f"{path}.map.pkl", "wb") as f:
            pickle.dump(self._index_id_map, f)

    def load_index(self, path: str) -> None:
        self._faiss_index = faiss.read_index(path)
        with open(f"{path}.map.pkl", "rb") as f:
            self._index_id_map = pickle.load(f)
