"""
backend/api/routes/jobs.py
───────────────────────────
Job description management endpoints.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.models.ranking import ParsedJobDescription
from backend.services.parser.jd_parser import JDParser

router = APIRouter()
_jd_store: dict[UUID, ParsedJobDescription] = {}
_jd_parser = JDParser()


class JDCreateRequest(BaseModel):
    raw_text: str
    title_override: str = ""


@router.post("/", response_model=ParsedJobDescription, status_code=201)
async def create_job(req: JDCreateRequest):
    """Parse and store a job description."""
    if len(req.raw_text.strip()) < 50:
        raise HTTPException(400, "JD text too short")

    jd = _jd_parser.parse(req.raw_text)
    if req.title_override:
        jd.title = req.title_override
    _jd_store[jd.jd_id] = jd
    return jd


@router.get("/{jd_id}", response_model=ParsedJobDescription)
async def get_job(jd_id: UUID):
    jd = _jd_store.get(jd_id)
    if not jd:
        raise HTTPException(404, "JD not found")
    return jd


@router.get("/", response_model=list[ParsedJobDescription])
async def list_jobs():
    return list(_jd_store.values())


@router.delete("/{jd_id}", status_code=204)
async def delete_job(jd_id: UUID):
    if jd_id not in _jd_store:
        raise HTTPException(404, "JD not found")
    del _jd_store[jd_id]


def get_jd_store() -> dict[UUID, ParsedJobDescription]:
    return _jd_store


# ─────────────────────────────────────────────────────────────────────────────
# backend/api/routes/ranking.py
# ─────────────────────────────────────────────────────────────────────────────
"""
Ranking endpoint — the core of the system.
"""

from __future__ import annotations

from fastapi import APIRouter as _RankRouter, HTTPException as _HTTPException, Request

from backend.models.ranking import (
    RankedCandidate,
    RankingRequest,
    RankingResponse,
    SkillGapResult,
)
from backend.services.ml.feature_eng import FeatureEngineer
from backend.services.ml.ranker import CandidateRanker
from backend.services.nlp.embedder import EmbeddingService
from backend.services.explainer.shap_explainer import SHAPExplainer
from backend.services.explainer.skill_gap import SkillGapAnalyzer
from backend.api.routes.resume import get_resume_store
from backend.api.routes.jobs import get_jd_store

ranking_router = _RankRouter()

_ranker = CandidateRanker()
_skill_gap_analyzer = SkillGapAnalyzer()


@ranking_router.post("/rank", response_model=RankingResponse)
async def rank_candidates(req: RankingRequest, request: Request):
    """
    Core ranking endpoint.
    Given a JD ID, rank all uploaded resumes (or specified subset).
    Returns sorted list with scores, explanations, and skill gap analysis.
    """
    resume_store = get_resume_store()
    jd_store = get_jd_store()
    embedder: EmbeddingService = request.app.state.embedder

    # Validate JD
    jd = jd_store.get(req.jd_id)
    if not jd:
        raise _HTTPException(404, f"JD {req.jd_id} not found")

    # Select resumes
    if req.resume_ids:
        resumes = [resume_store[rid] for rid in req.resume_ids if rid in resume_store]
    else:
        resumes = list(resume_store.values())

    if not resumes:
        raise _HTTPException(400, "No resumes available for ranking")

    # 1. Compute semantic similarity scores
    jd_embedding = embedder.embed(embedder.build_jd_text(jd))
    semantic_scores = []
    for resume in resumes:
        resume_text = embedder.build_resume_text(resume)
        resume_emb = embedder.embed(resume_text)
        sim = embedder.cosine_similarity(resume_emb, jd_embedding)
        semantic_scores.append(sim)

    # 2. Feature engineering
    fe = FeatureEngineer(embedder.model)
    feature_vecs = fe.compute_batch(resumes, jd, semantic_scores)

    # 3. ML ranking
    resume_metadata = {
        r.resume_id: {"name": r.candidate_name, "email": r.email}
        for r in resumes
    }
    raw_ranked = _ranker.rank(feature_vecs, resume_metadata, jd.title)

    # 4. SHAP explanations
    shap_explainer = SHAPExplainer()
    if _ranker._is_trained and _ranker.model:
        shap_explainer.init_from_booster(_ranker.model.get_booster())
    X_scaled = _ranker._to_matrix(feature_vecs)

    fv_by_id = {fv.resume_id: (fv, X_scaled[i]) for i, fv in enumerate(feature_vecs)}

    # 5. Build ranked candidates
    ranked_candidates = []
    top_k = req.top_k

    for item in raw_ranked:
        rid = item["resume_id"]
        fv, x_row = fv_by_id[rid]
        resume = resume_store[rid]

        # Skill gap
        skill_gap = _skill_gap_analyzer.analyze(resume, jd, fe._compute_skill_gap(resume, jd))

        # SHAP explanation
        explanation = shap_explainer.explain(fv, x_row.reshape(1, -1))

        score = item["final_score"]
        recommendation = _ranker.get_recommendation(score, fv.mandatory_skill_coverage)

        if not req.include_rejected and recommendation == "Reject":
            continue

        ranked_candidates.append(
            RankedCandidate(
                rank=len(ranked_candidates) + 1,
                resume_id=rid,
                candidate_name=item["meta"].get("name", "Unknown"),
                email=item["meta"].get("email"),
                final_score=score,
                semantic_score=round(fv.semantic_similarity * 100, 1),
                ml_score=round(item["ml_score"] * 100, 1),
                skill_match_score=round(fv.skill_match_ratio * 100, 1),
                experience_score=round(fv.experience_match_score * 100, 1),
                skill_gap=skill_gap.skill_gap,
                shap_explanation=explanation["shap_values"],
                top_positive_factors=explanation["top_positive_factors"],
                top_negative_factors=explanation["top_negative_factors"],
                is_duplicate=resume.is_duplicate,
                has_keyword_stuffing=resume.has_keyword_stuffing,
                recommendation=recommendation,
            )
        )

        if len(ranked_candidates) >= top_k:
            break

    return RankingResponse(
        jd_id=req.jd_id,
        jd_title=jd.title,
        total_candidates=len(resumes),
        ranked_candidates=ranked_candidates,
        ranking_metadata={
            "model": "XGBoost LambdaRank",
            "weights": {
                "semantic": settings.WEIGHT_SEMANTIC,
                "ml_score": settings.WEIGHT_ML_SCORE,
                "skill_match": settings.WEIGHT_SKILL_MATCH,
                "experience": settings.WEIGHT_EXPERIENCE,
            },
            "embedding_model": settings.EMBEDDING_MODEL,
        },
    )


from backend.core.config import get_settings
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# backend/api/routes/analytics.py
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import APIRouter as _AnalyticsRouter
from backend.api.routes.resume import get_resume_store as _grs

analytics_router = _AnalyticsRouter()


@analytics_router.get("/dashboard")
async def dashboard_stats():
    """Aggregate statistics for the analytics dashboard."""
    store = _grs()
    resumes = list(store.values())

    total = len(resumes)
    duplicates = sum(1 for r in resumes if r.is_duplicate)
    stuffed = sum(1 for r in resumes if r.has_keyword_stuffing)

    skill_counter: dict[str, int] = {}
    for r in resumes:
        for s in r.skills:
            skill_counter[s] = skill_counter.get(s, 0) + 1

    top_skills = sorted(skill_counter.items(), key=lambda x: x[1], reverse=True)[:20]

    avg_exp = (
        sum(r.total_experience_months for r in resumes) / max(total, 1) / 12
    )

    return {
        "total_resumes": total,
        "duplicate_count": duplicates,
        "keyword_stuffing_count": stuffed,
        "avg_experience_years": round(avg_exp, 1),
        "top_skills": [{"skill": s, "count": c} for s, c in top_skills],
        "education_distribution": _education_dist(resumes),
    }


def _education_dist(resumes) -> dict:
    from collections import Counter
    counter = Counter(r.highest_education_level.value for r in resumes)
    return dict(counter)
