"""
backend/services/ml/ranker.py
──────────────────────────────
XGBoost-based Learning-to-Rank model.

Uses XGBoost's rank:ndcg objective (LambdaRank algorithm).
Final score = weighted combination of:
  - Semantic similarity (40%)
  - ML rank score (35%)
  - Skill match ratio (15%)
  - Experience score (10%)
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional
from uuid import UUID

import numpy as np
import xgboost as xgb
from loguru import logger
from sklearn.preprocessing import StandardScaler

from backend.core.config import get_settings
from backend.models.ranking import FeatureVector, RankedCandidate
from backend.services.ml.feature_eng import FeatureEngineer

settings = get_settings()


class CandidateRanker:
    """
    XGBoost LambdaRank model for candidate ranking.

    Training mode:  CandidateRanker.train(features, labels, groups)
    Inference mode: ranker.rank(feature_vectors) → list[RankedCandidate]
    """

    MODEL_PARAMS = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@10",
        "learning_rate": 0.1,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 100,
        "early_stopping_rounds": 30,
        "tree_method": "hist",
        "device": "cpu",
        "seed": 42,
    }

    def __init__(self):
        self.model: Optional[xgb.XGBRanker] = None
        self.scaler: Optional[StandardScaler] = None
        self._is_trained = False
        self._load_if_exists()

    # ── Public API ─────────────────────────────────────────────────────────────

    def rank(
        self,
        feature_vecs: list[FeatureVector],
        resume_metadata: dict[UUID, dict],  # resume_id -> {name, email, ...}
        jd_title: str = "",
    ) -> list[RankedCandidate]:
        """
        Rank candidates given feature vectors.
        Returns sorted RankedCandidate list (best first).
        """
        if not feature_vecs:
            return []

        X = self._to_matrix(feature_vecs)
        ml_scores = self._predict_scores(X)

        # Compute final weighted scores
        results = []
        for fv, ml_score in zip(feature_vecs, ml_scores):
            final_score = self._weighted_score(fv, ml_score)
            meta = resume_metadata.get(fv.resume_id, {})

            results.append({
                "resume_id": fv.resume_id,
                "final_score": final_score,
                "semantic_score": fv.semantic_similarity,
                "ml_score": ml_score,
                "skill_match_score": fv.skill_match_ratio,
                "experience_score": fv.experience_match_score,
                "feature_vec": fv,
                "meta": meta,
            })

        # Sort by final score descending
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Raw model predictions for SHAP computation."""
        if not self._is_trained:
            return np.zeros(len(X))
        dmatrix = xgb.DMatrix(X, feature_names=FeatureEngineer.FEATURE_NAMES)
        return self.model.get_booster().predict(dmatrix)

    # ── Scoring Formula ────────────────────────────────────────────────────────

    def _weighted_score(self, fv: FeatureVector, ml_score: float) -> float:
        """
        Final Score = 0.40 × semantic_similarity
                    + 0.35 × ml_score (normalized 0-1)
                    + 0.15 × skill_match_ratio
                    + 0.10 × experience_match_score

        Penalty multipliers:
          - keyword_stuffing: × 0.7
          - is_duplicate:     × 0.0 (auto-disqualified)
        """
        w = settings
        raw = (
            w.WEIGHT_SEMANTIC       * fv.semantic_similarity
            + w.WEIGHT_ML_SCORE     * ml_score
            + w.WEIGHT_SKILL_MATCH  * fv.skill_match_ratio
            + w.WEIGHT_EXPERIENCE   * fv.experience_match_score
        )

        # Apply penalty multipliers
        if fv.is_duplicate:
            raw *= 0.0
        elif fv.keyword_stuffing_score > 0.7:
            raw *= 0.7

        # Scale to 0-100
        return round(min(100.0, raw * 100), 2)

    # ── Recommendation Threshold ───────────────────────────────────────────────

    @staticmethod
    def get_recommendation(score: float, mandatory_coverage: float) -> str:
        """
        Business rule layer on top of ML score.
        Missing mandatory skills hard-rejects regardless of score.
        """
        if mandatory_coverage < 0.5:
            return "Reject"
        if score >= 72:
            return "Strong Hire"
        if score >= 48:
            return "Consider"
        return "Reject"

    # ── Training ───────────────────────────────────────────────────────────────

    @classmethod
    def train(
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,        # relevance labels (0=irrelevant, 1=relevant, 2=ideal)
        groups_train: np.ndarray,   # number of docs per query
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        groups_val: Optional[np.ndarray] = None,
    ) -> "CandidateRanker":
        """Train XGBoost LambdaRank. Returns fitted ranker."""
        instance = cls.__new__(cls)
        instance.scaler = StandardScaler()
        X_train_s = instance.scaler.fit_transform(X_train)

        ranker = xgb.XGBRanker(**cls.MODEL_PARAMS)

        eval_set = None
        if X_val is not None:
            X_val_s = instance.scaler.transform(X_val)
            eval_set = [(X_val_s, y_val)]

        eval_group = [groups_val] if groups_val is not None else None
        ranker.fit(
            X_train_s,
            y_train,
            group=groups_train,
            eval_set=eval_set,
            eval_group=eval_group,
            qid=None,
            verbose=50,
        )

        instance.model = ranker
        instance._is_trained = True
        instance._save()
        logger.info("XGBoost LambdaRank training complete")
        return instance

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save(self):
        Path(settings.MODEL_DIR).mkdir(parents=True, exist_ok=True)
        self.model.save_model(settings.RANKER_MODEL_PATH)
        with open(settings.SCALER_PATH, "wb") as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Model saved to {settings.RANKER_MODEL_PATH}")

    def _load_if_exists(self):
        if (
            os.path.exists(settings.RANKER_MODEL_PATH)
            and os.path.exists(settings.SCALER_PATH)
        ):
            self.model = xgb.XGBRanker()
            self.model.load_model(settings.RANKER_MODEL_PATH)
            with open(settings.SCALER_PATH, "rb") as f:
                self.scaler = pickle.load(f)
            self._is_trained = True
            logger.info("Pre-trained XGBoost model loaded")
        else:
            logger.warning(
                "No pre-trained model found. Run ml/training/train_ranker.py first."
            )

    # ── Internal Helpers ───────────────────────────────────────────────────────

    def _to_matrix(self, feature_vecs: list[FeatureVector]) -> np.ndarray:
        """Convert FeatureVectors to numpy matrix."""
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
        X = np.array(rows, dtype=np.float32)
        if self.scaler:
            X = self.scaler.transform(X)
        return X

    def _predict_scores(self, X: np.ndarray) -> list[float]:
        if not self._is_trained:
            logger.warning("Model not trained. Returning uniform scores.")
            return [0.5] * len(X)

        raw_scores = self.model.predict(X)
        # Normalize to [0, 1] using min-max
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s == min_s:
            return [0.5] * len(X)
        return ((raw_scores - min_s) / (max_s - min_s)).tolist()
