"""
backend/services/explainer/shap_explainer.py
──────────────────────────────────────────────
SHAP-based explainability for the XGBoost ranking model.

Provides:
  - Per-candidate feature contribution breakdown
  - Human-readable positive/negative factor summaries
  - Global feature importance (across all candidates)
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

import numpy as np
import shap
import xgboost as xgb
from loguru import logger

from backend.services.ml.feature_eng import FeatureEngineer
from backend.models.ranking import FeatureVector

# Human-readable labels for each feature
FEATURE_LABELS = {
    "semantic_similarity": "Overall profile-JD alignment",
    "skill_match_ratio": "Skill match percentage",
    "mandatory_skill_coverage": "Coverage of mandatory skills",
    "preferred_skill_coverage": "Coverage of preferred skills",
    "experience_years": "Years of experience",
    "experience_match_score": "Experience level alignment",
    "role_title_similarity": "Job title relevance",
    "education_level_score": "Education qualification",
    "education_field_match": "Education field relevance",
    "keyword_stuffing_score": "Keyword stuffing risk",
    "is_duplicate": "Duplicate resume flag",
    "project_count": "Number of relevant projects",
    "has_relevant_projects": "Project technology relevance",
    "certification_count": "Professional certifications",
}

NEGATIVE_FEATURES = {"keyword_stuffing_score", "is_duplicate"}  # higher = worse


class SHAPExplainer:
    """
    Wraps SHAP TreeExplainer for XGBoost models.

    Usage:
        explainer = SHAPExplainer(ranker.model.get_booster())
        explanation = explainer.explain(feature_vec)
    """

    def __init__(self, booster: Optional[xgb.Booster] = None):
        self._explainer: Optional[shap.TreeExplainer] = None
        if booster is not None:
            self.init_from_booster(booster)

    def init_from_booster(self, booster: xgb.Booster):
        self._explainer = shap.TreeExplainer(booster)
        logger.info("SHAP TreeExplainer initialized")

    # ── Per-Candidate Explanation ──────────────────────────────────────────────

    def explain(
        self,
        feature_vec: FeatureVector,
        X_scaled: np.ndarray,    # the scaled feature row (1, n_features)
    ) -> dict:
        """
        Returns a dict with:
          - shap_values: feature -> shap_value
          - top_positive_factors: list of human-readable positive contributors
          - top_negative_factors: list of human-readable negative contributors
          - base_value: expected model output
        """
        if self._explainer is None:
            return self._fallback_explanation(feature_vec)

        shap_values = self._explainer.shap_values(X_scaled)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        values = shap_values[0] if shap_values.ndim > 1 else shap_values
        feature_names = FeatureEngineer.FEATURE_NAMES

        shap_dict = {
            feat: round(float(val), 5)
            for feat, val in zip(feature_names, values)
        }

        pos_factors, neg_factors = self._humanize(shap_dict, feature_vec)

        return {
            "shap_values": shap_dict,
            "top_positive_factors": pos_factors[:3],
            "top_negative_factors": neg_factors[:3],
            "base_value": round(float(self._explainer.expected_value), 5),
        }

    def explain_batch(
        self,
        feature_vecs: list[FeatureVector],
        X_scaled: np.ndarray,
    ) -> list[dict]:
        """Efficiently explain a batch of candidates."""
        if self._explainer is None:
            return [self._fallback_explanation(fv) for fv in feature_vecs]

        shap_values = self._explainer.shap_values(X_scaled)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        results = []
        for i, fv in enumerate(feature_vecs):
            values = shap_values[i]
            shap_dict = {
                feat: round(float(val), 5)
                for feat, val in zip(FeatureEngineer.FEATURE_NAMES, values)
            }
            pos_factors, neg_factors = self._humanize(shap_dict, fv)
            results.append({
                "shap_values": shap_dict,
                "top_positive_factors": pos_factors[:3],
                "top_negative_factors": neg_factors[:3],
                "base_value": round(float(self._explainer.expected_value), 5),
            })
        return results

    # ── Global Feature Importance ──────────────────────────────────────────────

    def global_importance(self, X_scaled: np.ndarray) -> dict[str, float]:
        """
        Mean |SHAP value| across all samples — global feature importance.
        Useful for analytics dashboard.
        """
        if self._explainer is None:
            return {}
        shap_values = self._explainer.shap_values(X_scaled)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        mean_abs = np.abs(shap_values).mean(axis=0)
        return {
            feat: round(float(val), 5)
            for feat, val in zip(FeatureEngineer.FEATURE_NAMES, mean_abs)
        }

    # ── Human-Readable Conversion ──────────────────────────────────────────────

    def _humanize(
        self,
        shap_dict: dict[str, float],
        fv: FeatureVector,
    ) -> tuple[list[str], list[str]]:
        """
        Convert SHAP values to human-readable explanations.

        For positive features: high SHAP → positive factor
        For negative features: high SHAP with high actual value → negative factor
        """
        positive_factors = []
        negative_factors = []

        # Sort by absolute SHAP value (most impactful first)
        sorted_feats = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)

        for feat, shap_val in sorted_feats:
            label = FEATURE_LABELS.get(feat, feat)
            actual_val = getattr(fv, feat, None)

            if feat in NEGATIVE_FEATURES:
                # Negative feature: high actual value = bad signal
                if shap_val < -0.01 or (actual_val and actual_val > 0.3):
                    negative_factors.append(
                        f"⚠️ {label} detected (score: {actual_val:.2f})"
                    )
            else:
                if shap_val > 0.01:
                    msg = self._positive_message(feat, label, actual_val)
                    positive_factors.append(msg)
                elif shap_val < -0.01:
                    msg = self._negative_message(feat, label, actual_val)
                    negative_factors.append(msg)

        return positive_factors, negative_factors

    def _positive_message(self, feat: str, label: str, val) -> str:
        if feat == "semantic_similarity":
            return f"✅ Strong profile-JD alignment ({val:.0%})"
        if feat == "skill_match_ratio":
            return f"✅ Good skill coverage ({val:.0%} skills matched)"
        if feat == "mandatory_skill_coverage":
            return f"✅ All mandatory skills covered ({val:.0%})"
        if feat == "experience_years":
            return f"✅ Relevant experience ({val:.1f} years)"
        if feat == "certification_count":
            return f"✅ {int(val)} professional certifications"
        if feat == "project_count":
            return f"✅ {int(val)} projects demonstrating hands-on experience"
        return f"✅ {label} is a positive signal"

    def _negative_message(self, feat: str, label: str, val) -> str:
        if feat == "semantic_similarity":
            return f"❌ Low profile-JD alignment ({val:.0%})"
        if feat == "skill_match_ratio":
            return f"❌ Skill gap: only {val:.0%} skills matched"
        if feat == "mandatory_skill_coverage":
            return f"❌ Missing mandatory skills ({val:.0%} covered)"
        if feat == "experience_years":
            return f"❌ Insufficient experience ({val:.1f} years)"
        if feat == "education_field_match":
            return "❌ Education field does not match JD requirements"
        return f"❌ {label} is below threshold"

    def _fallback_explanation(self, fv: FeatureVector) -> dict:
        """Rule-based explanation when SHAP is unavailable."""
        pos, neg = [], []

        if fv.semantic_similarity > 0.7:
            pos.append(f"✅ Strong profile-JD alignment ({fv.semantic_similarity:.0%})")
        else:
            neg.append(f"❌ Low profile-JD alignment ({fv.semantic_similarity:.0%})")

        if fv.mandatory_skill_coverage >= 0.8:
            pos.append(f"✅ Most mandatory skills covered ({fv.mandatory_skill_coverage:.0%})")
        elif fv.mandatory_skill_coverage < 0.5:
            neg.append(f"❌ Missing mandatory skills ({fv.mandatory_skill_coverage:.0%} covered)")

        if fv.experience_match_score >= 0.8:
            pos.append(f"✅ Experience level matches requirements")
        elif fv.experience_years < 1:
            neg.append("❌ Limited professional experience")

        if fv.keyword_stuffing_score > 0.5:
            neg.append(f"⚠️ Possible keyword stuffing detected")

        return {
            "shap_values": {},
            "top_positive_factors": pos[:3],
            "top_negative_factors": neg[:3],
            "base_value": 0.0,
        }
