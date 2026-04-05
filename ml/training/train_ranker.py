"""
ml/training/train_ranker.py
End-to-end XGBoost LambdaRank training pipeline.
"""
from __future__ import annotations
import json, os, pickle, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml.training.synthetic_data import generate_dataset
from ml.evaluation.metrics import ndcg_at_k, precision_at_k, mean_reciprocal_rank

FEATURE_COLS = [
    "semantic_similarity","skill_match_ratio","mandatory_skill_coverage",
    "preferred_skill_coverage","experience_years","experience_match_score",
    "role_title_similarity","education_level_score","education_field_match",
    "keyword_stuffing_score","is_duplicate","project_count",
    "has_relevant_projects","certification_count",
]

XGB_PARAMS = dict(
    objective="rank:ndcg", eval_metric=["ndcg@5","ndcg@10"],
    learning_rate=0.05, max_depth=6, min_child_weight=5,
    subsample=0.8, colsample_bytree=0.8, n_estimators=500,
    early_stopping_rounds=40, tree_method="hist", seed=42,
)

OUT = Path("ml/artifacts"); OUT.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    p = "data/synthetic/ltr_dataset.csv"
    if os.path.exists(p):
        logger.info(f"Loading {p}")
        return pd.read_csv(p)
    logger.info("Generating synthetic dataset …")
    return generate_dataset(n_jds=150, candidates_per_jd=50)


def split(df: pd.DataFrame):
    spl = GroupShuffleSplit(1, test_size=0.2, random_state=42)
    tr, va = next(spl.split(df, groups=df["query_id"]))
    return df.iloc[tr].copy(), df.iloc[va].copy()


def matrices(tr, va, scaler=None):
    Xt = tr[FEATURE_COLS].values.astype(np.float32)
    yt = tr["relevance_label"].values.astype(np.int32)
    gt = tr.groupby("query_id").size().values
    Xv = va[FEATURE_COLS].values.astype(np.float32)
    yv = va["relevance_label"].values.astype(np.int32)
    gv = va.groupby("query_id").size().values
    if scaler is None:
        scaler = StandardScaler(); Xt = scaler.fit_transform(Xt)
    else:
        Xt = scaler.transform(Xt)
    return Xt, yt, gt, scaler.transform(Xv), yv, gv, scaler


def train(Xt, yt, gt, Xv, yv, gv):
    m = xgb.XGBRanker(**XGB_PARAMS)
    m.fit(Xt, yt, group=gt, eval_set=[(Xv, yv)], verbose=50)
    logger.info(f"Best iteration: {m.best_iteration}")
    return m


def evaluate(m, Xv, val_df):
    scores = m.predict(Xv)
    val_df = val_df.copy(); val_df["pred"] = scores
    res = []
    for _, g in val_df.groupby("query_id"):
        g = g.sort_values("pred", ascending=False)
        tl = g["relevance_label"].tolist()
        il = sorted(tl, reverse=True)
        res.append({"ndcg@5": ndcg_at_k(tl,il,5), "ndcg@10": ndcg_at_k(tl,il,10),
                    "p@5": precision_at_k(tl,5,2), "mrr": mean_reciprocal_rank(tl,2)})
    metrics = {k: round(float(np.mean([r[k] for r in res])),4) for k in res[0]}
    logger.info("──── Evaluation ────")
    for k,v in metrics.items(): logger.info(f"  {k}: {v:.4f}")
    return metrics


def plot_importance(m):
    imp = {FEATURE_COLS[i]: v for i,v in enumerate(m.feature_importances_)}
    items = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    labels, vals = zip(*items)
    fig, ax = plt.subplots(figsize=(10, 7))
    median = float(np.median(vals))
    colors = ["#2ecc71" if v > median else "#3498db" for v in vals]
    ax.barh(range(len(labels)), vals, color=colors, alpha=0.85)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis(); ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("XGBoost LambdaRank — Feature Importance", fontweight="bold")
    ax.axvline(median, color="gray", linestyle="--", alpha=0.5, label="Median")
    ax.legend(); plt.tight_layout()
    fig.savefig(str(OUT / "feature_importance.png"), dpi=150); plt.close()
    logger.info("Feature importance → ml/artifacts/feature_importance.png")


def main():
    df = load_data()
    tr, va = split(df)
    logger.info(f"Train: {len(tr)} | Val: {len(va)}")
    Xt, yt, gt, Xv, yv, gv, scaler = matrices(tr, va)
    m = train(Xt, yt, gt, Xv, yv, gv)
    metrics = evaluate(m, Xv, va)
    m.save_model(str(OUT / "xgb_ranker.json"))
    with open(OUT / "feature_scaler.pkl", "wb") as f: pickle.dump(scaler, f)
    with open(OUT / "eval_metrics.json", "w") as f: json.dump(metrics, f, indent=2)
    plot_importance(m)
    logger.success("All artifacts saved to ml/artifacts/")
    return metrics


if __name__ == "__main__":
    main()
