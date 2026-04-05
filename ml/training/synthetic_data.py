"""
ml/training/synthetic_data.py
──────────────────────────────
Generates synthetic (resume, JD, relevance_label) training triplets
for XGBoost LambdaRank (rank:ndcg objective).

Label scale: 0=irrelevant, 1=partial, 2=good, 3=ideal
"""

from __future__ import annotations
import random, uuid
from dataclasses import dataclass
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

ML_SKILLS = ["Python","TensorFlow","PyTorch","scikit-learn","XGBoost","LightGBM",
    "NLP","Computer Vision","BERT","Transformers","Hugging Face","MLOps",
    "Kubeflow","MLflow","Airflow","Spark","Kafka","Pandas","NumPy","SQL",
    "PostgreSQL","MongoDB","Docker","Kubernetes","AWS SageMaker","SHAP","LLM","RAG"]

BACKEND_SKILLS = ["Python","Java","Go","Rust","Node.js","FastAPI","Django",
    "Spring Boot","gRPC","REST API","GraphQL","PostgreSQL","MySQL","MongoDB",
    "Redis","Kafka","Docker","Kubernetes","AWS","GCP","Azure","Terraform","CI/CD",
    "Microservices","System Design","Linux","Bash"]

DATA_SKILLS = ["Python","SQL","PySpark","Apache Spark","Hadoop","Hive","Airflow",
    "dbt","Kafka","Snowflake","BigQuery","Redshift","Delta Lake","Pandas","NumPy",
    "Tableau","Power BI","ETL","Data Warehousing","AWS Glue","PostgreSQL","Databricks"]

DEVOPS_SKILLS = ["Docker","Kubernetes","Terraform","Ansible","Jenkins","GitHub Actions",
    "AWS","GCP","Azure","Linux","Bash","Python","Prometheus","Grafana","ELK Stack",
    "Istio","Helm","ArgoCD"]

ROLES = {
    "ML Engineer": {"skills": ML_SKILLS, "min_exp": 2},
    "Senior ML Engineer": {"skills": ML_SKILLS, "min_exp": 4},
    "Backend Engineer": {"skills": BACKEND_SKILLS, "min_exp": 2},
    "Data Engineer": {"skills": DATA_SKILLS, "min_exp": 2},
    "Data Scientist": {"skills": ML_SKILLS + DATA_SKILLS, "min_exp": 1},
    "DevOps Engineer": {"skills": DEVOPS_SKILLS, "min_exp": 2},
    "MLOps Engineer": {"skills": ML_SKILLS + DEVOPS_SKILLS, "min_exp": 3},
}

EDU_LEVELS = ["high_school", "diploma", "bachelor", "master", "phd"]
EDU_SCORE = {l: i/4 for i, l in enumerate(EDU_LEVELS)}


def _overlap(a: list, b: list) -> float:
    if not b: return 0.0
    al = {x.lower() for x in a}
    return sum(1 for x in b if any(x.lower() in y or y in x.lower() for y in al)) / len(b)


def _label(skill: float, mandatory: float, exp: float, edu: float) -> int:
    if mandatory < 0.4 or (skill < 0.2 and exp < 0.3): return 0
    s = 0.45*skill + 0.25*mandatory + 0.20*exp + 0.10*edu
    if s >= 0.80: return 3
    if s >= 0.60: return 2
    if s >= 0.35: return 1
    return 0


def generate_dataset(n_jds: int = 100, candidates_per_jd: int = 50) -> pd.DataFrame:
    rows = []
    for qid in range(n_jds):
        role = random.choice(list(ROLES.keys()))
        cfg = ROLES[role]
        pool = cfg["skills"]
        min_exp = cfg["min_exp"]

        n_req = random.randint(6, 12)
        jd_req = random.sample(pool, min(n_req, len(pool)))
        mandatory = jd_req[:int(n_req*0.6)]
        preferred = random.sample([s for s in pool if s not in jd_req], min(5, len(pool)-n_req))

        for _ in range(candidates_per_jd):
            q = random.random()  # candidate quality 0-1
            n_skills = max(1, int(len(pool) * q))
            cand_skills = random.sample(pool, n_skills)
            # add domain noise
            cand_skills += random.sample(ML_SKILLS + BACKEND_SKILLS, random.randint(0, 4))

            skill_match = _overlap(cand_skills, jd_req)
            mand_cov    = _overlap(cand_skills, mandatory)
            pref_cov    = _overlap(cand_skills, preferred)

            exp_yrs   = random.uniform(0, 15)
            exp_match = min(1.0, exp_yrs / max(min_exp, 1)) if exp_yrs < min_exp else max(0.3, 1 - (exp_yrs - min_exp*2)/20)
            exp_match = max(0.0, exp_match)

            edu = random.choices(EDU_LEVELS, weights=[5,10,45,30,10])[0]
            edu_score = EDU_SCORE[edu]
            edu_field = random.choices([0.0, 0.6, 1.0], weights=[15,25,60])[0]

            sem_sim = float(np.clip(0.6*skill_match + 0.4*random.random() + np.random.normal(0,0.05), 0, 1))
            role_title_sim = random.choices([0.2,0.5,0.7,0.9], weights=[15,25,35,25])[0]

            is_dup   = float(random.random() < 0.05)
            stuffing = float(np.clip(np.random.exponential(0.1), 0, 1)) if random.random() > 0.9 else 0.0

            proj_count = max(0, int(np.random.poisson(3*q)))
            proj_rel   = float(np.clip(skill_match * random.uniform(0.5, 1.0), 0, 1))
            cert_count = max(0, int(np.random.poisson(1.5*q)))

            lbl = _label(skill_match, mand_cov, exp_match, edu_score)
            if is_dup or stuffing > 0.7:
                lbl = max(0, lbl - 2)

            rows.append({
                "query_id": qid, "resume_id": str(uuid.uuid4()), "jd_title": role,
                "semantic_similarity":        round(sem_sim, 4),
                "skill_match_ratio":          round(skill_match, 4),
                "mandatory_skill_coverage":   round(mand_cov, 4),
                "preferred_skill_coverage":   round(pref_cov, 4),
                "experience_years":           round(exp_yrs, 2),
                "experience_match_score":     round(exp_match, 4),
                "role_title_similarity":      round(role_title_sim, 4),
                "education_level_score":      round(edu_score, 4),
                "education_field_match":      round(edu_field, 4),
                "keyword_stuffing_score":     round(stuffing, 4),
                "is_duplicate":               is_dup,
                "project_count":              float(proj_count),
                "has_relevant_projects":      round(proj_rel, 4),
                "certification_count":        float(cert_count),
                "relevance_label":            lbl,
            })

    df = pd.DataFrame(rows)
    print(f"Generated {len(df)} records | Label dist:\n{df['relevance_label'].value_counts().sort_index()}")
    return df


if __name__ == "__main__":
    import pathlib
    pathlib.Path("data/synthetic").mkdir(parents=True, exist_ok=True)
    df = generate_dataset(n_jds=100, candidates_per_jd=50)
    df.to_csv("data/synthetic/ltr_dataset.csv", index=False)
    print("Saved → data/synthetic/ltr_dataset.csv")
