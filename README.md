# 🧠 Intelligent Hiring Engine
### Explainable AI Resume Screening + Candidate Fit Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A production-grade AI hiring system that parses resumes, understands job descriptions with LLMs, semantically matches candidates, and provides explainable rankings — built to demonstrate ML engineering depth for 12–20 LPA roles.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INTELLIGENT HIRING ENGINE                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────┐   PDF    ┌──────────────┐  JSON   ┌────────────────┐  │
│  │ Resume   │ ──────► │ Parse Service │ ──────► │ Feature        │  │
│  │ Upload   │          │ (PyMuPDF +   │         │ Engineering    │  │
│  │ (PDF/Doc)│          │  spaCy NLP)  │         │ Service        │  │
│  └──────────┘          └──────────────┘         └───────┬────────┘  │
│                                                          │           │
│  ┌──────────┐  Text   ┌──────────────┐  JSON            │           │
│  │   JD     │ ──────► │  JD Parser   │ ──────►          │           │
│  │  Input   │         │  (LLM-based) │         ┌────────▼────────┐  │
│  └──────────┘         └──────────────┘         │  Embedding Gen  │  │
│                                                 │ (SentenceTransf)│  │
│  ┌───────────────────────────────────────────┐  └────────┬────────┘  │
│  │           RANKING PIPELINE                │           │           │
│  │                                           │  ┌────────▼────────┐  │
│  │  Semantic Score (0.4) + ML Score (0.6)   │◄─│  XGBoost/LGBM   │  │
│  │                                           │  │  Ranking Model  │  │
│  └────────────────────┬──────────────────────┘  └─────────────────┘  │
│                       │                                               │
│  ┌────────────────────▼──────────────────────────────────────────┐   │
│  │                   EXPLAINABILITY LAYER                         │   │
│  │   SHAP Values │ Feature Importance │ Skill Gap Analysis        │   │
│  │   Keyword Stuffing Detection │ Duplicate Resume Detection      │   │
│  └────────────────────┬──────────────────────────────────────────┘   │
│                       │                                               │
│             ┌─────────▼──────────┐                                   │
│             │   FastAPI Backend   │◄── PostgreSQL / MongoDB           │
│             └─────────┬──────────┘                                   │
│                       │                                               │
│             ┌─────────▼──────────┐                                   │
│             │  Streamlit Frontend │                                   │
│             └────────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────┘
```

## 📂 Project Structure

```
hiring-engine/
├── backend/
│   ├── api/
│   │   └── routes/
│   │       ├── resume.py          # Resume upload + parse endpoints
│   │       ├── jobs.py            # JD management endpoints
│   │       ├── ranking.py         # Ranking + match endpoints
│   │       └── analytics.py       # Dashboard analytics endpoints
│   ├── core/
│   │   ├── config.py              # Settings (pydantic BaseSettings)
│   │   └── security.py            # API key auth
│   ├── models/
│   │   ├── resume.py              # Pydantic schemas
│   │   ├── job.py
│   │   └── ranking.py
│   ├── services/
│   │   ├── parser/
│   │   │   ├── resume_parser.py   # PyMuPDF + spaCy pipeline
│   │   │   └── jd_parser.py       # LLM-based JD understanding
│   │   ├── nlp/
│   │   │   ├── embedder.py        # SentenceTransformers
│   │   │   └── skill_extractor.py # Named entity + skill extraction
│   │   ├── ml/
│   │   │   ├── ranker.py          # XGBoost/LightGBM ranker
│   │   │   ├── feature_eng.py     # Feature engineering
│   │   │   └── duplicate_det.py   # LSH-based duplicate detection
│   │   └── explainer/
│   │       ├── shap_explainer.py  # SHAP explanations
│   │       └── skill_gap.py       # Skill gap analysis
│   ├── db/
│   │   ├── database.py            # SQLAlchemy + connection pool
│   │   └── repositories.py        # Data access layer
│   ├── utils/
│   │   ├── text_cleaner.py
│   │   └── keyword_stuffing.py    # Stuffing detection
│   └── main.py                    # FastAPI app entrypoint
├── frontend/
│   └── app.py                     # Streamlit UI
├── ml/
│   ├── training/
│   │   ├── train_ranker.py        # Model training script
│   │   └── synthetic_data.py      # Synthetic dataset generator
│   ├── evaluation/
│   │   └── metrics.py             # NDCG, Precision@K
│   └── artifacts/                 # Saved models + encoders
├── data/
│   ├── raw/                       # Raw resumes + JDs
│   ├── processed/                 # Cleaned + featurized data
│   └── synthetic/                 # Generated training data
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
├── tests/
│   ├── unit/
│   └── integration/
├── scripts/
│   ├── setup_db.py
│   └── seed_data.py
└── requirements.txt
```

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/yourname/hiring-engine
cd hiring-engine
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Setup environment
cp .env.example .env
# Edit .env with your API keys

# 3. Initialize DB
python scripts/setup_db.py

# 4. Train the ranking model
python ml/training/train_ranker.py

# 5. Run backend
uvicorn backend.main:app --reload --port 8000

# 6. Run frontend (new terminal)
streamlit run frontend/app.py

# OR: Docker Compose
docker-compose -f docker/docker-compose.yml up --build
```

## 📈 Ranking Formula

```
Final Score = 0.40 × Semantic_Similarity
            + 0.35 × ML_Rank_Score (XGBoost)
            + 0.15 × Skill_Match_Ratio
            + 0.10 × Experience_Score
```

## 🎯 Key Features

| Feature | Implementation |
|---|---|
| Resume Parsing | PyMuPDF + spaCy NER |
| JD Understanding | Claude API / OpenAI |
| Semantic Matching | all-MiniLM-L6-v2 |
| ML Ranking | XGBoost LambdaRank |
| Explainability | SHAP TreeExplainer |
| Skill Gap Analysis | Set difference + TF-IDF |
| Duplicate Detection | MinHash LSH |
| Keyword Stuffing | Statistical z-score |

## 📊 Evaluation Metrics

- **NDCG@10**: Normalized Discounted Cumulative Gain
- **Precision@K**: Top-K accuracy
- **MRR**: Mean Reciprocal Rank
- **Spearman ρ**: Rank correlation

---
*Built for portfolio demonstration — 12–20 LPA ML Engineering roles*
