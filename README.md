# 🧠 Intelligent Hiring Engine
### Explainable AI Resume Screening + Candidate Fit Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/tests-55%2F55%20passing-brightgreen.svg)](#-evaluation-metrics)
[![Accuracy](https://img.shields.io/badge/recommendation%20accuracy-93.3%25-success.svg)](#-evaluation-metrics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A production-grade AI hiring system that parses resumes, understands job descriptions with LLMs, semantically matches candidates, and provides explainable rankings

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
│  │  Semantic (0.25) + ML (0.45) +            │◄─│  XGBoost        │  │
│  │  Skill (0.20) + Experience (0.10)         │  │  LambdaRank     │  │
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
│   │   │   ├── ranker.py          # XGBoost LambdaRank ranker
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
│   │   └── metrics.py             # NDCG, Precision@K, MRR, MAP
│   └── artifacts/                 # Saved models + encoders
├── data/
│   ├── raw/                       # Raw resumes + JDs
│   ├── processed/                 # Cleaned + featurized data
│   └── synthetic/                 # ltr_dataset.csv (7,500 rows, auto-generated)
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
├── tests/
│   ├── unit/                      # 55 passing unit tests
│   └── integration/
├── scripts/
│   ├── setup_db.py
│   └── seed_data.py
└── requirements.txt
```

## 🌐 Live Deployment (Cloud)

- **Frontend (Streamlit Cloud):** [https://hiring-engine.streamlit.app/](https://hiring-engine.streamlit.app/)
- **Backend API (Render):** [https://hiring-engine-backend.onrender.com](https://hiring-engine-backend.onrender.com) / [Interactive Docs](https://hiring-engine-backend.onrender.com/docs)

## 🚀 Run Frontend Locally (Connecting to Cloud API)

Since the backend and database are hosted on Render, you can easily run just the UI locally and connect it to your live API!

```bash
# 1. Clone and install dependencies
git clone https://github.com/pranjal-2218/hiring-engine
cd hiring-engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run Streamlit with the API_URL environment variable
API_URL="https://hiring-engine-backend.onrender.com/api/v1" streamlit run frontend/app.py
```
*(Your browser will open to `http://localhost:8501`. Any edits you make to the UI code will reflect instantly!)*

## 🛠️ Full Local Setup (Backend + DB + Frontend)
*(If you want to run the entire system on your machine)*

```bash
# 1. Setup environment
cp .env.example .env
# Edit .env with your API keys and your local/remote DATABASE_URL

# 2. Initialize DB and Train Model
python scripts/setup_db.py
python ml/training/train_ranker.py

# 3. Run backend
uvicorn backend.main:app --reload --port 8000

# 4. Run frontend (new terminal)
streamlit run frontend/app.py

# OR: Docker Compose
docker-compose -f docker/docker-compose.yml up --build
```

## 📈 Ranking Formula

```
Final Score = 0.25 × Semantic_Similarity
            + 0.45 × ML_Rank_Score (XGBoost LambdaRank)
            + 0.20 × Skill_Match_Ratio
            + 0.10 × Experience_Score
```

**Recommendation Tiers:**

| Score | Tier |
|---|---|
| ≥ 72 | 🟢 Strong Hire |
| 48 – 71 | 🟡 Consider |
| < 48 or mandatory skills missing | 🔴 Reject |

**Penalty Multipliers:**
- Keyword stuffing detected: `× 0.70`
- Duplicate resume detected: `× 0.00` (auto-disqualified)

## 🎯 Key Features

| Feature | Implementation |
|---|---|
| Resume Parsing | PyMuPDF + spaCy NER |
| JD Understanding | Claude API / OpenAI |
| Semantic Matching | all-MiniLM-L6-v2 |
| ML Ranking | XGBoost LambdaRank (`rank:ndcg`) |
| Explainability | SHAP TreeExplainer |
| Skill Gap Analysis | Set difference + TF-IDF |
| Duplicate Detection | MinHash LSH |
| Keyword Stuffing | Statistical z-score |

## 📊 Evaluation Metrics

| Metric | Score |
|---|---|
| **NDCG@5** | 1.0000 |
| **NDCG@10** | 0.9970 |
| **Precision@5** | 1.0000 |
| **MRR** | 1.0000 |
| **Recommendation Accuracy** | **93.27%** |

Metrics evaluated on a held-out validation set of 1,500 candidates across 30 job descriptions. Evaluated on a synthetic dataset; real-world performance may differ

### Model Configuration

```python
XGBRanker(
    objective       = "rank:ndcg",
    learning_rate   = 0.10,
    max_depth       = 6,
    min_child_weight= 3,
    n_estimators    = 100,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    tree_method     = "hist",
)
```

### LTR Feature Set (14 features)

| Group | Features |
|---|---|
| Similarity | `semantic_similarity`, `skill_match_ratio`, `mandatory_skill_coverage`, `preferred_skill_coverage` |
| Experience | `experience_years`, `experience_match_score`, `role_title_similarity` |
| Education | `education_level_score`, `education_field_match` |
| Quality | `keyword_stuffing_score`, `is_duplicate` |
| Richness | `project_count`, `has_relevant_projects`, `certification_count` |

## 🔌 API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | System health check |
| `GET` | `/api/v1/health` | Health check (Streamlit-compatible) |
| `POST` | `/api/v1/resumes/upload` | Upload and parse a resume |
| `GET` | `/api/v1/resumes/` | List all parsed resumes |
| `POST` | `/api/v1/jobs/` | Parse and store a job description |
| `GET` | `/api/v1/jobs/` | List all job descriptions |
| `POST` | `/api/v1/ranking/rank` | Rank candidates for a JD |
| `GET` | `/api/v1/analytics/dashboard` | Aggregate resume analytics |

Interactive docs: [https://hiring-engine-backend.onrender.com/docs](https://hiring-engine-backend.onrender.com/docs)

---
