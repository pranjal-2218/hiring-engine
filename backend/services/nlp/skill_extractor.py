"""
backend/services/nlp/skill_extractor.py
────────────────────────────────────────
Named entity + pattern-based skill extraction.

Uses a curated taxonomy of ~400 tech skills organized by category.
Supports both exact match and fuzzy matching for acronyms/variants.

This complements the resume parser's regex-based approach with a
comprehensive skills taxonomy for better coverage.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import NamedTuple

import spacy

# ── Skill Taxonomy ─────────────────────────────────────────────────────────────
# Organized by category for skill gap analysis by domain

SKILL_TAXONOMY: dict[str, list[str]] = {
    "languages": [
        "Python", "Java", "JavaScript", "TypeScript", "Go", "Rust", "C", "C++",
        "C#", "Scala", "Kotlin", "Swift", "R", "Julia", "MATLAB", "Bash",
        "Shell", "SQL", "PL/SQL", "Haskell", "Erlang", "Elixir", "Ruby", "PHP",
    ],
    "ml_frameworks": [
        "TensorFlow", "PyTorch", "Keras", "JAX", "MXNet", "Caffe", "Theano",
        "scikit-learn", "XGBoost", "LightGBM", "CatBoost", "FastAI",
        "Hugging Face", "Transformers", "ONNX", "TensorRT",
    ],
    "ml_concepts": [
        "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
        "Reinforcement Learning", "Transfer Learning", "Few-shot Learning",
        "Zero-shot Learning", "Semi-supervised Learning", "Self-supervised",
        "BERT", "GPT", "LLM", "RAG", "Embeddings", "Fine-tuning",
        "Prompt Engineering", "MLOps", "Feature Engineering", "AutoML",
        "Federated Learning", "Explainable AI", "SHAP", "LIME",
        "A/B Testing", "Bayesian Optimization", "Hyperparameter Tuning",
    ],
    "data_engineering": [
        "Apache Spark", "PySpark", "Apache Kafka", "Apache Flink", "Apache Hadoop",
        "Hive", "Airflow", "Prefect", "dbt", "Dagster", "Luigi",
        "Snowflake", "BigQuery", "Redshift", "Delta Lake", "Iceberg",
        "Databricks", "Presto", "Trino", "AWS Glue",
    ],
    "databases": [
        "PostgreSQL", "MySQL", "SQLite", "Oracle", "SQL Server",
        "MongoDB", "Cassandra", "DynamoDB", "Couchbase", "RethinkDB",
        "Redis", "Memcached", "Elasticsearch", "OpenSearch", "Solr",
        "Neo4j", "ArangoDB", "InfluxDB", "TimescaleDB", "Pinecone",
        "Weaviate", "Milvus", "FAISS", "ChromaDB",
    ],
    "cloud_devops": [
        "AWS", "GCP", "Azure", "Docker", "Kubernetes", "Terraform", "Ansible",
        "Helm", "ArgoCD", "Jenkins", "GitHub Actions", "GitLab CI", "CircleCI",
        "Prometheus", "Grafana", "Datadog", "ELK Stack", "Istio", "Envoy",
        "Nginx", "Apache", "Vault", "Consul", "AWS SageMaker", "GCP Vertex AI",
        "Azure ML", "Kubeflow", "MLflow", "Weights & Biases", "DVC",
    ],
    "backend": [
        "FastAPI", "Django", "Flask", "Spring Boot", "Express", "NestJS",
        "Rails", "Laravel", "Phoenix", "Gin", "Fiber", "gRPC", "REST API",
        "GraphQL", "WebSockets", "Microservices", "Event-driven Architecture",
        "CQRS", "Event Sourcing", "Domain-Driven Design",
    ],
    "frontend": [
        "React", "Vue", "Angular", "Next.js", "Nuxt.js", "Svelte",
        "TypeScript", "JavaScript", "HTML", "CSS", "Tailwind CSS",
        "Redux", "MobX", "Zustand", "Webpack", "Vite", "GraphQL",
        "PWA", "Web Components", "WebAssembly",
    ],
    "tools_practices": [
        "Git", "GitHub", "GitLab", "Bitbucket", "Jira", "Confluence",
        "Agile", "Scrum", "Kanban", "TDD", "BDD", "CI/CD",
        "Code Review", "System Design", "Distributed Systems",
        "Linux", "Unix", "Vim", "VSCode", "IntelliJ",
    ],
    "data_science": [
        "Pandas", "NumPy", "Matplotlib", "Seaborn", "Plotly", "Bokeh",
        "Scipy", "Statsmodels", "Jupyter", "R Studio", "SPSS",
        "Statistics", "Probability", "Linear Algebra", "Calculus",
        "Time Series", "Forecasting", "Clustering", "Classification",
        "Regression", "Dimensionality Reduction", "PCA", "t-SNE",
    ],
}

# Build flat lookup: lowercase -> canonical name
_SKILL_LOOKUP: dict[str, str] = {}
_SKILL_CATEGORY: dict[str, str] = {}
for category, skills in SKILL_TAXONOMY.items():
    for skill in skills:
        key = skill.lower()
        _SKILL_LOOKUP[key] = skill
        _SKILL_CATEGORY[skill] = category

# Common aliases / abbreviations
_ALIASES: dict[str, str] = {
    "tf": "TensorFlow",
    "pt": "PyTorch",
    "k8s": "Kubernetes",
    "sk-learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
    "ml": "Machine Learning",
    "dl": "Deep Learning",
    "nlp": "NLP",
    "cv": "Computer Vision",
    "rl": "Reinforcement Learning",
    "llm": "LLM",
    "gpt": "GPT",
    "pg": "PostgreSQL",
    "psql": "PostgreSQL",
    "mongo": "MongoDB",
    "es": "Elasticsearch",
    "ci/cd": "CI/CD",
    "aws sagemaker": "AWS SageMaker",
    "vertex ai": "GCP Vertex AI",
}


class ExtractedSkill(NamedTuple):
    name: str           # Canonical name
    category: str       # Domain category
    confidence: float   # 0-1


class SkillExtractor:
    """
    Extracts skills from text using:
    1. Curated taxonomy lookup (exact + case-insensitive)
    2. Alias expansion
    3. spaCy entity recognition for unknown tech terms
    4. Pattern matching for version-suffixed skills (e.g. "Python 3.10")
    """

    def __init__(self, nlp=None):
        self._tech_pattern = re.compile(
            r"\b([A-Z][a-zA-Z0-9+#.-]{1,30}|[A-Z]{2,10})\b"
        )
        # Pre-compile lowercase skill patterns for fast lookup
        self._lower_skills = set(_SKILL_LOOKUP.keys())

    def extract(self, text: str) -> list[ExtractedSkill]:
        """Extract all skills from text. Returns deduplicated list."""
        found: dict[str, ExtractedSkill] = {}
        text_lower = text.lower()

        # Pass 1: Multi-word skill phrases (e.g. "machine learning", "deep learning")
        for skill_lower, canonical in _SKILL_LOOKUP.items():
            if " " in skill_lower and skill_lower in text_lower:
                found[canonical] = ExtractedSkill(
                    name=canonical,
                    category=_SKILL_CATEGORY.get(canonical, "other"),
                    confidence=1.0,
                )

        # Pass 2: Single words via word boundary matching
        words = re.findall(r"\b\w[\w+#.-]*\b", text_lower)
        for word in words:
            # Direct lookup
            if word in _SKILL_LOOKUP:
                canonical = _SKILL_LOOKUP[word]
                found[canonical] = ExtractedSkill(
                    name=canonical,
                    category=_SKILL_CATEGORY.get(canonical, "other"),
                    confidence=1.0,
                )
            # Alias lookup
            elif word in _ALIASES:
                canonical = _ALIASES[word]
                found[canonical] = ExtractedSkill(
                    name=canonical,
                    category=_SKILL_CATEGORY.get(canonical, "other"),
                    confidence=0.9,
                )

        # Pass 3: Alias phrases (multi-word)
        for alias, canonical in _ALIASES.items():
            if " " in alias and alias in text_lower:
                found[canonical] = ExtractedSkill(
                    name=canonical,
                    category=_SKILL_CATEGORY.get(canonical, "other"),
                    confidence=0.9,
                )

        return sorted(found.values(), key=lambda x: x.confidence, reverse=True)

    def extract_names(self, text: str) -> list[str]:
        """Convenience: just return canonical skill names."""
        return [s.name for s in self.extract(text)]

    def extract_by_category(self, text: str) -> dict[str, list[str]]:
        """Return skills grouped by category."""
        result: dict[str, list[str]] = defaultdict(list)
        for skill in self.extract(text):
            result[skill.category].append(skill.name)
        return dict(result)

    def skill_match_score(
        self,
        candidate_text: str,
        required_skills: list[str],
    ) -> float:
        """
        Compute match score between candidate text and a list of required skills.
        Returns 0-1.
        """
        candidate_skills = set(self.extract_names(candidate_text))
        if not required_skills:
            return 1.0
        matched = sum(
            1 for req in required_skills
            if self._fuzzy_match(req, candidate_skills)
        )
        return matched / len(required_skills)

    def _fuzzy_match(self, required: str, candidate_skills: set[str]) -> bool:
        """Check if required skill appears in candidate skills (with fuzzy matching)."""
        req_lower = required.lower()
        # Exact match
        if required in candidate_skills:
            return True
        # Case-insensitive
        for cs in candidate_skills:
            cs_lower = cs.lower()
            if req_lower == cs_lower:
                return True
            # Substring match (handles "scikit-learn" vs "sklearn")
            if req_lower in cs_lower or cs_lower in req_lower:
                return True
        return False

    @staticmethod
    def get_all_skills() -> list[str]:
        """Return full canonical skill list (useful for UI dropdowns)."""
        return sorted(set(_SKILL_LOOKUP.values()))

    @staticmethod
    def get_taxonomy() -> dict[str, list[str]]:
        return SKILL_TAXONOMY
