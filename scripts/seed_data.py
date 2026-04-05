"""
scripts/seed_data.py
─────────────────────
Seed the system with sample JDs for demo/testing purposes.

Usage:
    python scripts/seed_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from loguru import logger

API_BASE = "http://localhost:8000/api/v1"

SAMPLE_JDS = [
    {
        "raw_text": """Senior Machine Learning Engineer

We are building the next generation of AI infrastructure and are looking for
a Senior ML Engineer to join our core platform team.

Responsibilities:
• Design and implement large-scale ML training pipelines
• Build and maintain MLOps infrastructure (Kubeflow, MLflow)
• Deploy models to production using Docker and Kubernetes
• Collaborate with research team to productionize SOTA models
• Mentor junior engineers

Requirements:
• 4+ years experience in machine learning engineering
• Expert Python programming skills
• Proficiency with TensorFlow or PyTorch
• Experience with distributed training (Horovod, DeepSpeed)
• Strong knowledge of NLP, BERT, Transformer architectures
• MLOps experience: experiment tracking, model versioning, A/B testing
• SQL and working with large datasets (Spark, BigQuery)
• Docker and Kubernetes for containerized deployments
• AWS or GCP cloud platforms

Preferred:
• Experience with LLMs, fine-tuning, PEFT, LoRA
• Knowledge of RAG architectures
• Publications in top ML venues (NeurIPS, ICML, ACL)

Education: B.Tech/M.Tech in CS, AI, or related field
Location: Bangalore / Hybrid"""
    },
    {
        "raw_text": """Data Engineer — Platform Team

Join our data platform team building the backbone of our analytics infrastructure.

Requirements:
• 3+ years experience in data engineering
• Expert SQL and Python skills
• Apache Spark / PySpark for large-scale data processing
• Pipeline orchestration: Airflow or Prefect
• Modern data stack: dbt, Snowflake or BigQuery, Delta Lake
• Kafka for real-time streaming
• Data modeling and warehouse design
• AWS (S3, Glue, EMR) or GCP (Dataflow, GCS)

Preferred:
• Experience with Databricks
• Knowledge of data governance and data quality frameworks
• Stream processing with Flink

Education: B.Tech in CS or related field
Experience: 3-7 years"""
    },
    {
        "raw_text": """Backend Engineer — APIs & Infrastructure

We're looking for a Backend Engineer to build high-performance APIs and services.

Mandatory:
• 2+ years backend engineering experience
• Python (FastAPI or Django) OR Go for API development
• PostgreSQL and Redis for data storage and caching
• RESTful API design principles
• Docker for containerization
• Git and CI/CD pipelines

Nice to have:
• Microservices architecture experience
• Kafka or RabbitMQ for message queuing
• Kubernetes for orchestration
• GraphQL API experience
• AWS deployment experience

Education: B.Tech in CS or equivalent
Location: Delhi NCR / Remote"""
    },
]


def seed_jds():
    client = httpx.Client(base_url=API_BASE, timeout=30.0)

    logger.info(f"Seeding {len(SAMPLE_JDS)} job descriptions...")
    for i, jd_data in enumerate(SAMPLE_JDS):
        try:
            resp = client.post("/jobs/", json=jd_data)
            resp.raise_for_status()
            result = resp.json()
            logger.success(f"  [{i+1}] Created: '{result.get('title')}' (ID: {result.get('jd_id','')[:8]}...)")
        except httpx.HTTPStatusError as e:
            logger.error(f"  [{i+1}] Failed: {e}")
        except httpx.ConnectError:
            logger.error("Cannot connect to API. Make sure backend is running: uvicorn backend.main:app --reload")
            break

    client.close()
    logger.info("Seeding complete!")


if __name__ == "__main__":
    seed_jds()
