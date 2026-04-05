"""
backend/api/routes/resume.py
─────────────────────────────
Resume upload and parsing endpoints.
"""

from __future__ import annotations

import io
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from loguru import logger

from backend.core.config import get_settings
from backend.models.resume import ParsedResume, ResumeUploadResponse
from backend.services.parser.resume_parser import ResumeParser
from backend.services.ml.duplicate_det import DuplicateDetector, KeywordStuffingDetector

settings = get_settings()
router = APIRouter()

# In-memory store (replace with DB in production)
_resume_store: dict[UUID, ParsedResume] = {}
_duplicate_detector = DuplicateDetector()
_stuffing_detector = KeywordStuffingDetector()
_parser = ResumeParser()


def _validate_upload(file: UploadFile):
    import os
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )


@router.post("/upload", response_model=ResumeUploadResponse, status_code=201)
async def upload_resume(file: UploadFile = File(...)):
    """
    Upload and parse a single resume (PDF or DOCX).

    - Parses to structured JSON
    - Detects keyword stuffing
    - Checks for duplicates
    - Returns resume_id for later ranking
    """
    _validate_upload(file)

    file_bytes = await file.read()
    if len(file_bytes) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, "File too large")

    try:
        resume = _parser.parse(file_bytes, filename=file.filename or "resume.pdf")
    except Exception as e:
        logger.error(f"Parsing failed: {e}")
        raise HTTPException(422, f"Failed to parse resume: {str(e)}")

    warnings = []

    # Keyword stuffing check
    stuffing_score = _stuffing_detector.score(resume.raw_text, file_bytes)
    resume.keyword_stuffing_score = stuffing_score
    resume.has_keyword_stuffing = stuffing_score >= 0.7
    if resume.has_keyword_stuffing:
        warnings.append("⚠️ Possible keyword stuffing detected")

    # Duplicate check
    _duplicate_detector.add_resume(resume)
    if resume.is_duplicate:
        warnings.append(f"⚠️ Duplicate of resume {resume.duplicate_of}")

    _resume_store[resume.resume_id] = resume
    logger.info(f"Resume parsed: {resume.resume_id} ({resume.candidate_name})")

    return ResumeUploadResponse(
        resume_id=resume.resume_id,
        message="Resume parsed successfully",
        parse_status="success",
        warnings=warnings,
    )


@router.post("/upload/bulk", status_code=201)
async def upload_bulk_resumes(files: list[UploadFile] = File(...)):
    """Upload and parse multiple resumes at once."""
    results = []
    for file in files:
        try:
            _validate_upload(file)
            file_bytes = await file.read()
            resume = _parser.parse(file_bytes, filename=file.filename or "resume.pdf")
            stuffing_score = _stuffing_detector.score(resume.raw_text, file_bytes)
            resume.keyword_stuffing_score = stuffing_score
            _duplicate_detector.add_resume(resume)
            _resume_store[resume.resume_id] = resume
            results.append({"filename": file.filename, "resume_id": str(resume.resume_id), "status": "ok"})
        except Exception as e:
            results.append({"filename": file.filename, "status": "error", "detail": str(e)})
    return {"processed": len(results), "results": results}


@router.get("/{resume_id}", response_model=ParsedResume)
async def get_resume(resume_id: UUID):
    """Retrieve a parsed resume by ID."""
    resume = _resume_store.get(resume_id)
    if not resume:
        raise HTTPException(404, f"Resume {resume_id} not found")
    return resume


@router.get("/", response_model=list[ParsedResume])
async def list_resumes(skip: int = 0, limit: int = 50):
    """List all uploaded resumes."""
    all_resumes = list(_resume_store.values())
    return all_resumes[skip: skip + limit]


@router.delete("/{resume_id}", status_code=204)
async def delete_resume(resume_id: UUID):
    if resume_id not in _resume_store:
        raise HTTPException(404, "Resume not found")
    del _resume_store[resume_id]


# Expose store to other routes
def get_resume_store() -> dict[UUID, ParsedResume]:
    return _resume_store
