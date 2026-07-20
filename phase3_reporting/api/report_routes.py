"""
report_routes.py
----------------
FastAPI router for Phase 3 – Radiology Report Generation.

Endpoints
~~~~~~~~~
    POST  /generate-report        Generate a draft report from findings JSON.
    GET   /report/{study_id}      Retrieve the current draft.
    PUT   /report/{study_id}      Radiologist edits (findings / impression).
    POST  /sign-report            Finalise and persist a signed report.
    GET   /health                 Liveness / Ollama status check.
    GET   /reports                List all stored reports (admin / QA).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Body, Depends, HTTPException, status

from app.models import (
    GenerateReportResponse,
    GetReportResponse,
    Report,
    SignReportResponse,
    StructuredFindings,
    UpdateReportRequest,
    UpdateReportResponse,
    ValidationResult,
)
from services.llm_service import LLMService
from services.template_engine import TemplateEngine
from services.validator import ReportValidator
from storage.storage import ReportStorage, get_storage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Radiology Reports"])

# ====================================================================== #
#  Dependency injectors  (easily replaceable in tests)
# ====================================================================== #


def get_template_engine() -> TemplateEngine:
    return TemplateEngine()


def get_llm_service() -> LLMService:
    return LLMService()


def get_validator() -> ReportValidator:
    return ReportValidator()


def get_report_storage() -> ReportStorage:
    return get_storage()


# ====================================================================== #
#  POST /generate-report
# ====================================================================== #


@router.post(
    "/generate-report",
    response_model=GenerateReportResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate a draft radiology report from structured findings JSON",
    description=(
        "Accepts structured findings JSON (Phase 2 output), passes it through "
        "the deterministic template engine, optionally refines with Ollama, "
        "validates against the template, and stores a draft report in memory "
        "ready for radiologist review."
    ),
)
async def generate_report(
    findings: StructuredFindings = Body(...),
    engine: TemplateEngine = Depends(get_template_engine),
    llm: LLMService = Depends(get_llm_service),
    validator: ReportValidator = Depends(get_validator),
    storage: ReportStorage = Depends(get_report_storage),
) -> GenerateReportResponse:
    logger.info("POST /generate-report  study_id=%s", findings.study_id)

    # ── Step 1: Template engine (deterministic, zero-hallucination) ── #
    try:
        template_result = engine.generate(findings)
    except Exception as exc:
        logger.error("Template engine failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Template engine error: {exc}",
        ) from exc

    # ── Step 2: LLM refinement (best-effort) ──────────────────────── #
    refined_text, source = await llm.refine(template_result.findings_text)

    # ── Step 3: Validation ───────────────────────────────────────────── #
    if source == "llm":
        validation_result = validator.validate(findings, refined_text)
        if not validation_result.validated:
            # Reject LLM output; fall back to template
            logger.warning(
                "LLM output rejected (%s) \u2013 using template.", validation_result.reason
            )
            final_text = template_result.findings_text
            source = "template"
        else:
            final_text = refined_text
    else:
        # Already using template (LLM was unavailable or skipped)
        final_text = refined_text
        validation_result = ValidationResult(
            validated=True, reason="LLM not used", source="template"
        )

    # ── Step 4: Build and persist draft report ────────────────────── #
    now = datetime.now(timezone.utc)
    report = Report(
        study_id=findings.study_id,
        modality=findings.modality,
        protocol=findings.protocol,
        findings=final_text,
        impression="",
        validated=validation_result.validated,
        validation_reason=validation_result.reason,
        status="draft",
        source=source,
        created_at=now,
        updated_at=now,
    )

    try:
        storage.save_report(report)
    except Exception as exc:
        logger.error("Storage save failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to persist draft report: {exc}",
        ) from exc

    return GenerateReportResponse(
        study_id=report.study_id,
        findings=report.findings,
        impression=report.impression,
        validated=validation_result.validated,
        validator=validation_result.validator,
        status="draft",
        source=source,
    )


# ====================================================================== #
#  GET /report/{study_id}
# ====================================================================== #


@router.get(
    "/report/{study_id}",
    response_model=GetReportResponse,
    summary="Retrieve the current draft report for a study",
)
async def get_report(
    study_id: str,
    storage: ReportStorage = Depends(get_report_storage),
) -> GetReportResponse:
    logger.info("GET /report/%s", study_id)

    report = storage.get_report(study_id)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No report found for study_id '{study_id}'.",
        )

    return GetReportResponse(
        study_id=report.study_id,
        findings=report.findings,
        impression=report.impression,
        validated=report.validated,
        validation_reason=report.validation_reason,
        status=report.status,
        source=report.source,
        created_at=report.created_at,
        updated_at=report.updated_at,
    )


# ====================================================================== #
#  PUT /report/{study_id}
# ====================================================================== #


@router.put(
    "/report/{study_id}",
    response_model=UpdateReportResponse,
    summary="Update findings and/or impression (radiologist edits)",
)
async def update_report(
    study_id: str,
    update: UpdateReportRequest = Body(...),
    storage: ReportStorage = Depends(get_report_storage),
) -> UpdateReportResponse:
    logger.info("PUT /report/%s", study_id)

    report = storage.get_report(study_id)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No report found for study_id '{study_id}'.",
        )

    if report.status == "signed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A signed report cannot be modified.",
        )

    # Apply partial updates
    changed = False
    if update.findings is not None:
        report.findings = update.findings
        changed = True
    if update.impression is not None:
        report.impression = update.impression
        changed = True

    if changed:
        report.updated_at = datetime.now(timezone.utc)
        try:
            storage.save_report(report)
        except Exception as exc:
            logger.error("Storage update failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to persist report update: {exc}",
            ) from exc

    return UpdateReportResponse(success=True)


# ====================================================================== #
#  POST /sign-report
# ====================================================================== #


@router.post(
    "/sign-report",
    response_model=SignReportResponse,
    summary="Finalise and sign the report",
    description=(
        "Marks the report as signed and persists the final version.  "
        "Signed reports cannot be further edited via PUT /report/{study_id}."
    ),
)
async def sign_report(
    payload: Dict[str, Any] = Body(..., example={"study_id": "STUDY001"}),
    storage: ReportStorage = Depends(get_report_storage),
) -> SignReportResponse:
    study_id: str = payload.get("study_id", "")
    if not study_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="'study_id' is required in the request body.",
        )

    logger.info("POST /sign-report  study_id=%s", study_id)

    report = storage.get_report(study_id)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No report found for study_id '{study_id}'.",
        )

    if report.status == "signed":
        # Idempotent – already signed
        return SignReportResponse(
            saved=True, status="signed", signed_at=report.signed_at
        )

    now = datetime.now(timezone.utc)
    report.status = "signed"
    report.signed_at = now
    report.updated_at = now

    try:
        storage.save_report(report)
    except Exception as exc:
        logger.error("Storage sign-report failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to persist signed report: {exc}",
        ) from exc

    return SignReportResponse(saved=True, status="signed", signed_at=now)


# ====================================================================== #
#  GET /reports  (admin / QA listing)
# ====================================================================== #


@router.get(
    "/reports",
    summary="List all stored reports (admin endpoint)",
)
async def list_reports(
    storage: ReportStorage = Depends(get_report_storage),
) -> Dict[str, Any]:
    reports = storage.list_reports()
    return {
        "total": len(reports),
        "reports": [
            {
                "study_id": r.study_id,
                "status": r.status,
                "validated": r.validated,
                "source": r.source,
                "created_at": r.created_at.isoformat(),
                "updated_at": r.updated_at.isoformat(),
            }
            for r in reports
        ],
    }


# ====================================================================== #
#  GET /health
# ====================================================================== #


@router.get(
    "/health",
    summary="Liveness check",
)
async def health(
    llm: LLMService = Depends(get_llm_service),
) -> Dict[str, Any]:
    ollama_ok = await llm.health_check()
    return {
        "status": "ok",
        "ollama_reachable": ollama_ok,
        "ollama_model": llm.model,
    }
