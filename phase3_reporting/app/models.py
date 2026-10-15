from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ====================================================================== #
#  Phase-2 Input  (Structured Findings JSON)
# ====================================================================== #


class DimensionsMM(BaseModel):
    """Arbitrary dimension keys → values (e.g. length, width, ap, diameter)."""

    model_config = {"extra": "allow"}

    # Convenience: allow any string key so new organ types just work.
    def items(self):  # makes iteration ergonomic in the template engine
        return self.__pydantic_fields_set__  # pragma: no cover

    def as_dict(self) -> Dict[str, float]:
        return self.model_dump()


class DensityHU(BaseModel):
    """HU statistics for an anomaly ROI."""

    mean: float
    min: Optional[float] = None
    max: Optional[float] = None
    std: Optional[float] = None


class AnomalyShape(BaseModel):
    """Shape descriptors for a detected anomaly."""

    volume_cc: Optional[float] = None
    long_axis_mm: Optional[float] = None
    short_axis_mm: Optional[float] = None
    sphericity: Optional[float] = None
    margin: Optional[str] = None

    model_config = {"extra": "allow"}


class Anomaly(BaseModel):
    """A single detected anomaly within an organ."""

    anomaly_id: str
    type: str
    location: str
    shape: Optional[AnomalyShape] = None
    density_hu: Optional[DensityHU] = None
    sub_location: Optional[str] = None   # zone within the host organ (Phase 2 anatomy context)

    model_config = {"extra": "allow"}


class OrganFinding(BaseModel):
    """Findings for a single organ / structure."""

    organ: str                         # e.g. "left_kidney"
    status: Literal["normal", "anomaly_detected", "not_visualised"]
    dimensions_mm: Optional[DimensionsMM] = None
    mean_density_hu: Optional[float] = None
    location: Optional[str] = None
    anomalies: Optional[List[Anomaly]] = Field(default_factory=list)
    adjacent_structures: Optional[List[str]] = None   # structures in surface contact
    touches_fov_edge: Optional[bool] = None           # organ cut by scan boundary
    unassigned: Optional[bool] = None                 # lesions with no organ label

    model_config = {"extra": "allow"}


class StructuredFindings(BaseModel):
    """
    Root model for the Phase-2 output / Phase-3 input.

    Designed to be forward-compatible: extra fields at any level are
    preserved and available via model_extra.
    """

    study_id: str
    modality: str
    protocol: Optional[str] = None
    region: Optional[str] = None
    organs: List[OrganFinding] = Field(..., min_length=1)
    study_date: Optional[str] = None
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    patient_age: Optional[str] = None
    patient_sex: Optional[str] = None
    # ── Report-enrichment context (Phase 1 DICOM headers / Phase 2 QC) ──
    indication: Optional[str] = None
    referring_physician: Optional[str] = None
    accession_number: Optional[str] = None
    technique: Optional[Dict[str, Any]] = None
    comparison: Optional[Dict[str, Any]] = None
    volume_qc: Optional[Dict[str, Any]] = None
    orientation_verified: Optional[bool] = None

    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def organs_not_empty(self) -> "StructuredFindings":
        if not self.organs:
            raise ValueError("'organs' list must contain at least one entry.")
        return self


# ====================================================================== #
#  Internal Report Model
# ====================================================================== #


class ReportStatus(str):
    DRAFT = "draft"
    SIGNED = "signed"


class Report(BaseModel):
    """
    Internal report representation stored by the storage layer.
    """

    study_id: str
    modality: str
    protocol: Optional[str] = None
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    patient_age: Optional[str] = None
    patient_sex: Optional[str] = None
    findings: str                      # full findings text
    impression: str = ""               # filled by radiologist
    validated: bool = True
    validation_reason: str = ""
    status: str = "draft"              # "draft" | "signed"
    source: str = "llm"               # "llm" | "template"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    signed_at: Optional[datetime] = None

    model_config = {"extra": "allow"}


# ====================================================================== #
#  API  –  Request / Response Schemas
# ====================================================================== #


class ValidatorDetails(BaseModel):
    """Detailed status of individual validation checks."""
    numbers: bool = True
    measurements: bool = True
    densities: bool = True
    medical_terms: bool = True


class GenerateReportResponse(BaseModel):
    """Response body for POST /api/v1/generate-report."""

    study_id: str
    findings: str
    impression: str
    validated: bool
    validator: Optional[ValidatorDetails] = None
    status: str
    source: str


class GetReportResponse(BaseModel):
    """Response body for GET /report/{study_id}."""

    study_id: str
    findings: str
    impression: str
    validated: bool
    validation_reason: str
    status: str
    source: str
    created_at: datetime
    updated_at: datetime
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    patient_age: Optional[str] = None
    patient_sex: Optional[str] = None


class UpdateReportRequest(BaseModel):
    """Request body for PUT /report/{study_id}."""

    findings: Optional[str] = None
    impression: Optional[str] = None


class UpdateReportResponse(BaseModel):
    """Response body for PUT /report/{study_id}."""

    success: bool


class SignReportResponse(BaseModel):
    """Response body for POST /sign-report."""

    saved: bool
    status: str
    signed_at: Optional[datetime] = None


class ValidationResult(BaseModel):
    """Result returned by the validator service."""

    validated: bool
    validator: ValidatorDetails = Field(default_factory=ValidatorDetails)
    failed_check: Optional[str] = None
    reason: str = ""
    source: Literal["llm", "template"] = "llm"


class ErrorResponse(BaseModel):
    """Uniform error envelope."""

    detail: str
    code: Optional[str] = None
