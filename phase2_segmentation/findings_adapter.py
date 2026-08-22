"""
findings_adapter.py
-------------------
Converts the flat list of Candidate objects from Phase 2 into a
StructuredFindings dictionary compatible with Phase 3.

Clinical filtering rules (mirrors real PACS workflows):
  - Only candidates with fused_confidence >= MIN_CONFIDENCE are included
  - Only candidates with volume_cc >= MIN_VOLUME_CC are included
  - At most MAX_ANOMALIES_PER_ORGAN top-confidence candidates per organ
  - Organs that have candidates but none pass the filter are listed as
    status=normal so the report doesn't omit significant anatomy
"""

from typing import List, Dict, Any
from datetime import datetime, timezone
import logging

from schemas.candidate import Candidate

logger = logging.getLogger(__name__)

# ── Clinical significance thresholds ──────────────────────────────────────────
# Only candidates passing BOTH thresholds are included in the report.
# These mirror what a radiologist would flag as actionable.
MIN_CONFIDENCE: float = 0.45    # fused_confidence must be >= this (0-1)
MIN_VOLUME_CC: float = 0.3      # volume in cm³ must be >= this
MAX_ANOMALIES_PER_ORGAN: int = 5  # max findings listed per organ


def to_structured_findings(
    candidates: List[Candidate],
    series_meta: Dict[str, Any],
    target_organs: List[str] = None,
) -> Dict[str, Any]:
    """
    Transforms Phase 2 Candidates into Phase 3 StructuredFindings format.

    Applies clinical significance filters before grouping so the generated
    report contains only actionable findings, not thousands of noise candidates.
    """
    total_candidates = len([c for c in candidates if not c.suppressed])

    # ── Step 1: Apply significance filter ─────────────────────────────────────
    significant: List[Candidate] = []
    noise_count = 0

    for candidate in candidates:
        if candidate.suppressed:
            continue

        volume_cc = candidate.shape.volume_cc if candidate.shape else 0.0
        confidence = candidate.fused_confidence or 0.0

        if confidence >= MIN_CONFIDENCE and volume_cc >= MIN_VOLUME_CC:
            significant.append(candidate)
        else:
            noise_count += 1

    logger.info(
        "findings_adapter: %d total candidates → %d significant (confidence≥%.2f, volume≥%.1fcc), "
        "%d filtered as noise",
        total_candidates, len(significant), MIN_CONFIDENCE, MIN_VOLUME_CC, noise_count,
    )

    # ── Step 2: Group by organ, cap per organ ─────────────────────────────────
    organ_groups: Dict[str, List[Candidate]] = {}
    for candidate in significant:
        organ_groups.setdefault(candidate.organ_label, []).append(candidate)

    if target_organs:
        target_set = {o.lower() for o in target_organs}
        filtered_groups = {k: v for k, v in organ_groups.items() if k.lower() in target_set}
        if not filtered_groups:
            logger.warning("No candidates match target_organs=%s; returning all significant", target_organs)
        else:
            organ_groups = filtered_groups

    # Sort each organ's candidates by confidence descending, then cap
    for organ_name in organ_groups:
        organ_groups[organ_name].sort(key=lambda c: c.fused_confidence or 0.0, reverse=True)
        organ_groups[organ_name] = organ_groups[organ_name][:MAX_ANOMALIES_PER_ORGAN]

    organs = []

    # ── Step 3: Build organ finding entries ───────────────────────────────────
    for organ_name, organ_candidates in organ_groups.items():
        anomalies = []
        for candidate in organ_candidates:
            # Map shape
            shape = {}
            if candidate.shape:
                if candidate.shape.volume_cc > 0:
                    shape["volume_cc"] = round(candidate.shape.volume_cc, 3)
                if candidate.shape.long_axis_mm > 0:
                    shape["long_axis_mm"] = round(candidate.shape.long_axis_mm, 2)
                if candidate.shape.short_axis_mm > 0:
                    shape["short_axis_mm"] = round(candidate.shape.short_axis_mm, 2)
                if candidate.shape.sphericity > 0:
                    shape["sphericity"] = round(candidate.shape.sphericity, 4)
                if candidate.shape.margin_curvature_variance > 0:
                    if candidate.shape.margin_curvature_variance > 50:
                        shape["margin"] = "spiculated or irregular"
                    elif candidate.shape.margin_curvature_variance > 20:
                        shape["margin"] = "lobulated"
                    else:
                        shape["margin"] = "well-defined"

            # Map density
            density_hu = {}
            if candidate.density_hu:
                density_hu["mean"] = round(candidate.density_hu.mean, 1)
                if candidate.density_hu.min != 0 or candidate.density_hu.max != 0:
                    density_hu["min"] = round(candidate.density_hu.min, 1)
                    density_hu["max"] = round(candidate.density_hu.max, 1)
                if candidate.density_hu.std > 0:
                    density_hu["std"] = round(candidate.density_hu.std, 1)

            # Determine anomaly type
            atype = candidate.phase1_anomaly_type or "lesion"

            # Human-readable location from organ name
            location = organ_name.replace("_", " ").title()

            anomaly = {
                "anomaly_id": candidate.candidate_id,
                "type": atype,
                "location": location,
                "confidence": round(candidate.fused_confidence or 0.0, 3),
            }
            if shape:
                anomaly["shape"] = shape
            if density_hu:
                anomaly["density_hu"] = density_hu

            anomalies.append(anomaly)

        organ_finding = {
            "organ": organ_name,
            "status": "anomaly_detected" if anomalies else "normal",
        }
        if anomalies:
            organ_finding["anomalies"] = anomalies

        organs.append(organ_finding)

    # ── Step 4: Handle the no-significant-findings case ───────────────────────
    if not organs:
        organs.append({
            "organ": "whole_body",
            "status": "normal",
        })
        logger.info(
            "findings_adapter: no significant findings after filtering "
            "(%d noise candidates suppressed) — reporting as normal study.",
            noise_count,
        )

    structured_findings = {
        "study_id": series_meta.get("study_instance_uid", "unknown_study"),
        "modality": series_meta.get("modality", "CT"),
        "patient_id": series_meta.get("patient_id", "UNKNOWN_PATIENT"),
        "patient_name": series_meta.get("patient_name", "Unknown Patient"),
        "patient_age": series_meta.get("patient_age", ""),
        "patient_sex": series_meta.get("patient_sex", ""),
        "organs": organs,
    }

    protocol = series_meta.get("protocol")
    if protocol:
        structured_findings["protocol"] = protocol

    logger.info(
        "findings_adapter: built StructuredFindings with %d organ sections "
        "from %d significant candidates",
        len(organs), len(significant),
    )

    return structured_findings
