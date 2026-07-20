"""
validator.py
------------
Report validator – the anti-hallucination gate.

Before the LLM output is accepted, it is compared against the Structured Findings JSON
across four axes:

    1.  Numbers       – every integer and decimal from the JSON must be present.
    2.  Measurements  – every dimension must be reproduced exactly.
    3.  Densities     – every HU value must be reproduced.
    4.  Key terms     – organs, anomaly types, location words.

If any check fails the LLM output is REJECTED.
"""

from __future__ import annotations

import logging
import re
from typing import Set, NamedTuple

from app.models import StructuredFindings, ValidationResult, ValidatorDetails
from app.utils import extract_numbers, extract_words, format_dimension_string

logger = logging.getLogger(__name__)

class ValidationCheck(NamedTuple):
    """Result of an individual validation check."""
    passed: bool
    reason: str = ""

_MEASUREMENT_RE = re.compile(
    r"\d+(?:\.\d+)?"                                      # first number
    r"(?:(?:\s*mm)?\s*(?:[\u00d7x]|by)\s*\d+(?:\.\d+)?)*" # optional intermediate mm and 'x'/'by'
    r"\s*mm",                                             # must end with mm
    re.IGNORECASE
)

_DENSITY_RE = re.compile(r"-?\d+(?:\.\d+)?\s*HU", re.IGNORECASE)

class ReportValidator:
    def __init__(self, max_optional_missing_terms: int = 3):
        """
        Initialize validator with configurable thresholds.
        
        Parameters
        ----------
        max_optional_missing_terms : int
            Maximum number of optional descriptive terms that can be missing
            before failing validation (default: 3)
        """
        self.max_optional_missing_terms = max_optional_missing_terms

    def validate(
        self,
        findings: StructuredFindings,
        llm_text: str,
    ) -> ValidationResult:
        """
        Multi-level validation with strong anti-hallucination design:
        Level 1: Mandatory field validation (immediate rejection)
        Level 2: Separate numeric validations (exact match required)  
        Level 3: Optional terminology (configurable threshold)
        """
        details = ValidatorDetails()
        failed_check = None
        reason = ""

        # LEVEL 1: MANDATORY FIELD VALIDATION
        mandatory_check = self._validate_mandatory_fields(findings, llm_text)
        if not mandatory_check.passed:
            details.medical_terms = False
            failed_check = "mandatory_fields"
            reason = mandatory_check.reason
        
        # LEVEL 2A: NUMBERS VALIDATION
        if not failed_check:
            numbers_check = self._validate_numbers(findings, llm_text)
            if not numbers_check.passed:
                details.numbers = False
                failed_check = "numbers"
                reason = numbers_check.reason

        # LEVEL 2B: MEASUREMENTS VALIDATION  
        if not failed_check:
            measurements_check = self._validate_measurements(findings, llm_text)
            if not measurements_check.passed:
                details.measurements = False
                failed_check = "measurements"
                reason = measurements_check.reason

        # LEVEL 2C: DENSITIES VALIDATION
        if not failed_check:
            densities_check = self._validate_densities(findings, llm_text)
            if not densities_check.passed:
                details.densities = False
                failed_check = "densities"
                reason = densities_check.reason

        # LEVEL 3: OPTIONAL TERMINOLOGY
        if not failed_check:
            terminology_check = self._validate_optional_terminology(findings, llm_text)
            if not terminology_check.passed:
                details.medical_terms = False
                failed_check = "optional_terms"
                reason = terminology_check.reason

        is_valid = all([
            details.numbers,
            details.measurements,
            details.densities,
            details.medical_terms
        ])

        if not is_valid:
            logger.warning("Validation FAILED (%s): %s", failed_check, reason)
            return ValidationResult(
                validated=False,
                validator=details,
                failed_check=failed_check,
                reason=reason,
                source="template"
            )

        logger.info("Validation PASSED – LLM output accepted.")
        return ValidationResult(
            validated=True,
            validator=details,
            failed_check=None,
            reason="",
            source="llm"
        )

    def _validate_mandatory_fields(self, findings: StructuredFindings, llm_text: str) -> ValidationCheck:
        """
        LEVEL 1: Mandatory field validation - immediate rejection if any critical field missing.
        
        For every organ, verify the report contains:
        ✅ Organ name (kidney, liver, brain) - ALWAYS mandatory
        ✅ Location (if present in JSON) - only mandatory if specified
        
        For every anomaly, verify the report contains:
        ✅ Anomaly type (lesion, cyst, stone, hemorrhage) - ALWAYS mandatory
        ✅ Anomaly location (if present in JSON) - only mandatory if specified
        ✅ Volume (if present in JSON) - only mandatory if specified
        
        No threshold - any missing mandatory field = immediate rejection.
        """
        llm_words = set(extract_words(llm_text.lower()))
        
        for organ in findings.organs:
            # Check organ name is present (ALWAYS mandatory)
            organ_words = extract_words(organ.organ.lower())
            if not organ_words.issubset(llm_words):
                missing_organ_words = organ_words - llm_words
                return ValidationCheck(
                    False, 
                    f"MANDATORY: Organ name missing - '{' '.join(missing_organ_words)}' not found in report"
                )
            
            # Check organ location ONLY if specified in JSON
            if organ.location:
                location_words = extract_words(organ.location.lower())
                if not location_words.issubset(llm_words):
                    missing_location_words = location_words - llm_words
                    return ValidationCheck(
                        False,
                        f"MANDATORY: Organ location missing - '{' '.join(missing_location_words)}' not found in report (required because present in JSON)"
                    )
            
            # For anomalies, check critical fields
            if organ.anomalies:
                for anomaly in organ.anomalies:
                    # Check anomaly type (ALWAYS mandatory for anomalies)
                    anomaly_words = extract_words(anomaly.type.lower())
                    if not anomaly_words.issubset(llm_words):
                        missing_anomaly_words = anomaly_words - llm_words
                        return ValidationCheck(
                            False,
                            f"MANDATORY: Anomaly type missing - '{' '.join(missing_anomaly_words)}' not found in report"
                        )
                    
                    # Check anomaly location (ALWAYS mandatory for anomalies)  
                    anomaly_location_words = extract_words(anomaly.location.lower())
                    if not anomaly_location_words.issubset(llm_words):
                        missing_anomaly_location_words = anomaly_location_words - llm_words
                        return ValidationCheck(
                            False,
                            f"MANDATORY: Anomaly location missing - '{' '.join(missing_anomaly_location_words)}' not found in report"
                        )
                    
                    # Check volume ONLY if present in JSON
                    if anomaly.shape and anomaly.shape.volume_cc is not None:
                        volume_str = self._fmt(anomaly.shape.volume_cc)
                        if volume_str not in extract_numbers(llm_text):
                            return ValidationCheck(
                                False,
                                f"MANDATORY: Volume missing - '{volume_str} cc' not found in report (required because present in JSON)"
                            )
        
        return ValidationCheck(True)

    def _validate_numbers(self, findings: StructuredFindings, llm_text: str) -> ValidationCheck:
        """
        LEVEL 2A: Numbers validation - every number from JSON must appear exactly.
        
        Examples that must match exactly: 22.4, 18, 101, 94 (from confidence)
        """
        req_numbers, _, _, _ = self._extract_from_json(findings)
        
        llm_numbers = extract_numbers(llm_text)
        missing_numbers = req_numbers - llm_numbers
        if missing_numbers:
            return ValidationCheck(
                False,
                f"NUMBERS: Missing from report - {sorted(list(missing_numbers))[:5]}"
            )
        
        return ValidationCheck(True)

    def _validate_measurements(self, findings: StructuredFindings, llm_text: str) -> ValidationCheck:
        """
        LEVEL 2B: Measurements validation - dimension strings must appear exactly (if present).
        
        Examples that must match exactly: "22.4 × 14.1 mm", "101 × 47 × 41 mm"
        Only validates measurements that are actually present in the JSON.
        """
        _, req_measurements, _, _ = self._extract_from_json(findings)
        
        # Only validate if measurements exist in JSON
        if not req_measurements:
            return ValidationCheck(True)
        
        llm_measurements = {self._norm(m) for m in _MEASUREMENT_RE.findall(llm_text)}
        norm_req_measurements = {self._norm(m) for m in req_measurements}
        missing_measurements = norm_req_measurements - llm_measurements
        if missing_measurements:
            return ValidationCheck(
                False,
                f"MEASUREMENTS: Missing from report - {list(missing_measurements)} (required because present in JSON)"
            )
        
        return ValidationCheck(True)

    def _validate_densities(self, findings: StructuredFindings, llm_text: str) -> ValidationCheck:
        """
        LEVEL 2C: Densities validation - HU values must appear exactly (if present).
        
        Examples that must match exactly: "18 HU", "-10 to 45 HU", "30 HU"
        Handles complex patterns like density ranges.
        Only validates densities that are actually present in the JSON.
        """
        _, _, req_densities, _ = self._extract_from_json(findings)
        
        # Only validate if densities exist in JSON
        if not req_densities:
            return ValidationCheck(True)
        
        llm_densities = self._extract_density_values(llm_text)
        norm_req_densities = {self._norm(m) for m in req_densities}
        missing_densities = norm_req_densities - llm_densities
        if missing_densities:
            return ValidationCheck(
                False,
                f"DENSITIES: Missing from report - {list(missing_densities)} (required because present in JSON)"
            )
        
        return ValidationCheck(True)

    def _validate_optional_terminology(self, findings: StructuredFindings, llm_text: str) -> ValidationCheck:
        """
        LEVEL 3: Optional terminology - allows paraphrasing that preserves medical meaning.
        
        ONLY validates descriptive terms that actually exist in THIS patient's JSON findings.
        Does NOT use a generic dictionary of medical terms.
        
        Examples of acceptable paraphrasing that preserves medical meaning:
        - "well-defined" → "well-demarcated" (preserves meaning)
        - "ill-defined" → "poorly-defined" (preserves meaning)
        - If JSON has no descriptors → this check passes automatically
        
        Uses configurable threshold for missing optional descriptive terms.
        """
        # Extract ONLY the optional descriptive terms that exist in THIS patient's JSON
        expected_optional_terms = self._extract_optional_terms_from_json(findings)
        
        # If no optional terms exist in the JSON, automatically pass this level
        if not expected_optional_terms:
            return ValidationCheck(True)
        
        # Check how many of the ACTUAL optional terms are missing
        llm_words = set(extract_words(llm_text.lower()))
        missing_optional_terms = expected_optional_terms - llm_words
        
        if len(missing_optional_terms) > self.max_optional_missing_terms:
            return ValidationCheck(
                False,
                f"OPTIONAL_TERMS: Too many descriptive terms missing ({len(missing_optional_terms)} > {self.max_optional_missing_terms}): {sorted(list(missing_optional_terms))} - paraphrasing should preserve medical meaning"
            )
        
        return ValidationCheck(True)

    def _extract_optional_terms_from_json(self, findings: StructuredFindings) -> Set[str]:
        """
        Extract ONLY the optional descriptive terms that actually exist in this patient's JSON.
        
        Returns descriptive modifiers that enhance meaning but aren't critical for accuracy.
        For example, from "well-defined":
        - "defined" = mandatory (core meaning)
        - "well" = optional (descriptive modifier)
        """
        optional_terms = set()
        
        for organ in findings.organs:
            if organ.anomalies:
                for anomaly in organ.anomalies:
                    # Process margin descriptors if they exist
                    if anomaly.shape and anomaly.shape.margin:
                        margin_text = anomaly.shape.margin.lower()
                        margin_words = extract_words(margin_text)
                        
                        # For common patterns, extract the optional modifiers
                        if 'defined' in margin_words:
                            # "well-defined" → "well" is optional
                            # "ill-defined" → "ill" is optional  
                            # "poorly-defined" → "poorly" is optional
                            modifiers = margin_words - {'defined'}
                            optional_terms.update(modifiers)
                        
                        # Add other descriptive words that aren't core anatomy/measurements
                        descriptive_words = {
                            word for word in margin_words 
                            if word not in {'defined', 'margin', 'border', 'edge', 'contour'}
                            and len(word) > 2  # Skip short words like "of", "in"
                        }
                        optional_terms.update(descriptive_words)
        
        return optional_terms

    def _extract_density_values(self, text: str) -> Set[str]:
        """
        Extract density values from text, handling ranges like '-10 to 45 HU'.
        Returns normalized set of density strings like '18 hu', '-10 hu', etc.
        """
        densities = set()
        
        # First, get all numbers that appear in density contexts
        for match in re.finditer(r'-?\d+(?:\.\d+)?', text):
            num = match.group()
            start, end = match.span()
            
            # Look for 'HU' within reasonable distance after the number
            context_after = text[end:end + 50]  # look ahead up to 50 chars
            context_before = text[max(0, start-20):start]  # look back up to 20 chars
            
            # Check if this number is in a density context
            if ('HU' in context_after or 'hu' in context_after.lower()) and \
               any(keyword in context_before.lower() or keyword in context_after.lower() 
                   for keyword in ['density', 'range:', 'mean', 'sd', 'std']):
                densities.add(self._norm(f"{num} HU"))
        
        return densities

    def _norm(self, s: str) -> str:
        # 1. Strip out all 'mm' units (we will safely append a single one at the end)
        s = re.sub(r"\s*mm\s*", " ", s, flags=re.IGNORECASE)

        # 2. Canonicalize separators (x, ×, by) to " x "
        s = re.sub(r"\s*(?:[\u00d7x]|by)\s*", " x ", s, flags=re.IGNORECASE)

        # 3. Clean up extra spaces
        s = re.sub(r"\s+", " ", s).strip().lower()

        # 4. Re-append the standard unit
        return s + " mm"

    def _fmt(self, val: float) -> str:
        return str(int(val)) if val == int(val) else str(val)

    def _extract_from_json(self, findings: StructuredFindings) -> tuple[Set[str], Set[str], Set[str], Set[str]]:
        numbers: Set[str] = set()
        measurements: Set[str] = set()
        densities: Set[str] = set()
        terms: Set[str] = set()

        for organ in findings.organs:
            terms.update(extract_words(organ.organ))
            if organ.location:
                terms.update(extract_words(organ.location))
            
            # NOTE: anomaly count (e.g. "1") is intentionally NOT added to
            # required numbers. The LLM may legitimately rephrase "1 anomaly"
            # as "a lesion" — that is valid medical English and should not
            # be rejected. Clinical numbers (measurements, densities,
            # confidence) are enforced below.
            
            if organ.dimensions_mm:
                dims = {k: v for k, v in organ.dimensions_mm.as_dict().items() if v is not None}
                if dims:
                    for v in dims.values():
                        numbers.add(self._fmt(v))
                    dim_str = format_dimension_string(dims).replace("\u200b", "") # cleanup zero width
                    measurements.add(f"{dim_str}")
            
            if organ.mean_density_hu is not None:
                val = self._fmt(organ.mean_density_hu)
                numbers.add(val)
                densities.add(f"{val} HU")

            for anomaly in organ.anomalies:
                terms.update(extract_words(anomaly.type))
                terms.update(extract_words(anomaly.location))

                if anomaly.shape:
                    s = anomaly.shape
                    if s.margin:
                        terms.update(extract_words(s.margin))
                    
                    axes = []
                    if s.long_axis_mm is not None:
                        val = self._fmt(s.long_axis_mm)
                        numbers.add(val)
                        axes.append(val)
                    if s.short_axis_mm is not None:
                        val = self._fmt(s.short_axis_mm)
                        numbers.add(val)
                        axes.append(val)
                    if axes:
                        measurements.add(" x ".join(axes) + " mm")
                    
                    if s.volume_cc is not None:
                        numbers.add(self._fmt(s.volume_cc))
                    if s.sphericity is not None:
                        numbers.add(self._fmt(s.sphericity))
                
                if anomaly.density_hu:
                    d = anomaly.density_hu
                    val = self._fmt(d.mean)
                    numbers.add(val)
                    densities.add(f"{val} HU")
                    
                    if d.min is not None and d.max is not None:
                        min_v = self._fmt(d.min)
                        max_v = self._fmt(d.max)
                        numbers.update([min_v, max_v])
                        densities.add(f"{min_v} HU")
                        densities.add(f"{max_v} HU")
                    
                    if d.std is not None:
                        std_v = self._fmt(d.std)
                        numbers.add(std_v)
                        densities.add(f"{std_v} HU")

                if anomaly.confidence is not None:
                    numbers.add(str(round(anomaly.confidence * 100)))
                    
        return numbers, measurements, densities, terms
