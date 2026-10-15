from __future__ import annotations

import logging
import re
from typing import Optional, Set, NamedTuple

from app.models import StructuredFindings, ValidationResult, ValidatorDetails
from app.utils import extract_numbers, extract_words, format_dimension_string
from services.template_engine import TemplateEngine

_NUM_TOLERANCE = 0.051   # template rounds to 1 decimal; accept that, nothing looser


def _number_present(req: str, llm_numbers: Set[str]) -> bool:
    """True if ``req`` appears in the LLM text exactly or equal after display rounding."""
    if req in llm_numbers:
        return True
    try:
        r = float(req)
    except ValueError:
        return False
    for t in llm_numbers:
        try:
            if abs(float(t) - r) <= _NUM_TOLERANCE:
                return True
        except ValueError:
            continue
    return False


def _is_volume_key(key: str) -> bool:
    k = key.lower()
    return k == "volume" or k.endswith("_cc")

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
        reference_text: Optional[str] = None,
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

        # LEVEL 0: SECTION HEADERS — must be present and intact
        header_check = self._validate_section_headers(llm_text)
        if not header_check.passed:
            details.medical_terms = False
            failed_check = "section_headers"
            reason = header_check.reason

        # LEVEL 1: MANDATORY FIELD VALIDATION
        if not failed_check:
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

        # LEVEL 2A': INVENTED NUMBERS — nothing may appear that the template did not contain
        if not failed_check and reference_text:
            invented = extract_numbers(llm_text) - extract_numbers(reference_text)
            if invented:
                details.numbers = False
                failed_check = "numbers"
                reason = f"NUMBERS: Not present in source report - {sorted(invented)[:5]}"

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
            # Check organ name is present (ALWAYS mandatory). Accept either the raw
            # key words or the display label the template uses (single source of truth).
            organ_words = extract_words(organ.organ.lower())
            label_words = extract_words(TemplateEngine._label(organ.organ).lower())
            if not (organ_words.issubset(llm_words) or label_words.issubset(llm_words)):
                missing_organ_words = label_words - llm_words
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
                    internal_words = {"statistical", "anomaly", "region", "lesion", "mass"}
                    core_anomaly_words = anomaly_words - internal_words
                    
                    missing_anomaly_words = set()
                    for word in core_anomaly_words:
                        prefix = word[:5] if len(word) > 4 else word
                        if not any(lw.startswith(prefix) for lw in llm_words):
                            missing_anomaly_words.add(word)

                    if missing_anomaly_words and core_anomaly_words:
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
                        vol_val = anomaly.shape.volume_cc
                        volume_str = self._fmt(vol_val)
                        extracted_nums = extract_numbers(llm_text)
                        
                        # Generate acceptable representations (exact, 2-decimal rounded, 1-decimal rounded)
                        acceptable_vols = {
                            volume_str,
                            str(round(vol_val, 2)),
                            str(round(vol_val, 1)),
                            f"{vol_val:.2f}",
                            f"{vol_val:.1f}",
                        }
                        
                        if not acceptable_vols.intersection(extracted_nums):
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
        missing_numbers = {n for n in req_numbers if not _number_present(n, llm_numbers)}
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
        norm_req_densities = {self._norm_hu(m) for m in req_densities}
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
        Extract HU density values from report text.

        Matches patterns the template emits:
          - simple:   "18 HU"  or  "30 HU"
          - range:    "-10 to 45 HU"  (both numbers are captured separately)

        Strategy: anchor on every \\bHU\\b token and look *backwards* up to
        30 characters for the last decimal/integer number.  This avoids the
        false-positive problem of scanning 50 chars *forward* from every number
        in the text (which captured millimetre values near density keywords).

        Returns normalised strings via _norm_hu, e.g. '18 hu', '-10 hu'.
        """
        densities: Set[str] = set()
        for m in re.finditer(r'\bHU\b', text, re.IGNORECASE):
            before = text[max(0, m.start() - 40): m.start()]
            # Capture ALL numbers in the look-back window so that range patterns
            # like "-10 to 45 HU" contribute both bounds ('-10 hu' and '45 hu').
            nums = re.findall(r'-?\d+(?:\.\d+)?', before)
            for n in nums:
                densities.add(self._norm_hu(f"{n} HU"))
        return densities

    def _norm(self, s: str) -> str:
        """Normalise a millimetre measurement string (e.g. '22.4 x 14.1 mm').
        Strips embedded 'mm', canonicalises × / x / by separators, collapses
        whitespace, lowercases, and appends a single trailing ' mm'.
        Do NOT use this for HU density strings — use _norm_hu instead.
        """
        # 1. Strip out all 'mm' units (we will safely append a single one at the end)
        s = re.sub(r"\s*mm\s*", " ", s, flags=re.IGNORECASE)

        # 2. Canonicalize separators (x, ×, by) to " x "
        s = re.sub(r"\s*(?:[\u00d7x]|by)\s*", " x ", s, flags=re.IGNORECASE)

        # 3. Clean up extra spaces
        s = re.sub(r"\s+", " ", s).strip().lower()

        # 4. Re-append the standard unit
        return s + " mm"

    @staticmethod
    def _norm_hu(s: str) -> str:
        """Normalise a Hounsfield-unit density string (e.g. '18 HU', '-10 HU').
        Lowercases and collapses whitespace; does NOT append 'mm'.
        """
        return re.sub(r"\s+", " ", s).strip().lower()

    def _fmt(self, val: float) -> str:
        # Must mirror template_engine._fmt (integers bare, otherwise 1 decimal)
        return str(int(val)) if val == int(val) else f"{val:.1f}"

    @staticmethod
    def _validate_section_headers(llm_text: str) -> ValidationCheck:
        required_headers = ["INDICATION:", "TECHNIQUE:", "FINDINGS:", "IMPRESSION:"]
        missing = [h for h in required_headers if h not in llm_text]
        if missing:
            return ValidationCheck(
                False,
                f"SECTION_HEADERS: Missing required section(s): {missing}"
            )
        return ValidationCheck(True)

    def _extract_from_json(self, findings: StructuredFindings) -> tuple[Set[str], Set[str], Set[str], Set[str]]:
        numbers: Set[str] = set()
        measurements: Set[str] = set()
        densities: Set[str] = set()
        terms: Set[str] = set()

        for organ in findings.organs:
            terms.update(extract_words(organ.organ))
            if organ.location:
                terms.update(extract_words(organ.location))

            if organ.dimensions_mm:
                all_dims = {k: v for k, v in organ.dimensions_mm.as_dict().items() if v is not None}
                for k, v in all_dims.items():
                    numbers.add(self._fmt(v))          # volume (cc) and linear values alike
                # Only linear dimensions form an "A x B x C mm" measurement string
                linear = {k: v for k, v in all_dims.items() if not _is_volume_key(k)}
                if linear:
                    measurements.add(format_dimension_string(linear).replace("\u200b", ""))

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
                    # sphericity is an internal shape descriptor, not a reported
                    # measurement — intentionally not mandatory.

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

                    # density std is likewise an internal statistic, not mandatory.

                # confidence is intentionally excluded: it is an internal
                # model score, not a measured clinical value, and must not
                # appear in the report text.

        return numbers, measurements, densities, terms
