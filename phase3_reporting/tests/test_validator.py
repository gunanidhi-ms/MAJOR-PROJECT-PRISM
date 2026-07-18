"""
tests/test_validator.py
-----------------------
Unit tests for the anti-hallucination validator.

Run with:
    pytest tests/test_validator.py -v
"""

import pytest

from app.models import DimensionsMM, OrganFinding, StructuredFindings
from services.template_engine import TemplateEngine, TemplateResult
from services.validator import ReportValidator


@pytest.fixture
def engine() -> TemplateEngine:
    return TemplateEngine()


@pytest.fixture
def validator() -> ReportValidator:
    return ReportValidator()


@pytest.fixture
def simple_findings() -> StructuredFindings:
    from app.models import Anomaly, AnomalyShape, DensityHU

    anomaly = Anomaly(
        anomaly_id="A1",
        type="lesion",
        location="lower pole",
        shape=AnomalyShape(
            volume_cc=4.2,
            long_axis_mm=22.4,
            short_axis_mm=14.1,
            margin="well-defined",
        ),
        density_hu=DensityHU(mean=18, min=-10, max=45, std=15),
        confidence=0.94,
    )
    organ = OrganFinding(
        organ="right_kidney",
        status="anomaly_detected",
        dimensions_mm=DimensionsMM(**{"length": 101, "width": 47, "ap": 41}),
        mean_density_hu=30,
        location="right renal fossa",
        anomalies=[anomaly],
    )
    return StructuredFindings(
        study_id="TEST001",
        modality="CT",
        protocol="CT KUB",
        organs=[organ],
    )


@pytest.fixture
def simple_template_result(engine, simple_findings) -> TemplateResult:
    """A simple normal-organ template result used as baseline."""
    return engine.generate(simple_findings)


# ====================================================================== #
#  Tests: Validated = True (good LLM output)
# ====================================================================== #


class TestValidationPass:
    def test_identical_text_passes(self, validator, simple_findings, simple_template_result):
        """Returning the template unchanged must always pass."""
        result = validator.validate(
            simple_findings, simple_template_result.findings_text
        )
        assert result.validated is True
        assert result.reason == ""

    def test_minor_grammar_fix_passes(self, validator, simple_findings, simple_template_result):
        """A superficial grammar fix that preserves all numbers/terms must pass."""
        # Slightly reword but keep ALL numbers and organ names
        llm_text = simple_template_result.findings_text.replace(
            "appears normal", "is within normal limits"
        )
        # Ensure all numbers are still there
        result = validator.validate(simple_findings, llm_text)
        # Should pass because numbers are preserved
        assert result.validated is True


# ====================================================================== #
#  Tests: Validated = False (hallucination scenarios)
# ====================================================================== #


class TestValidationFail:
    def test_number_changed_fails(self, validator, simple_findings, simple_template_result):
        template = simple_template_result.findings_text
        bad_text = template.replace("22.4", "25.4")  # LLM hallucinates measurement
        result = validator.validate(simple_findings, bad_text)
        assert result.validated is False
        assert result.failed_check in ("numbers", "measurements")

    def test_measurement_removed_fails(self, validator, simple_findings, simple_template_result):
        template = simple_template_result.findings_text
        bad_text = template.replace("22.4 x 14.1 mm", "a small lesion")
        result = validator.validate(simple_findings, bad_text)
        assert result.validated is False
        assert result.failed_check in ("numbers", "measurements")

    def test_density_changed_fails(self, validator, simple_findings, simple_template_result):
        template = simple_template_result.findings_text
        bad_text = template.replace("18 HU", "25 HU")
        result = validator.validate(simple_findings, bad_text)
        assert result.validated is False
        assert result.failed_check in ("numbers", "densities")

    def test_empty_llm_output_fails(self, validator, simple_findings, simple_template_result):
        result = validator.validate(simple_findings, "")
        assert result.validated is False
        assert result.failed_check is not None

    def test_completely_different_text_fails(self, validator, simple_findings, simple_template_result):
        bad_text = "The kidney looks completely normal. No issues found."
        result = validator.validate(simple_findings, bad_text)
        assert result.validated is False
        assert result.failed_check is not None


# ====================================================================== #
#  Tests: Result source field
# ====================================================================== #


class TestValidationSource:
    def test_source_is_llm_on_pass(self, validator, simple_findings, simple_template_result):
        result = validator.validate(
            simple_findings, simple_template_result.findings_text
        )
        assert result.source == "llm"

    def test_source_is_template_on_fail(self, validator, simple_findings, simple_template_result):
        bad_text = simple_template_result.findings_text.replace("22.4", "99.9")
        result = validator.validate(simple_findings, bad_text)
        assert result.source == "template"


# ====================================================================== #
#  Tests: Multiple anomalies / organs
# ====================================================================== #


def _make_multi_organ_findings() -> StructuredFindings:
    from app.models import Anomaly, AnomalyShape, DensityHU

    return StructuredFindings(
        study_id="T2",
        modality="CT",
        organs=[
            OrganFinding(
                organ="left_kidney",
                status="normal",
                dimensions_mm=DimensionsMM(**{"length": 98, "width": 45, "ap": 40}),
                mean_density_hu=32,
            ),
            OrganFinding(
                organ="right_kidney",
                status="anomaly_detected",
                anomalies=[
                    Anomaly(
                        anomaly_id="A1",
                        type="cyst",
                        location="upper pole",
                        shape=AnomalyShape(long_axis_mm=15.6, short_axis_mm=10.3),
                        density_hu=DensityHU(mean=8),
                        confidence=0.97,
                    ),
                ],
            ),
        ],
    )


class TestMultiOrganValidation:
    def test_multi_organ_identical_passes(self, validator, engine):
        findings = _make_multi_organ_findings()
        template = engine.generate(findings)
        result = validator.validate(findings, template.findings_text)
        assert result.validated is True

    def test_multi_organ_number_change_fails(self, validator, engine):
        findings = _make_multi_organ_findings()
        template = engine.generate(findings)
        bad = template.findings_text.replace("15.6", "20.0")
        result = validator.validate(findings, bad)
        assert result.validated is False
