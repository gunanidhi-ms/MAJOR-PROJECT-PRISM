"""
tests/test_template_engine.py
------------------------------
Unit tests for the deterministic template engine.

Run with:
    pytest tests/test_template_engine.py -v
"""

import pytest

from app.models import (
    Anomaly,
    AnomalyShape,
    DensityHU,
    DimensionsMM,
    OrganFinding,
    StructuredFindings,
)
from services.template_engine import TemplateEngine, TemplateResult


@pytest.fixture
def engine() -> TemplateEngine:
    return TemplateEngine()


# ====================================================================== #
#  Helper factories
# ====================================================================== #


def make_normal_organ(organ: str = "left_kidney", **kwargs) -> OrganFinding:
    return OrganFinding(
        organ=organ,
        status="normal",
        dimensions_mm=DimensionsMM(**{"length": 98, "width": 45, "ap": 40}),
        mean_density_hu=32,
        location="left renal fossa",
        **kwargs,
    )


def make_findings(organs: list) -> StructuredFindings:
    return StructuredFindings(
        study_id="TEST001",
        modality="CT",
        protocol="CT KUB",
        organs=organs,
    )


# ====================================================================== #
#  Tests: Normal organ
# ====================================================================== #


class TestNormalOrgan:
    def test_normal_organ_contains_name(self, engine):
        findings = make_findings([make_normal_organ("left_kidney")])
        result = engine.generate(findings)
        assert "left kidney" in result.findings_text.lower()

    def test_normal_organ_contains_dimensions(self, engine):
        findings = make_findings([make_normal_organ()])
        result = engine.generate(findings)
        assert "98" in result.findings_text
        assert "45" in result.findings_text
        assert "40" in result.findings_text

    def test_normal_organ_contains_density(self, engine):
        findings = make_findings([make_normal_organ()])
        result = engine.generate(findings)
        assert "32" in result.findings_text
        assert "HU" in result.findings_text

    def test_normal_organ_contains_normal_status(self, engine):
        findings = make_findings([make_normal_organ()])
        result = engine.generate(findings)
        assert "normal" in result.findings_text.lower()

    def test_normal_organ_location(self, engine):
        findings = make_findings([make_normal_organ()])
        result = engine.generate(findings)
        assert "left renal fossa" in result.findings_text.lower()

    def test_returns_template_result_type(self, engine):
        findings = make_findings([make_normal_organ()])
        result = engine.generate(findings)
        assert isinstance(result, TemplateResult)


# ====================================================================== #
#  Tests: Not visualised organ
# ====================================================================== #


class TestNotVisualisedOrgan:
    def test_not_visualised_text(self, engine):
        organ = OrganFinding(organ="pancreas", status="not_visualised")
        findings = make_findings([organ])
        result = engine.generate(findings)
        assert "not visualised" in result.findings_text.lower()
        assert "pancreas" in result.findings_text.lower()


# ====================================================================== #
#  Tests: Single anomaly
# ====================================================================== #


class TestSingleAnomaly:
    def _make_anomaly_finding(self) -> StructuredFindings:
        anomaly = Anomaly(
            anomaly_id="A1",
            type="lesion",
            location="lower pole",
            shape=AnomalyShape(
                volume_cc=4.2,
                long_axis_mm=22.4,
                short_axis_mm=14.1,
                sphericity=0.71,
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
        return make_findings([organ])

    def test_anomaly_type_in_text(self, engine):
        result = engine.generate(self._make_anomaly_finding())
        assert "lesion" in result.findings_text.lower()

    def test_anomaly_location_in_text(self, engine):
        result = engine.generate(self._make_anomaly_finding())
        assert "lower pole" in result.findings_text.lower()

    def test_anomaly_measurements_in_text(self, engine):
        result = engine.generate(self._make_anomaly_finding())
        assert "22.4" in result.findings_text
        assert "14.1" in result.findings_text

    def test_anomaly_volume_in_text(self, engine):
        result = engine.generate(self._make_anomaly_finding())
        assert "4.2" in result.findings_text

    def test_anomaly_density_in_text(self, engine):
        result = engine.generate(self._make_anomaly_finding())
        assert "18" in result.findings_text

    def test_anomaly_confidence_in_text(self, engine):
        result = engine.generate(self._make_anomaly_finding())
        assert "94" in result.findings_text  # 0.94 → 94%

    def test_anomaly_margin_in_text(self, engine):
        result = engine.generate(self._make_anomaly_finding())
        assert "well-defined" in result.findings_text.lower()

    def test_anomaly_sphericity_in_text(self, engine):
        result = engine.generate(self._make_anomaly_finding())
        assert "0.71" in result.findings_text

    def test_density_range_in_text(self, engine):
        result = engine.generate(self._make_anomaly_finding())
        assert "-10" in result.findings_text
        assert "45" in result.findings_text

    def test_organ_name_in_text(self, engine):
        result = engine.generate(self._make_anomaly_finding())
        assert "right kidney" in result.findings_text.lower()


# ====================================================================== #
#  Tests: Multiple organs
# ====================================================================== #


class TestMultipleOrgans:
    def test_two_organs_both_appear(self, engine):
        findings = make_findings([
            make_normal_organ("left_kidney"),
            make_normal_organ("right_kidney"),
        ])
        result = engine.generate(findings)
        assert "left kidney" in result.findings_text.lower()
        assert "right kidney" in result.findings_text.lower()

    def test_organ_section_count(self, engine):
        findings = make_findings([
            make_normal_organ("left_kidney"),
            make_normal_organ("right_kidney"),
            OrganFinding(organ="pancreas", status="not_visualised"),
        ])
        result = engine.generate(findings)
        assert len(result.organ_sections) == 3

    def test_non_kidney_organ_works(self, engine):
        """Ensure the engine is not kidney-specific."""
        organ = OrganFinding(
            organ="liver",
            status="normal",
            dimensions_mm=DimensionsMM(**{"length": 168, "width": 142, "ap": 122}),
            mean_density_hu=52,
            location="right upper quadrant",
        )
        findings = make_findings([organ])
        result = engine.generate(findings)
        assert "liver" in result.findings_text.lower()
        assert "168" in result.findings_text


# ====================================================================== #
#  Tests: Multiple anomalies
# ====================================================================== #


class TestMultipleAnomalies:
    def _make_multi_anomaly_findings(self) -> StructuredFindings:
        anomalies = [
            Anomaly(
                anomaly_id="A1",
                type="cyst",
                location="upper pole",
                shape=AnomalyShape(long_axis_mm=15.6, short_axis_mm=10.3),
                density_hu=DensityHU(mean=8),
                confidence=0.97,
            ),
            Anomaly(
                anomaly_id="A2",
                type="calculus",
                location="mid pole",
                shape=AnomalyShape(long_axis_mm=6.2, short_axis_mm=4.8),
                density_hu=DensityHU(mean=820),
                confidence=0.99,
            ),
        ]
        organ = OrganFinding(
            organ="right_kidney",
            status="anomaly_detected",
            dimensions_mm=DimensionsMM(**{"length": 108, "width": 52, "ap": 46}),
            mean_density_hu=35,
            location="right renal fossa",
            anomalies=anomalies,
        )
        return make_findings([organ])

    def test_all_anomaly_types_present(self, engine):
        result = engine.generate(self._make_multi_anomaly_findings())
        assert "cyst" in result.findings_text.lower()
        assert "calculus" in result.findings_text.lower()

    def test_all_anomaly_measurements_present(self, engine):
        result = engine.generate(self._make_multi_anomaly_findings())
        assert "15.6" in result.findings_text
        assert "10.3" in result.findings_text
        assert "6.2" in result.findings_text
        assert "4.8" in result.findings_text

    def test_all_densities_present(self, engine):
        result = engine.generate(self._make_multi_anomaly_findings())
        assert "8" in result.findings_text
        assert "820" in result.findings_text


# ====================================================================== #
#  Tests: Malformed / minimal input
# ====================================================================== #


class TestEdgeCases:
    def test_organ_without_dimensions(self, engine):
        organ = OrganFinding(
            organ="gallbladder",
            status="normal",
        )
        findings = make_findings([organ])
        result = engine.generate(findings)
        assert "gallbladder" in result.findings_text.lower()
        # No crash, no dimensions in output
        assert "×" not in result.findings_text

    def test_organ_without_density(self, engine):
        organ = OrganFinding(
            organ="spleen",
            status="normal",
            dimensions_mm=DimensionsMM(**{"length": 112, "width": 68, "ap": 52}),
        )
        findings = make_findings([organ])
        result = engine.generate(findings)
        assert "spleen" in result.findings_text.lower()
        assert "HU" not in result.findings_text

    def test_anomaly_without_shape(self, engine):
        anomaly = Anomaly(
            anomaly_id="A1",
            type="lesion",
            location="hilum",
        )
        organ = OrganFinding(
            organ="right_kidney",
            status="anomaly_detected",
            anomalies=[anomaly],
        )
        findings = make_findings([organ])
        result = engine.generate(findings)
        assert "lesion" in result.findings_text.lower()
        assert "hilum" in result.findings_text.lower()
