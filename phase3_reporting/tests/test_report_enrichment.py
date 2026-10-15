"""
tests/test_report_enrichment.py
-------------------------------
Covers the report-enrichment features: indication, technique from DICOM tags,
comparison, sub-location, organ contacts, unclassified-lesion listing,
volume-deviation prominence and grouped impression.
"""

import pytest

from app.models import StructuredFindings
from services.template_engine import TemplateEngine


def _lesion(aid, typ="hyperdense lesion", location="Liver", long=60.0, short=40.0,
            vol=50.0, mean=150.0, sub=None):
    d = {
        "anomaly_id": aid, "type": typ, "location": location,
        "shape": {"long_axis_mm": long, "short_axis_mm": short, "volume_cc": vol},
        "density_hu": {"mean": mean, "min": 100.0, "max": 200.0},
    }
    if sub:
        d["sub_location"] = sub
    return d


@pytest.fixture
def payload():
    return {
        "study_id": "S1", "modality": "CT", "protocol": "CT Abdomen Pelvis",
        "region": "abdomen_pelvis", "patient_age": "021Y", "patient_sex": "M",
        "indication": "abdominal pain",
        "technique": {"manufacturer": "SIEMENS", "model": "Somatom", "kvp": "120",
                      "slice_thickness_mm": 5.0, "contrast_route": "IV"},
        "comparison": {"prior_studies": []},
        "volume_qc": {"slice_count": 120, "z_extent_mm": 600.0, "warnings": []},
        "organs": [
            {"organ": "liver", "status": "anomaly_detected",
             "dimensions_mm": {"volume_cc": 6903.0}, "mean_density_hu": 60.0,
             "adjacent_structures": ["stomach", "portal_vein_and_splenic_vein"],
             "anomalies": [_lesion("a1", sub="lateral, posterior portion")]},
            {"organ": "spleen", "status": "normal",
             "dimensions_mm": {"volume_cc": 0.2}, "mean_density_hu": 50.0},
            {"organ": "kidney_right", "status": "normal",
             "dimensions_mm": {"volume_cc": 3.2}, "mean_density_hu": 30.0},
            {"organ": "kidney_left", "status": "normal",
             "dimensions_mm": {"volume_cc": 36.8}, "mean_density_hu": 30.0},
            {"organ": "lung_upper_lobe_left", "status": "anomaly_detected",
             "anomalies": [_lesion("l1", location="Lung Upper Lobe Left", long=12, short=9,
                                   vol=0.5, sub="peripheral, posterior, upper portion")]},
            {"organ": "lung_lower_lobe_right", "status": "anomaly_detected",
             "anomalies": [_lesion("l2", location="Lung Lower Lobe Right", long=8, short=6, vol=0.2)]},
            {"organ": "unclassified", "status": "anomaly_detected", "unassigned": True,
             "anomalies": [
                 _lesion("u1", location="right paravertebral region, at the T10 level",
                         long=14, short=9, vol=0.7),
                 _lesion("u2", location="left-sided posterior location, at approximately the T11 level",
                         long=10, short=7, vol=0.3),
             ]},
        ],
    }


@pytest.fixture
def text(payload):
    return TemplateEngine().generate(StructuredFindings(**payload)).findings_text


def test_indication_and_age(text):
    assert "21-year-old male patient." in text
    assert "Clinical indication: abdominal pain." in text


def test_missing_indication_is_stated(payload):
    payload.pop("indication")
    t = TemplateEngine().generate(StructuredFindings(**payload)).findings_text
    assert "not provided in the study header" in t


def test_technique_from_header(text):
    assert "SIEMENS Somatom scanner" in text
    assert "slice thickness 5 mm" in text and "120 kVp" in text
    assert "Contrast: route IV (per DICOM header)." in text
    assert "Oral contrast: not recorded" in text
    assert "multiplanar" not in text.lower()   # unverifiable claim removed


def test_comparison_none_and_prior(payload):
    t = TemplateEngine().generate(StructuredFindings(**payload)).findings_text
    assert "No prior comparator study available" in t
    payload["comparison"] = {"prior_studies": [{"archive": "x.zip", "archived_on": "2025-01-02"}]}
    t = TemplateEngine().generate(StructuredFindings(**payload)).findings_text
    assert "archived 2025-01-02" in t and "image comparison" in t


def test_sub_location_and_contacts(text):
    assert "in the peripheral, posterior, upper portion" in text
    assert "in contact with: Stomach, Portal and Splenic Vein" in text
    assert "portal and hepatic venous patency" in text      # liver advisory
    assert "mass effect" in text                              # >= 50 mm lesion


def test_unclassified_is_listed_with_positions_not_repeated(text):
    assert text.count("is identified within the Unclassified") == 0
    assert "1. Hyperdense lesion: right paravertebral region, at the T10 level" in text
    assert "2. Hyperdense lesion: left-sided posterior location" in text
    assert "require radiologist anatomical correlation" in text


def test_volume_deviation_leads_impression(text):
    imp = text.split("IMPRESSION:")[1]
    first_point = imp.split("\n")[1]
    assert first_point.startswith("1. Reduced organ volume")
    assert "Spleen markedly reduced (0.2 cc; reference 60\u2013350 cc)" in first_point
    assert "Right Kidney markedly reduced (3.2 cc" in first_point
    assert "Left Kidney markedly reduced (36.8 cc" in first_point or "Left Kidney reduced (36.8 cc" in first_point
    assert "Enlarged organ volume: Liver markedly enlarged (6903 cc" in imp
    assert "Near-absent measured volume" in imp


def test_impression_grouped_not_templated_per_line(text):
    imp = text.split("IMPRESSION:")[1]
    assert imp.count("Radiological correlation with clinical history recommended.") == 0
    assert "Pulmonary: multiple" in imp and "bilateral" in imp
    assert "Hepatobiliary:" in imp
    assert "Abnormalities span" in imp
    assert "additional focal densities not assigned" in imp
    assert "Relation to the stated indication (abdominal pain)" in imp


def test_all_section_headers_present(text):
    for h in ("INDICATION:", "TECHNIQUE:", "COMPARISON:", "FINDINGS:", "IMPRESSION:", "RECOMMENDATIONS:"):
        assert h in text


def test_fov_truncated_organ_noted(payload):
    payload["organs"][1]["touches_fov_edge"] = True
    t = TemplateEngine().generate(StructuredFindings(**payload)).findings_text
    assert "under-represents its true size" in t
    assert "extend to the edge of the scanned volume" in t


def test_normal_study_unchanged_behaviour():
    p = {"study_id": "N", "modality": "CT",
         "organs": [{"organ": "liver", "status": "normal",
                     "dimensions_mm": {"volume_cc": 1500.0}, "mean_density_hu": 55.0}]}
    t = TemplateEngine().generate(StructuredFindings(**p)).findings_text
    assert "No acute or significant findings identified" in t
    assert "Routine clinical follow-up" in t
