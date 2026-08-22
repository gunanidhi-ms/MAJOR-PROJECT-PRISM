"""
test_findings_adapter.py
------------------------
Unit tests for the Phase 2 to Phase 3 findings adapter.
"""

import pytest
from phase2_segmentation.findings_adapter import to_structured_findings
from schemas.candidate import Candidate, ShapeFeatures, DensityHU

def test_to_structured_findings_basic():
    # Setup mock candidates
    c1 = Candidate(
        candidate_id="CAND-001",
        organ_label="liver",
        phase1_anomaly_type="lesion",
        fused_confidence=0.85,
        shape=ShapeFeatures(
            volume_cc=12.5,
            long_axis_mm=25.0,
            short_axis_mm=18.0,
            sphericity=0.8,
            margin_curvature_variance=10.0
        ),
        density_hu=DensityHU(
            mean=45.0,
            min=10.0,
            max=80.0,
            std=15.0
        )
    )
    
    c2 = Candidate(
        candidate_id="CAND-002",
        organ_label="liver",
        phase1_anomaly_type="cyst",
        fused_confidence=0.92,
        shape=ShapeFeatures(
            volume_cc=5.0,
            long_axis_mm=10.0,
            short_axis_mm=8.0,
            sphericity=0.95,
            margin_curvature_variance=5.0
        ),
        density_hu=DensityHU(
            mean=5.0,
            min=-5.0,
            max=15.0,
            std=2.0
        )
    )
    
    # Supressed candidate should be ignored
    c3 = Candidate(
        candidate_id="CAND-003",
        organ_label="kidney_right",
        suppressed=True
    )
    
    series_meta = {
        "study_instance_uid": "STUDY-123",
        "series_instance_uid": "SERIES-456",
        "modality": "CT",
        "protocol": "CT Abdomen Pelvis"
    }
    
    result = to_structured_findings([c1, c2, c3], series_meta)
    
    assert result["study_id"] == "STUDY-123"
    assert result["modality"] == "CT"
    assert result["protocol"] == "CT Abdomen Pelvis"
    
    organs = result["organs"]
    assert len(organs) == 1
    assert organs[0]["organ"] == "liver"
    assert organs[0]["status"] == "anomaly_detected"
    
    anomalies = organs[0]["anomalies"]
    assert len(anomalies) == 2
    
    assert anomalies[0]["anomaly_id"] == "CAND-001"
    assert anomalies[0]["type"] == "lesion"
    assert anomalies[0]["location"] == "Liver"
    assert anomalies[0]["confidence"] == 0.85
    assert anomalies[0]["shape"]["volume_cc"] == 12.5
    assert anomalies[0]["shape"]["margin"] == "well-defined"
    assert anomalies[0]["density_hu"]["mean"] == 45.0
    
    assert anomalies[1]["type"] == "cyst"

def test_to_structured_findings_empty():
    series_meta = {
        "study_instance_uid": "STUDY-999",
        "modality": "CT"
    }
    
    result = to_structured_findings([], series_meta)
    
    assert result["study_id"] == "STUDY-999"
    organs = result["organs"]
    assert len(organs) == 1
    assert organs[0]["organ"] == "whole_body"
    assert organs[0]["status"] == "not_visualised"
