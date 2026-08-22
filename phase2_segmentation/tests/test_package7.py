import os
import json
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from schemas.candidate import Candidate, DensityHU, ShapeFeatures
from phase2_segmentation.scorer import score_candidates, CONFIDENCE_GATE_THRESHOLD
from phase2_segmentation.phase2_api import export_findings, trigger_phase3

def test_corroborated_high_zscore_auto_confirmed():
    c = Candidate(
        organ_label="liver",
        phase1_confidence=0.8,
        organ_local_zscore=7.0, # maxes out at 6.0
        corroborated=True,
        persistence_ok=True,
        shape=ShapeFeatures(sphericity=0.9),
        density_hu=DensityHU(std=50.0)
    )
    score_candidates([c], "abdomen")
    assert c.fused_confidence >= CONFIDENCE_GATE_THRESHOLD
    assert c.gate == "AUTO_CONFIRMED"
    assert c.emergency_score > 70

def test_weak_uncorroborated_manual_review():
    c = Candidate(
        organ_label="liver",
        organ_local_zscore=1.0,
        corroborated=False,
        persistence_ok=False,
    )
    score_candidates([c], "abdomen")
    assert c.fused_confidence < 0.55
    assert c.gate == "MANUAL_REVIEW"
    assert c.emergency_score < 40

def test_suppressed_candidate_untouched():
    c = Candidate(
        organ_label="liver",
        suppressed=True,
        suppression_reason="population_common"
    )
    score_candidates([c], "abdomen")
    assert c.fused_confidence == 0.0
    assert c.gate == ""
    assert c.emergency_score == 0

def test_unclassified_organ_always_manual_review():
    c = Candidate(
        organ_label="unclassified",
        phase1_confidence=1.0,
        organ_local_zscore=10.0,
        corroborated=True,
        persistence_ok=True
    )
    score_candidates([c], "abdomen")
    assert c.gate == "MANUAL_REVIEW"
    assert c.fused_confidence > 0.0

def test_emergency_score_range():
    c = Candidate(
        organ_label="liver",
        phase1_confidence=1.0,
        organ_local_zscore=100.0,
        corroborated=True,
        persistence_ok=True,
        shape=ShapeFeatures(sphericity=1.0, volume_cc=1000.0),
        density_hu=DensityHU(std=1000.0)
    )
    score_candidates([c], "abdomen")
    assert 0 <= c.emergency_score <= 100

def test_corroboration_bonus():
    c1 = Candidate(organ_label="liver", organ_local_zscore=3.0, corroborated=True)
    c2 = Candidate(organ_label="liver", organ_local_zscore=3.0, corroborated=False)
    score_candidates([c1, c2], "abdomen")
    assert c1.emergency_score > c2.emergency_score
    assert c1.fused_confidence > c2.fused_confidence

def test_persistence_bonus():
    c1 = Candidate(organ_label="liver", organ_local_zscore=3.0, persistence_ok=True)
    c2 = Candidate(organ_label="liver", organ_local_zscore=3.0, persistence_ok=False)
    score_candidates([c1, c2], "abdomen")
    assert c1.fused_confidence > c2.fused_confidence

def test_phase1_provenance_contributes():
    c1 = Candidate(organ_label="liver", phase1_confidence=0.9, organ_local_zscore=3.0)
    c2 = Candidate(organ_label="liver", phase1_confidence=0.0, organ_local_zscore=3.0)
    score_candidates([c1, c2], "abdomen")
    assert c1.fused_confidence > c2.fused_confidence

def test_export_findings_excludes_suppressed():
    c1 = Candidate(organ_label="liver", suppressed=False)
    c2 = Candidate(organ_label="kidney", suppressed=True)
    
    with tempfile.TemporaryDirectory() as d:
        out_path = export_findings([c1, c2], d, {}, region="chest")
        with open(out_path, "r") as f:
            data = json.load(f)
        assert data["candidate_count"] == 1
        assert len(data["findings"]) == 1
        assert data["findings"][0]["organ_label"] == "liver"

def test_export_findings_schema_shape():
    c = Candidate(organ_label="liver")
    with tempfile.TemporaryDirectory() as d:
        out_path = export_findings([c], d, {"series_instance_uid": "123"}, region="chest")
        with open(out_path, "r") as f:
            data = json.load(f)
        assert "series_instance_uid" in data
        assert "region" in data
        assert "generated_at" in data
        assert "candidate_count" in data
        assert "findings" in data
        assert data["series_instance_uid"] == "123"
        assert data["region"] == "chest"

@patch('phase2_segmentation.phase2_api.httpx.Client')
def test_trigger_phase3_graceful_failure(mock_client_cls):
    """Ensure trigger_phase3 returns False gracefully when Phase 3 is unreachable."""
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    import httpx
    mock_client.post.side_effect = httpx.ConnectError("Network unreachable")
    mock_client_cls.return_value = mock_client

    os.environ["PHASE3_ENABLED"] = "true"
    result = trigger_phase3([], {}, "http://fake-url")
    assert result is False
    # Restore
    os.environ.pop("PHASE3_ENABLED", None)

def test_score_all_suppressed_empty_export():
    c = Candidate(suppressed=True)
    score_candidates([c], "abdomen")
    with tempfile.TemporaryDirectory() as d:
        out_path = export_findings([c], d, {}, region="abdomen")
        with open(out_path, "r") as f:
            data = json.load(f)
        assert data["candidate_count"] == 0
        assert len(data["findings"]) == 0


