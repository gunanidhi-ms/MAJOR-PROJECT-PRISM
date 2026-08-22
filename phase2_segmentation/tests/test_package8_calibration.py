import pytest
from schemas.candidate import Candidate, DensityHU, ShapeFeatures
from phase2_segmentation.scorer import score_candidates

@pytest.mark.slow
def test_massive_hemorrhage_emergency_score_high():
    c = Candidate(
        organ_label="brain",
        phase1_confidence=0.9,
        organ_local_zscore=5.0,
        corroborated=True,
        persistence_ok=True,
        shape=ShapeFeatures(sphericity=0.8, volume_cc=40.0),
        density_hu=DensityHU(mean=200.0, std=20.0)
    )
    score_candidates([c], "neuro")
    assert c.emergency_score >= 70

@pytest.mark.slow
def test_tiny_blip_low_confidence():
    c = Candidate(
        organ_label="liver",
        phase1_confidence=0.0,
        organ_local_zscore=1.5,
        corroborated=False,
        persistence_ok=False,
        shape=ShapeFeatures(sphericity=0.2, volume_cc=0.5),
        density_hu=DensityHU(mean=40.0, std=5.0)
    )
    score_candidates([c], "abdomen")
    assert c.fused_confidence < 0.5
    assert c.gate == "MANUAL_REVIEW"

@pytest.mark.slow
def test_corroborated_beats_uncorroborated():
    c1 = Candidate(
        organ_label="kidney",
        phase1_confidence=0.8,
        organ_local_zscore=3.5,
        corroborated=True,
        persistence_ok=True,
        shape=ShapeFeatures(sphericity=0.6, volume_cc=15.0),
        density_hu=DensityHU(std=30.0)
    )
    
    c2 = Candidate(
        organ_label="kidney",
        phase1_confidence=0.0,
        organ_local_zscore=3.5,
        corroborated=False,
        persistence_ok=True,
        shape=ShapeFeatures(sphericity=0.6, volume_cc=15.0),
        density_hu=DensityHU(std=30.0)
    )
    
    score_candidates([c1, c2], "abdomen")
    assert c1.emergency_score > c2.emergency_score

@pytest.mark.slow
def test_false_positive_rate_on_normal_volumes():
    candidates = [
        Candidate(organ_label="liver", organ_local_zscore=1.2, corroborated=False),
        Candidate(organ_label="spleen", organ_local_zscore=0.8, corroborated=False),
        Candidate(organ_label="kidney", organ_local_zscore=1.5, corroborated=False),
        Candidate(organ_label="lung", organ_local_zscore=0.5, corroborated=False),
        Candidate(organ_label="heart", organ_local_zscore=1.0, corroborated=False),
    ]
    
    score_candidates(candidates, "cardiothoracic")
    for c in candidates:
        assert c.gate != "AUTO_CONFIRMED"
