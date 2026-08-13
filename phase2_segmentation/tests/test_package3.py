"""
test_package3.py — Test Suite for Package 3

Verifies organ statistical baseline computations and body region detection
including majority-vote and overrides.
"""

import math
import logging
import numpy as np
import pytest

from phase2_segmentation.organ_baseline import compute_organ_baseline
from phase2_segmentation.region_detector import detect_region

logger = logging.getLogger(__name__)

# ─── Organ Baseline Tests ────────────────────────────────────────────────────

def test_baseline_math_correctness():
    """
    Test compute_organ_baseline against hand-computed values.
    Synthetic array of 15 values: 10 to 150 (inclusive, step 10).
    
    Hand-calculated values:
      - p5 = 17.0
      - q1 = 45.0
      - median = 80.0
      - q3 = 115.0
      - p95 = 143.0
      - iqr = 70.0
      - trimmed (17.0 <= x <= 143.0) = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
      - trimmed mean = 80.0
      - trimmed std = sqrt(18200 / 13) ≈ 37.4165738677
      - mad = median(|x - 80|) = 40.0
    """
    arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150], dtype=float)
    voxel_vol = 0.5  # 0.5 cc
    
    stats = compute_organ_baseline(arr, voxel_volume_cc=voxel_vol)
    
    assert stats["voxel_count"] == 15
    assert math.isclose(stats["volume_cc"], 7.5, abs_tol=1e-6)
    assert math.isclose(stats["p5"], 17.0, abs_tol=1e-6)
    assert math.isclose(stats["q1"], 45.0, abs_tol=1e-6)
    assert math.isclose(stats["median"], 80.0, abs_tol=1e-6)
    assert math.isclose(stats["q3"], 115.0, abs_tol=1e-6)
    assert math.isclose(stats["p95"], 143.0, abs_tol=1e-6)
    assert math.isclose(stats["iqr"], 70.0, abs_tol=1e-6)
    
    # Trimmed stats check
    assert math.isclose(stats["trimmed_mean"], 80.0, abs_tol=1e-6)
    expected_std = math.sqrt(18200.0 / 13.0)
    assert math.isclose(stats["trimmed_std"], expected_std, abs_tol=1e-6)
    
    # MAD check
    assert math.isclose(stats["mad"], 40.0, abs_tol=1e-6)


def test_empty_baseline():
    """Verify empty voxel array handles gracefully and returns 0.0 values."""
    arr = np.array([], dtype=float)
    stats = compute_organ_baseline(arr, voxel_volume_cc=0.5)
    
    assert stats["voxel_count"] == 0
    assert stats["volume_cc"] == 0.0
    assert stats["trimmed_mean"] == 0.0
    assert stats["trimmed_std"] == 0.0
    assert stats["median"] == 0.0
    assert stats["mad"] == 0.0


def test_tiny_baseline():
    """Verify function computes stats for very tiny arrays (<5 elements) without crashing."""
    arr = np.array([10, 20, 30], dtype=float)
    stats = compute_organ_baseline(arr, voxel_volume_cc=0.1)
    
    assert stats["voxel_count"] == 3
    assert math.isclose(stats["volume_cc"], 0.3, abs_tol=1e-6)
    # Check that it executed percentiles, mean, std without exceptions
    assert stats["median"] == 20.0
    assert stats["trimmed_mean"] > 0.0
    assert stats["trimmed_std"] >= 0.0


# ─── Region Detector Tests ───────────────────────────────────────────────────

@pytest.fixture
def pure_chest_fixture():
    return ["lung_upper_lobe_left", "lung_lower_lobe_right", "heart", "aorta", "trachea"]


@pytest.fixture
def pure_abdomen_fixture():
    return ["liver", "spleen", "kidney_left", "pancreas", "stomach", "gallbladder"]


@pytest.fixture
def mixed_chest_abdomen_fixture():
    # 3 abdominal organs (liver, spleen, kidney_left) + 2 chest organs (heart, aorta)
    return ["liver", "spleen", "kidney_left", "heart", "aorta"]


@pytest.fixture
def head_fixture():
    return ["brain", "skull"]


@pytest.fixture
def empty_fixture():
    return []


def test_pure_chest_classification(pure_chest_fixture):
    assert detect_region(pure_chest_fixture) == "cardiothoracic"


def test_pure_abdomen_classification(pure_abdomen_fixture):
    assert detect_region(pure_abdomen_fixture) == "abdomen_pelvis"


def test_head_classification(head_fixture):
    assert detect_region(head_fixture) == "neuro"


def test_empty_classification(empty_fixture):
    assert detect_region(empty_fixture) == "unknown"


def test_mixed_classification(mixed_chest_abdomen_fixture, caplog):
    """mixed chest+abdomen votes 3 abdomen to 2 chest -> resolves to abdomen_pelvis."""
    with caplog.at_level(logging.INFO):
        region = detect_region(mixed_chest_abdomen_fixture)
        
    assert region == "abdomen_pelvis"
    # Verify that the distribution tally was logged
    assert "Mixed body regions" in caplog.text
    assert "abdomen_pelvis: 3" in caplog.text
    assert "cardiothoracic: 2" in caplog.text


def test_spine_wildcards():
    """Verify that vertebrae_* wildcards successfully map to spine."""
    labels = ["vertebrae_c1", "vertebrae_t12", "vertebrae_l5", "sacrum", "spinal_cord"]
    assert detect_region(labels) == "spine"


def test_ortho_wildcards():
    """Verify that rib_* and extremity bones map to ortho."""
    labels = ["rib_left_1", "rib_right_10", "femur_left", "hip_right", "humerus_left"]
    assert detect_region(labels) == "ortho"


def test_kub_clinical_override():
    """
    Verify KUB override conditions:
      - Kidney present AND bladder present AND liver absent AND spleen absent -> kub
    """
    # 1. KUB Override active
    labels_kub = ["kidney_left", "urinary_bladder"]
    assert detect_region(labels_kub) == "kub"
    
    # 2. Kidney/bladder present, but liver present -> falls back to abdomen_pelvis majority
    labels_abdomen = ["kidney_left", "urinary_bladder", "liver"]
    assert detect_region(labels_abdomen) == "abdomen_pelvis"
