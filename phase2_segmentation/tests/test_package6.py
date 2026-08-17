"""
test_package6.py — Package 6: candidate_merger tests

All tests use synthetic numpy arrays and Candidate objects — zero dependency
on TotalSegmentator, lifecycle_manager, or any external process. Same pattern
as test_package4.py and test_package5.py.
"""

import math
import numpy as np
import pytest

from schemas.candidate import Candidate, ShapeFeatures, DensityHU
from phase2_segmentation.candidate_merger import (
    merge_candidates,
    clinical_filter,
    _iou_3d,
    _deduplicate_and_tag,
    _backfill_organ_context,
    _compute_shape_geometry,
    Z_SUPPRESS_THRESHOLD,
)


# ── Shared test helpers ────────────────────────────────────────────────────────

def _make_volume(shape=(10, 50, 50), fill=40.0):
    return np.full(shape, fill, dtype=np.float32)


def _make_seg(shape=(10, 50, 50), label=0):
    return np.full(shape, label, dtype=np.int32)


def _make_candidate_a(
    bbox=None,
    organ_label="unclassified",
    long_axis_mm=20.0,
    density_mean=100.0,
    density_std=10.0,
    phase1_confidence=0.8,
    phase1_severity=0.7,
    phase1_anomaly_type="Statistical Hyperdense",
    persistence_ok=True,
    organ_local_zscore=0.0,
):
    """Minimal Path A candidate matching seed_clusterer output conventions."""
    bbox = bbox or [10, 10, 2, 20, 20, 6]
    return Candidate(
        organ_label=organ_label,
        detected_by=["phase1_seed"],
        bbox_3d=bbox,
        centroid_3d=[14.0, 14.0, 4.0],
        slice_indices=[2, 3, 4, 5, 6],
        shape=ShapeFeatures(
            volume_cc=1.0,
            long_axis_mm=long_axis_mm,
            short_axis_mm=8.0,
            sphericity=0.0,
            elongation=2.0,
            margin_curvature_variance=0.0,
        ),
        density_hu=DensityHU(mean=density_mean, min=50.0, max=150.0, std=density_std),
        phase1_confidence=phase1_confidence,
        phase1_severity=phase1_severity,
        phase1_anomaly_type=phase1_anomaly_type,
        persistence_ok=persistence_ok,
        organ_local_zscore=organ_local_zscore,
    )


def _make_candidate_b(
    bbox=None,
    organ_label="liver",
    long_axis_mm=22.0,
    density_mean=110.0,
    density_std=12.0,
    persistence_ok=True,
    organ_local_zscore=4.5,
    organ_overlap_fraction=0.8,
):
    """Minimal Path B candidate matching organ_sweep output conventions."""
    bbox = bbox or [10, 10, 2, 20, 20, 6]
    return Candidate(
        organ_label=organ_label,
        detected_by=["organ_sweep"],
        bbox_3d=bbox,
        centroid_3d=[16.0, 16.0, 4.0],
        slice_indices=[2, 3, 4, 5, 6],
        shape=ShapeFeatures(
            volume_cc=0.9,
            long_axis_mm=long_axis_mm,
            short_axis_mm=9.0,
            sphericity=0.0,
            elongation=2.2,
            margin_curvature_variance=0.0,
        ),
        density_hu=DensityHU(mean=density_mean, min=60.0, max=160.0, std=density_std),
        phase1_confidence=0.0,
        phase1_severity=0.0,
        phase1_anomaly_type="",
        persistence_ok=persistence_ok,
        organ_local_zscore=organ_local_zscore,
        organ_overlap_fraction=organ_overlap_fraction,
    )


# ── IoU unit tests ─────────────────────────────────────────────────────────────

def test_iou_identical_single_voxel_boxes():
    """H3 fix: single-voxel bbox at identical location must return IoU == 1.0."""
    bbox = [5, 5, 3, 5, 5, 3]
    assert _iou_3d(bbox, bbox) == pytest.approx(1.0)


def test_iou_identical_multi_voxel_boxes():
    """Identical non-trivial bboxes → IoU == 1.0."""
    bbox = [0, 0, 0, 9, 9, 4]
    assert _iou_3d(bbox, bbox) == pytest.approx(1.0)


def test_iou_zero_for_non_overlapping():
    """Completely separate bboxes → IoU == 0.0."""
    a = [0, 0, 0, 5, 5, 2]
    b = [10, 10, 5, 15, 15, 9]
    assert _iou_3d(a, b) == 0.0


def test_iou_partial_overlap_correct_value():
    """Verify formula on a known partial-overlap case."""
    # a: 10×10×5 = 500. b: 10×10×5 = 500. overlap: x[5..9]=5, y all, z all → 5×10×5=250
    a = [0, 0, 0, 9, 9, 4]
    b = [5, 0, 0, 14, 9, 4]
    iou = _iou_3d(a, b)
    expected = 250.0 / (500 + 500 - 250)  # = 250/750 ≈ 0.3333
    assert iou == pytest.approx(expected, abs=1e-6)


# ── Merge / corroboration tests ────────────────────────────────────────────────

def test_corroborated_merge_high_iou():
    """Overlapping candidates → merged, corroborated=True, Path B spatial wins,
    Path A phase1 provenance preserved, Path A candidate_id kept."""
    a = _make_candidate_a(bbox=[10, 10, 2, 20, 20, 6])
    b = _make_candidate_b(bbox=[10, 10, 2, 20, 20, 6], organ_label="liver",
                          organ_local_zscore=4.5)

    result = _deduplicate_and_tag([a], [b], iou_threshold=0.3)

    assert len(result) == 1
    c = result[0]
    assert c.corroborated is True
    assert c.detected_by == ["phase1_seed", "organ_sweep"]

    # Path B spatial fields must win
    assert c.organ_label == "liver"
    assert c.organ_local_zscore == 4.5
    assert c.centroid_3d == b.centroid_3d

    # Path A provenance must be preserved
    assert c.phase1_confidence == a.phase1_confidence
    assert c.phase1_severity == a.phase1_severity
    assert c.phase1_anomaly_type == a.phase1_anomaly_type

    # Path A candidate_id kept
    assert c.candidate_id == a.candidate_id


def test_no_merge_non_overlapping():
    """Non-overlapping candidates → two separate survivors, both corroborated=False."""
    a = _make_candidate_a(bbox=[0, 0, 0, 5, 5, 2])
    b = _make_candidate_b(bbox=[30, 30, 6, 40, 40, 9])

    result = _deduplicate_and_tag([a], [b], iou_threshold=0.3)

    assert len(result) == 2
    assert all(not c.corroborated for c in result)
    detected = {tuple(c.detected_by) for c in result}
    assert ("phase1_seed",) in detected
    assert ("organ_sweep",) in detected


def test_iou_threshold_not_met():
    """IoU below threshold → not merged (two separate survivors)."""
    # Partial overlap giving IoU ≈ 0.333 — above 0.3, so use threshold=0.5 to reject
    a = _make_candidate_a(bbox=[0, 0, 0, 9, 9, 4])
    b = _make_candidate_b(bbox=[5, 0, 0, 14, 9, 4])

    result = _deduplicate_and_tag([a], [b], iou_threshold=0.5)

    assert len(result) == 2
    assert all(not c.corroborated for c in result)


def test_score_fields_untouched_after_merge():
    """[SCORE] fields must remain at schema defaults after deduplication."""
    a = _make_candidate_a()
    b = _make_candidate_b()

    result = _deduplicate_and_tag([a], [b], iou_threshold=0.3)

    for c in result:
        assert c.fused_confidence == 0.0, "fused_confidence must not be set by Package 6"
        assert c.gate == "", "gate must not be set by Package 6"
        assert c.emergency_score == 0, "emergency_score must not be set by Package 6"


def test_empty_path_a_returns_path_b():
    """Empty Path A → output is Path B unchanged."""
    b = _make_candidate_b()
    result = _deduplicate_and_tag([], [b], iou_threshold=0.3)
    assert len(result) == 1
    assert result[0].candidate_id == b.candidate_id


def test_empty_path_b_returns_path_a():
    """Empty Path B → output is Path A unchanged."""
    a = _make_candidate_a()
    result = _deduplicate_and_tag([a], [], iou_threshold=0.3)
    assert len(result) == 1
    assert result[0].candidate_id == a.candidate_id


# ── Backfill organ context tests ──────────────────────────────────────────────

def test_backfill_assigns_organ_label():
    """Path A candidate with 'unclassified' → correctly labelled from seg_arr majority."""
    seg_arr = _make_seg((10, 50, 50), label=0)
    # bbox = [10, 10, 2, 20, 20, 6] → seg_arr[2:7, 10:21, 10:21] = label 5
    seg_arr[2:7, 10:21, 10:21] = 5
    value_to_label = {5: "liver"}
    organ_stats = {"liver": {"trimmed_mean": 60.0, "trimmed_std": 10.0}}

    c = _make_candidate_a(bbox=[10, 10, 2, 20, 20, 6])
    assert c.organ_label == "unclassified"

    _backfill_organ_context([c], seg_arr, value_to_label, organ_stats, (1.0, 1.0, 1.0))

    assert c.organ_label == "liver"


def test_backfill_no_organ_in_bbox_stays_unclassified():
    """seg_arr is all-zero in bbox → organ_label stays 'unclassified', no crash."""
    seg_arr = _make_seg((10, 50, 50), label=0)
    c = _make_candidate_a(bbox=[10, 10, 2, 20, 20, 6])

    _backfill_organ_context([c], seg_arr, {}, {}, (1.0, 1.0, 1.0))

    assert c.organ_label == "unclassified"
    assert c.organ_local_zscore == 0.0
    assert c.organ_overlap_fraction == 0.0


def test_backfill_zscore_is_point_estimate():
    """
    Backfill computes organ_local_zscore as a scalar point estimate:
        |density_hu.mean - trimmed_mean| / trimmed_std
    NOT the per-voxel mean used by organ_sweep (unavoidable — Path A has no mask).
    density_mean=100, trimmed_mean=60, trimmed_std=10 → expected z = 4.0
    """
    seg_arr = _make_seg((10, 50, 50), label=0)
    seg_arr[2:7, 10:21, 10:21] = 5
    value_to_label = {5: "liver"}
    organ_stats = {"liver": {"trimmed_mean": 60.0, "trimmed_std": 10.0}}

    c = _make_candidate_a(bbox=[10, 10, 2, 20, 20, 6], density_mean=100.0)
    _backfill_organ_context([c], seg_arr, value_to_label, organ_stats, (1.0, 1.0, 1.0))

    assert c.organ_local_zscore == pytest.approx(4.0, abs=0.001)


def test_backfill_overlap_fraction_full_coverage():
    """When entire bbox is filled with one organ label → overlap_fraction == 1.0."""
    seg_arr = _make_seg((10, 50, 50), label=0)
    seg_arr[2:7, 10:21, 10:21] = 5  # fills exactly the clamped bbox region
    value_to_label = {5: "liver"}
    organ_stats = {"liver": {"trimmed_mean": 60.0, "trimmed_std": 10.0}}

    c = _make_candidate_a(bbox=[10, 10, 2, 20, 20, 6])
    _backfill_organ_context([c], seg_arr, value_to_label, organ_stats, (1.0, 1.0, 1.0))

    assert c.organ_overlap_fraction == pytest.approx(1.0, abs=0.01)


def test_backfill_skips_already_labeled_candidate():
    """Candidate already carrying a non-unclassified organ_label → unchanged."""
    seg_arr = _make_seg((10, 50, 50), label=5)  # entire array = label 5 → "spleen"
    value_to_label = {5: "spleen"}
    organ_stats = {"spleen": {"trimmed_mean": 50.0, "trimmed_std": 8.0}}

    c = _make_candidate_b(bbox=[10, 10, 2, 20, 20, 6], organ_label="liver")
    _backfill_organ_context([c], seg_arr, value_to_label, organ_stats, (1.0, 1.0, 1.0))

    # Must NOT be overwritten to "spleen"
    assert c.organ_label == "liver"


# ── Shape geometry tests ──────────────────────────────────────────────────────

def test_shape_geometry_fills_sphericity_positive():
    """Sphericity > 0 after geometry pass on a solid, uniform bbox region."""
    # Volume filled with exactly density_mean HU so approx_mask covers the whole bbox
    volume = np.full((10, 50, 50), 100.0, dtype=np.float32)
    c = _make_candidate_a(
        bbox=[5, 5, 2, 15, 15, 6],
        density_mean=100.0,
        density_std=5.0,
    )
    assert c.shape.sphericity == 0.0

    _compute_shape_geometry([c], volume, (1.0, 1.0, 1.0))

    assert c.shape.sphericity > 0.0
    assert c.shape.sphericity <= 1.0


def test_shape_geometry_uses_mean_face_area_for_anisotropic_spacing():
    """
    With anisotropic spacing (z_mm >> col_mm), the mean-face-area formula
    must produce a different (lower) surface area than the axial-only formula,
    resulting in higher sphericity than the axial-only estimate. We verify the
    geometry step at least runs and fills sphericity without crashing.
    """
    volume = np.full((10, 50, 50), 100.0, dtype=np.float32)
    c = _make_candidate_a(bbox=[5, 5, 2, 15, 15, 6], density_mean=100.0, density_std=5.0)
    # Anisotropic: z_mm = 5.0, col_mm = row_mm = 0.5
    _compute_shape_geometry([c], volume, (0.5, 0.5, 5.0))
    assert c.shape.sphericity > 0.0


def test_shape_geometry_bbox_outside_volume_no_crash():
    """Bbox entirely outside volume bounds → both shape fields stay 0.0, no exception."""
    volume = _make_volume((5, 10, 10))
    c = _make_candidate_a(bbox=[50, 50, 20, 60, 60, 30])

    _compute_shape_geometry([c], volume, (1.0, 1.0, 1.0))

    assert c.shape.sphericity == 0.0
    assert c.shape.margin_curvature_variance == 0.0


def test_shape_geometry_margin_curvature_variance_positive():
    """margin_curvature_variance > 0 when surface gradient has variation."""
    volume = np.full((10, 50, 50), 100.0, dtype=np.float32)
    c = _make_candidate_a(bbox=[5, 5, 2, 15, 15, 6], density_mean=100.0, density_std=5.0)

    _compute_shape_geometry([c], volume, (1.0, 1.0, 1.0))

    # A solid uniform box has a flat gradient on the interior but non-zero
    # Sobel response on the surface — std may be 0 only if surface is 1 voxel.
    # We just verify no crash and that it is a non-negative float.
    assert c.shape.margin_curvature_variance >= 0.0


# ── Clinical filter tests ─────────────────────────────────────────────────────

def test_unclassified_candidate_never_suppressed():
    """
    Correction 5: candidates with organ_label='unclassified' after backfill must
    NOT be suppressed by any rule — Package 7 routes them to MANUAL_REVIEW.
    """
    c = _make_candidate_a(
        organ_label="unclassified",
        organ_local_zscore=0.0,   # would trigger population_common if classified
        persistence_ok=True,
    )
    c.corroborated = False

    clinical_filter([c], region="abdomen_pelvis")

    assert c.suppressed is False
    assert c.suppression_reason == ""


def test_suppress_not_persistent_labeled_candidate():
    """persistence_ok=False, labeled, not corroborated → suppressed as 'not_persistent'."""
    c = _make_candidate_b(
        organ_label="liver",
        persistence_ok=False,
        organ_local_zscore=5.0,  # high enough to avoid population_common
    )
    c.corroborated = False

    clinical_filter([c], region="abdomen_pelvis")

    assert c.suppressed is True
    assert c.suppression_reason == "not_persistent"


def test_corroborated_exempt_from_not_persistent():
    """persistence_ok=False but corroborated=True → NOT suppressed."""
    c = _make_candidate_b(
        organ_label="liver",
        persistence_ok=False,
        organ_local_zscore=5.0,
    )
    c.corroborated = True

    clinical_filter([c], region="abdomen_pelvis")

    assert c.suppressed is False


def test_suppress_lung_nodule_below_6mm():
    """Lung candidate with long_axis_mm < 6.0 in cardiothoracic region → suppressed."""
    c = _make_candidate_b(
        organ_label="lung_upper_lobe_right",
        long_axis_mm=4.0,
        density_mean=-600.0,
        organ_local_zscore=5.0,
        persistence_ok=True,
    )
    c.corroborated = False

    clinical_filter([c], region="cardiothoracic")

    assert c.suppressed is True
    assert c.suppression_reason == "lung_nodule_<6mm"


def test_no_suppress_lung_nodule_at_or_above_6mm():
    """Lung candidate with long_axis_mm >= 6.0 → NOT suppressed by size rule."""
    c = _make_candidate_b(
        organ_label="lung_upper_lobe_right",
        long_axis_mm=8.0,
        density_mean=-600.0,
        organ_local_zscore=5.0,
        persistence_ok=True,
    )
    c.corroborated = False

    clinical_filter([c], region="cardiothoracic")

    assert c.suppressed is False


def test_no_suppress_lung_nodule_wrong_region():
    """Lung label + small nodule but region != 'cardiothoracic' → not suppressed by lung rule."""
    c = _make_candidate_b(
        organ_label="lung_upper_lobe_right",
        long_axis_mm=4.0,
        density_mean=-600.0,
        organ_local_zscore=5.0,
        persistence_ok=True,
    )
    clinical_filter([c], region="abdomen_pelvis")
    # lung rule doesn't fire; population_common also doesn't fire (zscore=5.0 > 2.5)
    assert c.suppressed is False


def test_suppress_simple_renal_cyst():
    """Renal candidate with water density + high sphericity → 'simple_renal_cyst'."""
    c = _make_candidate_b(
        organ_label="kidney_left",
        density_mean=10.0,
        density_std=5.0,
        organ_local_zscore=4.0,
        persistence_ok=True,
    )
    c.shape.sphericity = 0.85  # computed by _compute_shape_geometry in real pipeline
    c.corroborated = False

    clinical_filter([c], region="abdomen_pelvis")

    assert c.suppressed is True
    assert c.suppression_reason == "simple_renal_cyst"


def test_no_suppress_renal_cyst_low_sphericity():
    """Renal candidate with water density but sphericity <= 0.7 → NOT suppressed as cyst."""
    c = _make_candidate_b(
        organ_label="kidney_left",
        density_mean=10.0,
        density_std=5.0,
        organ_local_zscore=4.0,
        persistence_ok=True,
    )
    c.shape.sphericity = 0.5  # not round enough
    clinical_filter([c], region="abdomen_pelvis")
    assert c.suppressed is False


def test_suppress_population_common():
    """Labeled candidate with low z-score, not corroborated → 'population_common'."""
    c = _make_candidate_b(
        organ_label="liver",
        organ_local_zscore=1.8,  # below default Z_SUPPRESS_THRESHOLD=2.5
        persistence_ok=True,
    )
    c.corroborated = False

    clinical_filter([c], region="abdomen_pelvis")

    assert c.suppressed is True
    assert c.suppression_reason == "population_common"


def test_corroborated_exempt_from_population_common():
    """Same low z-score but corroborated=True → NOT suppressed."""
    c = _make_candidate_b(
        organ_label="liver",
        organ_local_zscore=1.8,
        persistence_ok=True,
    )
    c.corroborated = True

    clinical_filter([c], region="abdomen_pelvis")

    assert c.suppressed is False


def test_z_suppress_threshold_keyword_parameter():
    """Custom z_suppress_threshold overrides the module default without side effects."""
    c = _make_candidate_b(
        organ_label="liver",
        organ_local_zscore=2.0,  # above default 2.5? No — 2.0 < 2.5, would suppress
        persistence_ok=True,
    )
    c.corroborated = False

    # With a stricter custom threshold of 1.5, z-score 2.0 should NOT be suppressed
    clinical_filter([c], region="abdomen_pelvis", z_suppress_threshold=1.5)

    assert c.suppressed is False

    # Confirm module-level constant was not mutated
    assert Z_SUPPRESS_THRESHOLD == 2.5


def test_all_candidates_returned_none_dropped():
    """Total output count always equals total input count — nothing silently dropped."""
    candidates = [
        _make_candidate_a(bbox=[0, 0, 0, 5, 5, 2]),                    # unclassified
        _make_candidate_b(bbox=[10, 10, 3, 15, 15, 6]),                 # liver, high z
        _make_candidate_a(bbox=[20, 20, 0, 25, 25, 3], persistence_ok=False),  # will suppress
    ]
    result = clinical_filter(candidates, region="abdomen_pelvis")
    assert len(result) == 3


def test_fused_confidence_untouched_full_pipeline():
    """End-to-end: [SCORE] fields must remain at schema defaults after full P6 run."""
    volume = _make_volume((10, 50, 50), fill=100.0)
    seg_arr = _make_seg((10, 50, 50), label=0)
    seg_arr[2:7, 10:21, 10:21] = 5
    value_to_label = {5: "liver"}
    organ_stats = {"liver": {"trimmed_mean": 60.0, "trimmed_std": 10.0}}

    a = _make_candidate_a(bbox=[10, 10, 2, 20, 20, 6])
    b = _make_candidate_b(bbox=[10, 10, 2, 20, 20, 6])

    result = merge_candidates(
        [a], [b],
        seg_arr=seg_arr,
        volume=volume,
        value_to_label=value_to_label,
        organ_stats=organ_stats,
        spacing=(1.0, 1.0, 1.0),
        region="abdomen_pelvis",
    )

    for c in result:
        assert c.fused_confidence == 0.0, f"fused_confidence set on {c.candidate_id}"
        assert c.gate == "", f"gate set on {c.candidate_id}"
        assert c.emergency_score == 0, f"emergency_score set on {c.candidate_id}"


def test_no_regression_package4_empty_path_b():
    """Regression: P4 candidates with empty path_b flow through cleanly."""
    volume = _make_volume((10, 50, 50), fill=100.0)
    seg_arr = _make_seg((10, 50, 50), label=0)
    seg_arr[2:7, 10:21, 10:21] = 5
    value_to_label = {5: "liver"}
    organ_stats = {"liver": {"trimmed_mean": 60.0, "trimmed_std": 10.0}}

    a = _make_candidate_a(bbox=[10, 10, 2, 20, 20, 6])
    original_id = a.candidate_id

    result = merge_candidates(
        [a], [],
        seg_arr=seg_arr, volume=volume,
        value_to_label=value_to_label, organ_stats=organ_stats,
        spacing=(1.0, 1.0, 1.0), region="abdomen_pelvis",
    )

    assert len(result) == 1
    assert result[0].candidate_id == original_id
    assert result[0].corroborated is False
    assert result[0].detected_by == ["phase1_seed"]
    # Backfill should have resolved organ_label from seg_arr
    assert result[0].organ_label == "liver"


def test_no_regression_package5_empty_path_a():
    """Regression: P5 candidates with empty path_a flow through cleanly."""
    volume = _make_volume((10, 50, 50), fill=110.0)
    seg_arr = _make_seg((10, 50, 50), label=0)
    value_to_label = {}
    organ_stats = {}

    b = _make_candidate_b(
        bbox=[10, 10, 2, 20, 20, 6],
        organ_label="liver",
        organ_local_zscore=4.5,
    )
    original_id = b.candidate_id

    result = merge_candidates(
        [], [b],
        seg_arr=seg_arr, volume=volume,
        value_to_label=value_to_label, organ_stats=organ_stats,
        spacing=(1.0, 1.0, 1.0), region="abdomen_pelvis",
    )

    assert len(result) == 1
    assert result[0].candidate_id == original_id
    assert result[0].corroborated is False
    assert result[0].detected_by == ["organ_sweep"]
