import numpy as np

from phase2_segmentation.organ_sweep import sweep_organs
from phase2_segmentation.organ_baseline import compute_organ_baseline


def _make_test_volume_and_mask(shape=(10, 50, 50), seed=42):
    rng = np.random.default_rng(seed)
    volume = np.full(shape, 40.0, dtype=np.float32)
    seg_arr = np.zeros(shape, dtype=np.int32)

    organ_region = (slice(1, 9), slice(5, 45), slice(5, 45))
    seg_arr[organ_region] = 1
    volume[organ_region] += rng.normal(0, 5.0, size=volume[organ_region].shape).astype(np.float32)

    return volume, seg_arr


def test_hyperdense_component_detected():
    volume, seg_arr = _make_test_volume_and_mask()
    volume[3:7, 20:25, 20:25] = 300.0

    baseline = compute_organ_baseline(volume[seg_arr == 1], voxel_volume_cc=1.0)
    candidates = sweep_organs(
        volume, seg_arr, {1: "liver"}, {"liver": baseline},
        spacing=(1.0, 1.0, 1.0), min_component_voxels=10,
    )

    assert len(candidates) == 1
    c = candidates[0]
    assert c.organ_label == "liver"
    assert c.detected_by == ["organ_sweep"]
    assert c.density_hu.mean > 200.0
    assert c.persistence_ok is True
    assert c.organ_overlap_fraction > 0.0
    assert c.organ_local_zscore > 3.0


def test_no_outliers_returns_empty():
    volume, seg_arr = _make_test_volume_and_mask()
    baseline = compute_organ_baseline(volume[seg_arr == 1], voxel_volume_cc=1.0)
    candidates = sweep_organs(
        volume, seg_arr, {1: "liver"}, {"liver": baseline}, spacing=(1.0, 1.0, 1.0),
    )
    assert candidates == []


def test_tiny_component_filtered_out():
    volume, seg_arr = _make_test_volume_and_mask()
    volume[5, 22, 22] = 500.0

    baseline = compute_organ_baseline(volume[seg_arr == 1], voxel_volume_cc=1.0)
    candidates = sweep_organs(
        volume, seg_arr, {1: "liver"}, {"liver": baseline},
        spacing=(1.0, 1.0, 1.0), min_component_voxels=10,
    )
    assert candidates == []


def test_single_slice_component_not_persistent():
    # A 10x10 patch on one slice bleeds slightly into Z±1 via the IQR fence
    # on the noisy background — the resulting component spans 3 slices [4,5,6].
    # Setting min_persistent_slices=4 verifies the persistence gate correctly
    # marks a sub-threshold component as persistence_ok=False.
    volume, seg_arr = _make_test_volume_and_mask()
    volume[5, 15:25, 15:25] = 300.0

    baseline = compute_organ_baseline(volume[seg_arr == 1], voxel_volume_cc=1.0)
    candidates = sweep_organs(
        volume, seg_arr, {1: "liver"}, {"liver": baseline},
        spacing=(1.0, 1.0, 1.0), min_component_voxels=10, min_persistent_slices=4,
    )
    assert len(candidates) == 1
    assert candidates[0].persistence_ok is False