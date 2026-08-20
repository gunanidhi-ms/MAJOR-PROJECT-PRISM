import numpy as np
from dataclasses import dataclass

from phase2_segmentation.seed_clusterer import cluster_phase1_seeds


@dataclass
class FakeFinding:
    bbox: list
    centroid: list
    confidence: float = 0.7
    severity_score: float = 0.5
    anomaly_type: str = "Statistical Hyperdense"


def _make_volume(shape=(10, 50, 50), fill=40.0):
    return np.full(shape, fill, dtype=np.float32)


def test_persistent_track_becomes_candidate():
    volume = _make_volume()
    volume[3:6, 20:25, 20:25] = 200.0

    track = [
        (3, FakeFinding(bbox=[20, 20, 5, 5], centroid=[22.5, 22.5])),
        (4, FakeFinding(bbox=[20, 20, 5, 5], centroid=[22.5, 22.5])),
        (5, FakeFinding(bbox=[20, 20, 5, 5], centroid=[22.5, 22.5])),
    ]

    candidates = cluster_phase1_seeds([track], volume, spacing=(1.0, 1.0, 1.0))

    assert len(candidates) == 1
    c = candidates[0]
    assert c.bbox_3d == [20, 20, 3, 25, 25, 5]
    assert c.centroid_3d == [22.5, 22.5, 4.0]
    assert c.density_hu.mean > 150.0
    assert c.persistence_ok is True
    assert c.detected_by == ["phase1_seed"]


def test_short_track_excluded():
    volume = _make_volume()
    track = [
        (3, FakeFinding(bbox=[20, 20, 5, 5], centroid=[22.5, 22.5])),
        (4, FakeFinding(bbox=[20, 20, 5, 5], centroid=[22.5, 22.5])),
    ]

    candidates = cluster_phase1_seeds([track], volume, spacing=(1.0, 1.0, 1.0))
    assert candidates == []


def test_single_slice_wide_track():
    volume = _make_volume()
    track = [
        (5, FakeFinding(bbox=[10, 10, 2, 2], centroid=[11.0, 11.0])),
        (5, FakeFinding(bbox=[10, 10, 2, 2], centroid=[11.0, 11.0])),
        (5, FakeFinding(bbox=[10, 10, 2, 2], centroid=[11.0, 11.0])),
    ]
    candidates = cluster_phase1_seeds([track], volume, spacing=(1.0, 1.0, 1.0), min_slices=1)
    assert len(candidates) == 1
    assert candidates[0].bbox_3d[2] == candidates[0].bbox_3d[5] == 5
