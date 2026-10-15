"""
Tests for anatomy_context: orientation inference, FOV-edge flag, true organ
contacts, intra-organ sub-location and anatomical description of unassigned lesions.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from phase2_segmentation.anatomy_context import build_anatomy_context

CMAP = {
    1: "vertebrae_T10", 2: "vertebrae_T11", 3: "vertebrae_L1",
    4: "kidney_left", 5: "kidney_right", 6: "liver", 7: "spleen",
}


@pytest.fixture
def seg():
    a = np.zeros((60, 64, 80), dtype=np.uint8)        # (Z, Y, X)
    a[20:36, 46:52, 30:35] = 1                         # T10 (posterior, midline x≈32)
    a[36:50, 46:52, 30:35] = 2                         # T11 (more superior)
    a[2:19, 46:52, 30:35] = 3                          # L1 (inferior)
    a[22:41, 20:41, 38:59] = 4                         # left kidney (high x = patient left)
    a[22:41, 20:41, 6:27] = 5                          # right kidney
    a[0:30, 15:35, 8:25] = 6                           # liver, touches z=0 edge
    a[22:41, 20:41, 59:70] = 7                         # spleen touching left kidney
    return a


def _cand(cid, organ, centroid, bbox):
    return SimpleNamespace(candidate_id=cid, organ_label=organ, suppressed=False,
                           centroid_3d=list(centroid), bbox_3d=list(bbox))


def test_orientation_and_fov_edge(seg):
    ctx = build_anatomy_context(seg, CMAP, (1.0, 1.0, 1.0), [])
    o = ctx["orientation"]
    assert o["left_is_high_x"] and o["posterior_is_high_y"] and o["superior_is_high_z"]
    assert all(o["verified"].values())
    assert ctx["organs"]["liver"]["touches_fov_edge"] is True
    assert ctx["organs"]["kidney_left"]["touches_fov_edge"] is False


def test_flipped_lr_is_detected(seg):
    flipped = seg[:, :, ::-1].copy()                   # mirror x: left kidney now at low x
    ctx = build_anatomy_context(flipped, CMAP, (1.0, 1.0, 1.0), [])
    assert ctx["orientation"]["left_is_high_x"] is False


def test_sub_location_inside_kidney(seg):
    c = _cand("k1", "kidney_left", (56, 38, 38), (54, 36, 36, 58, 40, 40))
    ctx = build_anatomy_context(seg, CMAP, (1.0, 1.0, 1.0), [c])
    phrase = ctx["candidates"]["k1"]["sub_location"]
    assert "lateral" in phrase and "posterior" in phrase and "superior" in phrase


def test_organ_contacts_are_true_contacts(seg):
    c = _cand("k1", "kidney_left", (48, 30, 30), (46, 28, 28, 50, 32, 32))
    ctx = build_anatomy_context(seg, CMAP, (1.0, 1.0, 1.0), [c])
    contacts = ctx["organs"]["kidney_left"]["contacts"]
    assert "spleen" in contacts
    assert "liver" not in contacts and "vertebrae_T10" not in contacts


def test_unclassified_gets_side_level_and_paravertebral(seg):
    c = _cand("u1", "unclassified", (22, 50, 28), (20, 48, 26, 24, 52, 30))
    ctx = build_anatomy_context(seg, CMAP, (1.0, 1.0, 1.0), [c])
    info = ctx["candidates"]["u1"]
    assert info["side"] == "right"
    assert info["level"] == "at the T10 level"
    assert info["paravertebral"] is True
    assert info["description"].startswith("right paravertebral region, at the T10 level")


def test_suppressed_candidates_ignored(seg):
    c = _cand("s1", "kidney_left", (56, 38, 38), (54, 36, 36, 58, 40, 40))
    c.suppressed = True
    ctx = build_anatomy_context(seg, CMAP, (1.0, 1.0, 1.0), [c])
    assert "s1" not in ctx["candidates"]
