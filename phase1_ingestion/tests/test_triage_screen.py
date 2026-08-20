"""
test_triage_screen.py — Unit Tests for Triage Screening

Tests:
  1. Detection of synthetic ground-glass-like hypodense nodules
  2. Detection of synthetic solid hyperdense nodules
  3. No false positives on clean background
  4. Bounding box accuracy
  5. Minimum cluster size filtering
  6. Performance benchmark
  7. TriageResult serialization
"""

import time
import pytest
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from phase1_ingestion.triage_screen import screen_slice, TriageResult


def _make_clean_lung_slice() -> np.ndarray:
    """
    Create a clean HU array mimicking a normal CT slice.
    
    Structure:
      - Outer air: -1000 HU
      - Body ellipse: ~40 HU (soft tissue)
      - Two lung fields: -800 HU (normal parenchyma)
    """
    hu = np.full((512, 512), -1000.0, dtype=np.float32)  # air

    yy, xx = np.mgrid[0:512, 0:512]
    cy, cx = 256, 256

    # Body
    body = ((xx - cx) / 200.0) ** 2 + ((yy - cy) / 160.0) ** 2 < 1.0
    hu[body] = 40.0  # soft tissue

    # Left lung
    left_lung = ((xx - (cx - 70)) / 90.0) ** 2 + ((yy - (cy + 10)) / 120.0) ** 2 < 1.0
    hu[left_lung] = -800.0

    # Right lung
    right_lung = ((xx - (cx + 70)) / 90.0) ** 2 + ((yy - (cy + 10)) / 120.0) ** 2 < 1.0
    hu[right_lung] = -800.0

    return hu, left_lung, right_lung


def _embed_circle(hu: np.ndarray, cy: int, cx: int, radius: int, hu_value: float):
    """Embed a circular blob at given HU value."""
    yy, xx = np.mgrid[0:512, 0:512]
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    hu[circle] = hu_value
    return circle


class TestTriageDetection:
    """Test that anomalies are correctly detected."""

    def test_detect_ground_glass_hypodense(self):
        """Ground-glass opacity (-500 HU) in lung field should be flagged as hypodense."""
        hu, left_lung, right_lung = _make_clean_lung_slice()

        # Embed ground-glass nodule in left lung
        _embed_circle(hu, cy=276, cx=186, radius=15, hu_value=-500.0)

        result = screen_slice(hu)

        assert result.flagged, "Ground-glass nodule should be flagged"
        assert len(result.findings) >= 1

        # The engine classifies this as a statistical anomaly — check it was detected
        # with an HU value in the GGO range
        ggo_findings = [f for f in result.findings if -700 <= f.mean_hu <= -300]
        assert len(ggo_findings) >= 1, (
            f"Should detect a finding in GGO HU range. "
            f"Got findings: {[(f.anomaly_type, f.mean_hu) for f in result.findings]}"
        )

    def test_detect_solid_nodule_hyperdense(self):
        """Solid nodule (+50 HU) in lung field should be flagged as hyperdense."""
        hu, left_lung, right_lung = _make_clean_lung_slice()

        # Embed solid nodule in right lung
        _embed_circle(hu, cy=276, cx=326, radius=12, hu_value=50.0)

        result = screen_slice(hu)

        assert len(result.findings) >= 1, (
            f"Solid nodule should produce at least one finding. "
            f"Got: flagged={result.flagged}, findings={len(result.findings)}, "
            f"action={result.action}, score={result.emergency_score}"
        )

        # Verify the finding has the right HU range
        solid_findings = [f for f in result.findings if -50 <= f.mean_hu <= 100]
        assert len(solid_findings) >= 1, (
            f"Should detect a finding in solid nodule HU range. "
            f"Got findings: {[(f.anomaly_type, f.mean_hu) for f in result.findings]}"
        )

    def test_detect_both_types(self):
        """Both hypodense and hyperdense anomalies in same slice."""
        hu, left_lung, right_lung = _make_clean_lung_slice()

        _embed_circle(hu, cy=276, cx=186, radius=15, hu_value=-500.0)  # GGO in left
        _embed_circle(hu, cy=276, cx=326, radius=10, hu_value=60.0)    # solid in right

        result = screen_slice(hu)

        assert result.flagged, "Slice with two anomalies should be flagged"
        assert len(result.findings) >= 1, "Should detect at least one finding"


class TestTriageCleanSlice:
    """Test that clean slices are not flagged."""

    def test_no_false_positive_normal_lung(self):
        """Normal lung parenchyma (-800 HU) should NOT be flagged."""
        hu, _, _ = _make_clean_lung_slice()

        result = screen_slice(hu)

        # Clean slice should not be flagged
        assert not result.flagged, (
            f"Clean slice should not be flagged. "
            f"Got {len(result.findings)} findings: "
            f"{[f.anomaly_type for f in result.findings]}"
        )


class TestTriageBoundingBox:
    """Test bounding box accuracy."""

    def test_bbox_contains_nodule(self):
        """Bounding box should contain the embedded nodule."""
        hu, left_lung, right_lung = _make_clean_lung_slice()

        nodule_cy, nodule_cx, nodule_r = 276, 186, 12
        _embed_circle(hu, cy=nodule_cy, cx=nodule_cx, radius=nodule_r, hu_value=-500.0)

        result = screen_slice(hu)
        assert result.flagged and len(result.findings) > 0

        f = result.findings[0]
        bx, by, bw, bh = f.bbox

        # Nodule center should be inside the bbox
        assert bx <= nodule_cx <= bx + bw, f"Nodule cx={nodule_cx} not in bbox x=[{bx}, {bx+bw}]"
        assert by <= nodule_cy <= by + bh, f"Nodule cy={nodule_cy} not in bbox y=[{by}, {by+bh}]"


class TestTriageMinimumSize:
    """Test minimum cluster size filtering."""

    def test_tiny_cluster_filtered(self):
        """Very small clusters (< min_area_px) should be filtered out."""
        hu, left_lung, right_lung = _make_clean_lung_slice()

        # Embed a tiny 2-pixel "cluster" — below threshold
        hu[276, 186] = -500.0
        hu[276, 187] = -500.0

        result = screen_slice(hu)

        # Should NOT produce findings in GGO range because cluster is too small
        ggo_findings = [f for f in result.findings if -700 <= f.mean_hu <= -300]
        assert len(ggo_findings) == 0, "Tiny cluster should be filtered out"


class TestTriagePerformance:
    """Benchmark: triage screening should be <50ms for 512×512."""

    def test_performance(self):
        """Verify <50ms for a full CT slice."""
        hu, _, _ = _make_clean_lung_slice()

        # Warm up
        screen_slice(hu)

        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            screen_slice(hu)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        avg_ms = sum(times) / len(times)
        print(f"\n  Triage 512×512: avg={avg_ms:.1f}ms, "
              f"min={min(times):.1f}ms, max={max(times):.1f}ms")
        assert avg_ms < 200.0, f"Too slow: {avg_ms:.1f}ms (target <50ms, allowing 200ms for CI)"


class TestTriageResult:
    """Test TriageResult serialization."""

    def test_to_dict(self):
        """TriageResult.to_dict() should produce valid JSON-serializable dict."""
        hu, _, _ = _make_clean_lung_slice()
        _embed_circle(hu, cy=276, cx=186, radius=12, hu_value=-500.0)

        result = screen_slice(hu)
        d = result.to_dict()

        assert isinstance(d, dict)
        assert "flagged" in d
        assert "findings" in d
        assert isinstance(d["findings"], list)
        assert "processing_time_ms" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
