"""
test_hu_transform.py — Unit Tests for HU Transform

Tests:
  1. Correct HU calculation against known values
  2. Vectorized operation (no per-pixel loops)
  3. Performance benchmark: <10ms for 512×512 array
  4. Clipping behavior
  5. Windowing function
"""

import time
import pytest
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from phase1_ingestion.hu_transform import (
    apply_hu_transform,
    apply_window,
    hu_stats,
    HU_MIN,
    HU_MAX,
)


class TestHUTransformCorrectness:
    """Test that HU values are computed correctly."""

    def test_basic_transform(self):
        """HU = slope * pixel + intercept with default LUNA16 params."""
        # Default: slope=1.0, intercept=-1024.0
        pixel = np.array([[0, 100, 1024, 2000]], dtype=np.int16)
        hu = apply_hu_transform(pixel, slope=1.0, intercept=-1024.0, clip=False)

        np.testing.assert_allclose(hu[0, 0], -1024.0)   # air
        np.testing.assert_allclose(hu[0, 1], -924.0)
        np.testing.assert_allclose(hu[0, 2], 0.0)       # water
        np.testing.assert_allclose(hu[0, 3], 976.0)

    def test_custom_slope_intercept(self):
        """Non-default slope and intercept."""
        pixel = np.array([[100, 200]], dtype=np.int16)
        hu = apply_hu_transform(pixel, slope=2.0, intercept=-500.0, clip=False)

        np.testing.assert_allclose(hu[0, 0], -300.0)  # 2*100 + (-500)
        np.testing.assert_allclose(hu[0, 1], -100.0)  # 2*200 + (-500)

    def test_float32_output(self):
        """Output should always be float32."""
        pixel = np.array([[100]], dtype=np.int16)
        hu = apply_hu_transform(pixel)
        assert hu.dtype == np.float32

    def test_shape_preserved(self):
        """Output shape should match input shape."""
        pixel = np.random.randint(-1000, 1000, (512, 512), dtype=np.int16)
        hu = apply_hu_transform(pixel)
        assert hu.shape == (512, 512)

    def test_clipping(self):
        """Values should be clipped to [HU_MIN, HU_MAX] when clip=True."""
        # Very high stored value → should clip to HU_MAX
        pixel = np.array([[5000, -5000]], dtype=np.int16)
        hu = apply_hu_transform(pixel, slope=1.0, intercept=-1024.0, clip=True)

        assert hu[0, 0] <= HU_MAX
        assert hu[0, 1] >= HU_MIN

    def test_no_clipping(self):
        """Values should NOT be clipped when clip=False."""
        pixel = np.array([[5000]], dtype=np.int16)
        hu = apply_hu_transform(pixel, slope=1.0, intercept=-1024.0, clip=False)
        expected = 5000.0 * 1.0 + (-1024.0)
        np.testing.assert_allclose(hu[0, 0], expected)

    def test_luna16_air_value(self):
        """LUNA16 air: stored=0 → HU=-1024 (with default params)."""
        pixel = np.array([[0]], dtype=np.int16)
        hu = apply_hu_transform(pixel, slope=1.0, intercept=-1024.0)
        np.testing.assert_allclose(hu[0, 0], -1024.0)

    def test_luna16_water_value(self):
        """LUNA16 water: stored=1024 → HU=0 (with default params)."""
        pixel = np.array([[1024]], dtype=np.int16)
        hu = apply_hu_transform(pixel, slope=1.0, intercept=-1024.0)
        np.testing.assert_allclose(hu[0, 0], 0.0)


class TestHUTransformPerformance:
    """Benchmark: HU transform should be <10ms for 512×512."""

    def test_performance_512x512(self):
        """Verify <10ms for a full CT slice (512×512)."""
        pixel = np.random.randint(-2000, 4000, (512, 512), dtype=np.int16)

        # Warm up
        apply_hu_transform(pixel)

        # Time 10 runs
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            apply_hu_transform(pixel)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        avg_ms = sum(times) / len(times)
        print(f"\n  HU Transform 512×512: avg={avg_ms:.2f}ms, "
              f"min={min(times):.2f}ms, max={max(times):.2f}ms")
        assert avg_ms < 10.0, f"Too slow: {avg_ms:.2f}ms (target <10ms)"


class TestWindowFunction:
    """Test HU windowing for visualization."""

    def test_lung_window(self):
        """Lung window: center=-600, width=1500."""
        hu = np.array([[-1350, -600, 150]], dtype=np.float32)
        windowed = apply_window(hu, window_center=-600, window_width=1500)

        assert windowed.dtype == np.uint8
        assert windowed[0, 0] == 0      # below window → black
        assert windowed[0, 1] == 127 or windowed[0, 1] == 128  # center → mid-gray
        assert windowed[0, 2] == 255    # above window → white

    def test_output_range(self):
        """Windowed output should be [0, 255]."""
        hu = np.random.uniform(-2000, 2000, (100, 100)).astype(np.float32)
        windowed = apply_window(hu, window_center=0, window_width=400)
        assert windowed.min() >= 0
        assert windowed.max() <= 255


class TestHUStats:
    """Test HU statistics computation."""

    def test_basic_stats(self):
        """Stats on a known array."""
        hu = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
        stats = hu_stats(hu)
        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["median"] == 3.0

    def test_masked_stats(self):
        """Stats with a boolean mask."""
        hu = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
        mask = np.array([[True, False, True, False]])
        stats = hu_stats(hu, mask=mask)
        assert stats["mean"] == 20.0  # (10+30)/2
        assert stats["min"] == 10.0
        assert stats["max"] == 30.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
