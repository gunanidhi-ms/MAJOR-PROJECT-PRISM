"""
test_slice_buffer.py — Unit Tests for SliceBuffer

Tests:
  1. Out-of-order insertion → correct sorted output
  2. flush_if_stale timeout behavior
  3. pop_next_ready sequential processing
  4. Duplicate slice handling
  5. Thread safety with concurrent inserts
"""

import time
import threading
import pytest
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from phase1_ingestion.slice_buffer import SliceBuffer


class TestSliceBufferOrdering:
    """Test that slices are correctly sorted by InstanceNumber."""

    def test_sorted_order_basic(self):
        """Slices inserted out of order should be retrievable in sorted order."""
        buf = SliceBuffer()
        buf.add_slice(5, {"pixel_array": np.zeros((2, 2)), "rescale_slope": 1.0, "rescale_intercept": -1024.0})
        buf.add_slice(2, {"pixel_array": np.zeros((2, 2)), "rescale_slope": 1.0, "rescale_intercept": -1024.0})
        buf.add_slice(8, {"pixel_array": np.zeros((2, 2)), "rescale_slope": 1.0, "rescale_intercept": -1024.0})
        buf.add_slice(1, {"pixel_array": np.zeros((2, 2)), "rescale_slope": 1.0, "rescale_intercept": -1024.0})
        buf.add_slice(3, {"pixel_array": np.zeros((2, 2)), "rescale_slope": 1.0, "rescale_intercept": -1024.0})

        ordered = buf.get_ordered_slices()
        keys = [k for k, v in ordered]
        assert keys == [1, 2, 3, 5, 8], f"Expected sorted keys, got {keys}"

    def test_sorted_order_reversed(self):
        """Slices arriving in reverse order."""
        buf = SliceBuffer()
        for i in range(10, 0, -1):
            buf.add_slice(i, {"pixel_array": np.zeros((2, 2)), "rescale_slope": 1.0, "rescale_intercept": -1024.0})

        ordered = buf.get_ordered_slices()
        keys = [k for k, v in ordered]
        assert keys == list(range(1, 11)), f"Expected 1-10, got {keys}"

    def test_size_property(self):
        """Size should track buffer contents."""
        buf = SliceBuffer()
        assert buf.size == 0
        buf.add_slice(1, {"pixel_array": np.zeros((2, 2)), "rescale_slope": 1.0, "rescale_intercept": -1024.0})
        assert buf.size == 1
        buf.add_slice(3, {"pixel_array": np.zeros((2, 2)), "rescale_slope": 1.0, "rescale_intercept": -1024.0})
        assert buf.size == 2

    def test_duplicate_slice_ignored(self):
        """Duplicate InstanceNumber should be ignored (not overwrite)."""
        buf = SliceBuffer()
        buf.add_slice(1, {"value": "first", "pixel_array": np.zeros((2, 2)), "rescale_slope": 1.0, "rescale_intercept": -1024.0})
        buf.add_slice(1, {"value": "second", "pixel_array": np.zeros((2, 2)), "rescale_slope": 1.0, "rescale_intercept": -1024.0})
        assert buf.size == 1
        ordered = buf.get_ordered_slices()
        assert ordered[0][1]["value"] == "first"

    def test_invalid_instance_number(self):
        """InstanceNumber < 1 should raise ValueError."""
        buf = SliceBuffer()
        with pytest.raises(ValueError):
            buf.add_slice(0, {"pixel_array": np.zeros((2, 2)), "rescale_slope": 1.0, "rescale_intercept": -1024.0})
        with pytest.raises(ValueError):
            buf.add_slice(-5, {"pixel_array": np.zeros((2, 2)), "rescale_slope": 1.0, "rescale_intercept": -1024.0})


class TestSliceBufferPopReady:
    """Test sequential pop_next_ready processing."""

    def test_pop_sequential(self):
        """Pop returns slices in order when available sequentially."""
        buf = SliceBuffer()
        buf.add_slice(3, {"val": 3, "pixel_array": np.zeros((2, 2)), "rescale_slope": 1.0, "rescale_intercept": -1024.0})
        buf.add_slice(1, {"val": 1, "pixel_array": np.zeros((2, 2)), "rescale_slope": 1.0, "rescale_intercept": -1024.0})
        buf.add_slice(2, {"val": 2, "pixel_array": np.zeros((2, 2)), "rescale_slope": 1.0, "rescale_intercept": -1024.0})

        result = buf.pop_all_ready()
        assert len(result) == 3
        assert [r[0] for r in result] == [1, 2, 3]

    def test_pop_with_gap(self):
        """Pop stops at gaps in the sequence."""
        buf = SliceBuffer()
        buf.add_slice(1, {"val": 1, "pixel_array": np.zeros((2, 2)), "rescale_slope": 1.0, "rescale_intercept": -1024.0})
        buf.add_slice(2, {"val": 2, "pixel_array": np.zeros((2, 2)), "rescale_slope": 1.0, "rescale_intercept": -1024.0})
        buf.add_slice(5, {"val": 5, "pixel_array": np.zeros((2, 2)), "rescale_slope": 1.0, "rescale_intercept": -1024.0})

        result = buf.pop_all_ready()
        assert len(result) == 2
        assert [r[0] for r in result] == [1, 2]
        assert buf.size == 1  # slice 5 still buffered


class TestSliceBufferFlush:
    """Test timeout-triggered flush behavior."""

    def test_no_flush_when_fresh(self):
        """No flush if timeout hasn't elapsed."""
        buf = SliceBuffer()
        buf.add_slice(1, {"pixel_array": np.zeros((2, 2)), "rescale_slope": 1.0, "rescale_intercept": -1024.0})
        result = buf.flush_if_stale(timeout_seconds=10.0)
        assert result == []

    def test_flush_when_stale(self):
        """Flush triggers after timeout elapses."""
        buf = SliceBuffer()
        buf.add_slice(1, {"pixel_array": np.zeros((2, 2)), "rescale_slope": 1.0, "rescale_intercept": -1024.0})
        buf.add_slice(3, {"pixel_array": np.zeros((2, 2)), "rescale_slope": 1.0, "rescale_intercept": -1024.0})

        # Wait for timeout
        time.sleep(0.15)
        result = buf.flush_if_stale(timeout_seconds=0.1)

        assert len(result) == 2
        assert [r[0] for r in result] == [1, 3]
        assert buf.size == 0

    def test_flush_empty_buffer(self):
        """No flush on empty buffer."""
        buf = SliceBuffer()
        result = buf.flush_if_stale(timeout_seconds=0.01)
        assert result == []


class TestSliceBufferThreadSafety:
    """Test concurrent access from multiple threads."""

    def test_concurrent_inserts(self):
        """Multiple threads inserting simultaneously should not corrupt state."""
        buf = SliceBuffer()
        errors = []

        def insert_range(start, end):
            try:
                for i in range(start, end):
                    buf.add_slice(i, {"pixel_array": np.zeros((2, 2)), "rescale_slope": 1.0, "rescale_intercept": -1024.0})
                    time.sleep(0.001)  # tiny delay for interleaving
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=insert_range, args=(1, 11)),
            threading.Thread(target=insert_range, args=(11, 21)),
            threading.Thread(target=insert_range, args=(21, 31)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert buf.size == 30

        ordered = buf.get_ordered_slices()
        keys = [k for k, v in ordered]
        assert keys == list(range(1, 31)), f"Keys not sorted: {keys}"


class TestSliceBufferDiagnostics:
    """Test diagnostic/introspection methods."""

    def test_peek_state(self):
        """peek_buffer_state should return accurate info."""
        buf = SliceBuffer()
        buf.add_slice(3, {"pixel_array": np.zeros((2, 2)), "rescale_slope": 1.0, "rescale_intercept": -1024.0})
        buf.add_slice(1, {"pixel_array": np.zeros((2, 2)), "rescale_slope": 1.0, "rescale_intercept": -1024.0})

        state = buf.peek_buffer_state()
        assert state["buffered_count"] == 2
        assert state["instance_numbers"] == [1, 3]
        assert state["next_expected"] == 1

    def test_clear(self):
        """Clear should reset all state."""
        buf = SliceBuffer()
        buf.add_slice(1, {"pixel_array": np.zeros((2, 2)), "rescale_slope": 1.0, "rescale_intercept": -1024.0})
        buf.add_slice(2, {"pixel_array": np.zeros((2, 2)), "rescale_slope": 1.0, "rescale_intercept": -1024.0})
        buf.clear()
        assert buf.size == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
