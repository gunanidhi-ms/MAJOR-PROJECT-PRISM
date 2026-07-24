"""
slice_buffer.py — Thread-safe Sorting Buffer for CT Slices

Implements an OrderedDict-based buffer that re-sorts incoming slices by
InstanceNumber. Handles out-of-order arrival (common with network jitter)
and provides a timeout-triggered flush guardrail when expected slices
don't arrive.

Usage:
    buffer = SliceBuffer(expected_count=20)
    buffer.add_slice(instance_number=5, data={...})
    buffer.add_slice(instance_number=3, data={...})
    
    ordered = buffer.get_ordered_slices()
    # Returns slices sorted: [3, 5]
"""

import time
import threading
import logging
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)


class SliceBuffer:
    """
    Thread-safe sorting buffer for CT slices.
    
    Slices arrive out-of-order over the network. This buffer accumulates
    them, keeps them sorted by InstanceNumber, and supports a 
    timeout-triggered flush when the next expected slice doesn't arrive.
    """

    def __init__(self, expected_count: int | None = None):
        """
        Args:
            expected_count: Expected total number of slices in the series.
                           If None, the buffer operates in streaming mode
                           without a completion check.
        """
        self._lock = threading.Lock()
        self._buffer: OrderedDict[int, dict] = OrderedDict()
        self._expected_count = expected_count
        self._last_insert_time: float = time.monotonic()
        self._next_expected: int = 1  # 1-indexed InstanceNumber
        self._flushed_up_to: int = 0  # last flushed instance number

    @property
    def size(self) -> int:
        """Number of slices currently in the buffer."""
        with self._lock:
            return len(self._buffer)

    @property
    def is_complete(self) -> bool:
        """True if all expected slices have been received."""
        if self._expected_count is None:
            return False
        with self._lock:
            return len(self._buffer) >= self._expected_count

    def add_slice(self, instance_number: int, data: dict[str, Any]) -> None:
        """
        Add a slice to the buffer and re-sort by InstanceNumber.
        
        Args:
            instance_number: DICOM InstanceNumber (0020,0013), 1-indexed
            data: Dict containing at minimum:
                  - 'pixel_array': np.ndarray (raw pixel data)
                  - 'rescale_slope': float
                  - 'rescale_intercept': float
        
        Raises:
            ValueError: If instance_number is invalid or already buffered.
        """
        if instance_number < 1:
            raise ValueError(f"InstanceNumber must be >= 1, got {instance_number}")

        with self._lock:
            if instance_number in self._buffer:
                logger.warning(
                    "Duplicate slice %d received — ignoring", instance_number
                )
                return

            self._buffer[instance_number] = data
            self._last_insert_time = time.monotonic()

            # Re-sort the OrderedDict by key (InstanceNumber)
            sorted_items = sorted(self._buffer.items(), key=lambda x: x[0])
            self._buffer = OrderedDict(sorted_items)

            logger.info(
                "Buffered slice %d (buffer size: %d)",
                instance_number,
                len(self._buffer),
            )

    def get_ordered_slices(self) -> list[tuple[int, dict]]:
        """
        Return all buffered slices sorted by InstanceNumber.
        
        Returns:
            List of (instance_number, data) tuples in ascending order.
        """
        with self._lock:
            return list(self._buffer.items())

    def pop_next_ready(self) -> tuple[int, dict] | None:
        """
        Pop and return the next expected slice if it's available.
        
        Advances the internal pointer. Returns None if the next
        expected slice hasn't arrived yet.
        """
        with self._lock:
            target = self._flushed_up_to + 1
            if target in self._buffer:
                data = self._buffer.pop(target)
                self._flushed_up_to = target
                return (target, data)
            return None

    def pop_all_ready(self) -> list[tuple[int, dict]]:
        """
        Pop all consecutive slices starting from the next expected one.
        
        Returns as many slices as are available in sequence.
        Example: if flushed_up_to=3 and buffer has {4, 5, 7},
                 returns [(4, data4), (5, data5)] and stops at 7.
        """
        results = []
        while True:
            item = self.pop_next_ready()
            if item is None:
                break
            results.append(item)
        return results

    def flush_if_stale(self, timeout_seconds: float = 5.0) -> list[tuple[int, dict]]:
        """
        Timeout-Triggered Flush guardrail.
        
        If the expected next slice hasn't arrived within `timeout_seconds`,
        flush whatever is currently buffered (sorted). This prevents the 
        pipeline from stalling indefinitely when slices are lost or the
        series is shorter than expected.
        
        Args:
            timeout_seconds: Max time to wait for the next expected slice.
        
        Returns:
            List of (instance_number, data) tuples if flushed, else empty list.
        """
        with self._lock:
            if len(self._buffer) == 0:
                return []

            elapsed = time.monotonic() - self._last_insert_time
            if elapsed < timeout_seconds:
                return []

            # Flush everything
            logger.warning(
                "Stale flush triggered after %.1fs — releasing %d buffered slices",
                elapsed,
                len(self._buffer),
            )
            items = list(self._buffer.items())
            self._buffer.clear()
            if items:
                self._flushed_up_to = items[-1][0]
            return items

    def peek_buffer_state(self) -> dict:
        """Return a snapshot of the buffer state for diagnostics."""
        with self._lock:
            return {
                "buffered_count": len(self._buffer),
                "instance_numbers": list(self._buffer.keys()),
                "flushed_up_to": self._flushed_up_to,
                "next_expected": self._flushed_up_to + 1,
                "seconds_since_last_insert": round(
                    time.monotonic() - self._last_insert_time, 2
                ),
            }

    def clear(self) -> None:
        """Clear all buffered slices."""
        with self._lock:
            self._buffer.clear()
            self._flushed_up_to = 0
            self._last_insert_time = time.monotonic()
