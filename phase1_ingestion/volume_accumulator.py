"""
volume_accumulator.py — Phase 1 → Phase 2 Volume Accumulator

Collects HU-transformed 2D slices alongside their per-slice findings as they
stream in from the real-time pipeline. When the series is complete (detected
via association-release, slice-count floor, or timeout fallback), exports
a spatially-ordered 3D volume.

CRITICAL: Ordering is by ImagePositionPatient Z-coordinate, NOT by
InstanceNumber.  These can (and routinely do) disagree in clinical data.

Usage:
    from phase1_ingestion.volume_accumulator import VolumeAccumulator

    acc = VolumeAccumulator(min_slices=20, timeout_sec=5.0)
    acc.add(instance_number, hu_array, findings, spacing_meta)

    if acc.ready_on_association_release():
        volume, findings, inst_nums, spacing = acc.export_volume()
"""

import time
import logging
import threading
from typing import List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


class VolumeAccumulator:
    """
    Accumulates HU-transformed slices and per-slice findings for Phase 2
    volume assembly.

    Three independent trigger paths for readiness:
      1. association_release — external signal that DICOM association closed
      2. min_slices floor   — enough slices accumulated for a useful volume
      3. timeout fallback   — no new slice in timeout_sec seconds

    Ordering is always by ImagePositionPatient Z-coordinate (spatial order),
    never by InstanceNumber.
    """

    def __init__(self, min_slices: int = 20, timeout_sec: float = 5.0):
        """
        Args:
            min_slices: Minimum number of slices before the volume is
                        considered potentially ready (floor trigger).
            timeout_sec: Seconds of inactivity after which the timeout
                         fallback fires.
        """
        self._min_slices = min_slices
        self._timeout_sec = timeout_sec
        self._lock = threading.Lock()

        # Storage: keyed by instance_number to prevent duplicates
        self._slices: dict[int, dict] = {}
        # Each entry: {
        #   "hu_array": np.ndarray (2D),
        #   "findings": list[Finding],
        #   "z_coordinate": float,  (from ImagePositionPatient[2])
        #   "instance_number": int,
        #   "spacing_meta": dict,
        # }

        self._last_add_time: float = time.monotonic()
        self._association_released: bool = False
        self._series_instance_uid: str = ""
        self._study_instance_uid: str = ""
        self._modality: str = "CT"

    def add(
        self,
        instance_number: int,
        hu_array: np.ndarray,
        findings: list,
        spacing_meta: dict,
    ) -> None:
        """
        Add a single HU-transformed slice and its findings.

        Args:
            instance_number: DICOM InstanceNumber for this slice.
            hu_array: 2D float32 HU-transformed array.
            findings: List of Finding objects from triage_screen for this slice.
            spacing_meta: Dict with keys:
                - z_coordinate: float (from ImagePositionPatient[2])
                - pixel_spacing: list[float] [row_mm, col_mm]
                - slice_thickness: float
                - image_position_patient: list[float] [x, y, z]
                - image_orientation_patient: list[float] (6 cosines)
                - series_instance_uid: str
                - study_instance_uid: str
                - modality: str
        """
        with self._lock:
            if instance_number in self._slices:
                logger.debug(
                    "VolumeAccumulator: duplicate instance %d — ignoring",
                    instance_number,
                )
                return

            self._slices[instance_number] = {
                "hu_array": hu_array,
                "findings": findings,
                "z_coordinate": spacing_meta.get("z_coordinate", 0.0),
                "instance_number": instance_number,
                "spacing_meta": spacing_meta,
            }

            # Capture series-level metadata from the first slice
            if not self._series_instance_uid:
                self._series_instance_uid = spacing_meta.get(
                    "series_instance_uid", ""
                )
                self._study_instance_uid = spacing_meta.get(
                    "study_instance_uid", ""
                )
                self._modality = spacing_meta.get("modality", "CT")

            self._last_add_time = time.monotonic()

            logger.debug(
                "VolumeAccumulator: added instance %d (z=%.2f, total=%d)",
                instance_number,
                spacing_meta.get("z_coordinate", 0.0),
                len(self._slices),
            )

    def signal_association_release(self) -> None:
        """
        Signal that the DICOM association has been released (all slices
        for this series have been sent by the modality).
        """
        with self._lock:
            self._association_released = True
            logger.info(
                "VolumeAccumulator: association release signaled "
                "(accumulated %d slices)",
                len(self._slices),
            )

    def ready_on_association_release(self) -> bool:
        """
        Check if the volume is ready because the DICOM association was released.

        Returns:
            True if association was released AND we have at least 1 slice.
        """
        with self._lock:
            ready = self._association_released and len(self._slices) > 0
            if ready:
                logger.info(
                    "VolumeAccumulator: READY via association-release "
                    "(%d slices)",
                    len(self._slices),
                )
            return ready

    def ready_on_slice_floor(self) -> bool:
        """
        Check if the volume has accumulated at least min_slices.

        Returns:
            True if we have >= min_slices slices accumulated.
        """
        with self._lock:
            ready = len(self._slices) >= self._min_slices
            if ready:
                logger.info(
                    "VolumeAccumulator: READY via slice-count floor "
                    "(%d >= %d slices)",
                    len(self._slices),
                    self._min_slices,
                )
            return ready

    def ready_on_timeout(self) -> bool:
        """
        Check if the timeout fallback has triggered (no new slice received
        within timeout_sec).

        Returns:
            True if we have slices AND the timeout has elapsed.
        """
        with self._lock:
            if len(self._slices) == 0:
                return False
            elapsed = time.monotonic() - self._last_add_time
            ready = elapsed >= self._timeout_sec
            if ready:
                logger.info(
                    "VolumeAccumulator: READY via timeout fallback "
                    "(%.1fs elapsed, %d slices)",
                    elapsed,
                    len(self._slices),
                )
            return ready

    def is_ready(self) -> bool:
        """
        Check any of the three readiness triggers.

        Returns:
            True if any trigger condition is met.
        """
        return (
            self.ready_on_association_release()
            or self.ready_on_slice_floor()
            or self.ready_on_timeout()
        )

    @property
    def slice_count(self) -> int:
        """Number of slices currently accumulated."""
        with self._lock:
            return len(self._slices)

    def export_volume(
        self,
    ) -> Tuple[np.ndarray, List[list], List[int], Tuple[float, float, float]]:
        """
        Export the accumulated volume, sorted by ImagePositionPatient Z.

        CRITICAL: Orders by Z-coordinate from ImagePositionPatient, NOT by
        InstanceNumber.  These routinely disagree in clinical DICOM data.

        Returns:
            Tuple of:
                - volume: np.ndarray of shape (Z, Y, X) — 3D HU volume
                - ordered_findings: list of lists — findings per slice,
                  in spatial order
                - ordered_instance_numbers: list of int — InstanceNumbers
                  in spatial order
                - spacing: (row_mm, col_mm, z_mm) tuple
        """
        with self._lock:
            if len(self._slices) == 0:
                raise ValueError("VolumeAccumulator is empty — nothing to export")

            # Sort by Z-coordinate (ImagePositionPatient[2]), NOT InstanceNumber
            sorted_entries = sorted(
                self._slices.values(),
                key=lambda e: e["z_coordinate"],
            )

            # Stack into 3D volume
            volume = np.stack(
                [entry["hu_array"] for entry in sorted_entries], axis=0
            )

            ordered_findings = [entry["findings"] for entry in sorted_entries]
            ordered_instance_numbers = [
                entry["instance_number"] for entry in sorted_entries
            ]

            # Compute spacing from the first slice's metadata
            first_meta = sorted_entries[0]["spacing_meta"]
            row_mm = first_meta.get("pixel_spacing", [1.0, 1.0])[0]
            col_mm = first_meta.get("pixel_spacing", [1.0, 1.0])[1]

            # Compute z_mm from actual slice positions if possible
            if len(sorted_entries) >= 2:
                z_positions = [e["z_coordinate"] for e in sorted_entries]
                z_diffs = [
                    abs(z_positions[i + 1] - z_positions[i])
                    for i in range(len(z_positions) - 1)
                ]
                z_mm = float(np.median(z_diffs))
                if z_mm == 0.0:
                    # Fallback to SliceThickness
                    z_mm = first_meta.get("slice_thickness", 1.0) or 1.0
            else:
                z_mm = first_meta.get("slice_thickness", 1.0) or 1.0

            spacing = (float(row_mm), float(col_mm), float(z_mm))

            logger.info(
                "VolumeAccumulator: exported volume shape=%s, "
                "spacing=(%.2f, %.2f, %.2f)mm, %d slices",
                volume.shape,
                *spacing,
                len(sorted_entries),
            )

            return volume, ordered_findings, ordered_instance_numbers, spacing

    def get_series_metadata(self) -> dict:
        """
        Return series-level metadata for the Phase 2 handoff.
        """
        with self._lock:
            return {
                "series_instance_uid": self._series_instance_uid,
                "study_instance_uid": self._study_instance_uid,
                "modality": self._modality,
            }

    def reset(self) -> None:
        """Clear all accumulated data for the next series."""
        with self._lock:
            self._slices.clear()
            self._association_released = False
            self._series_instance_uid = ""
            self._study_instance_uid = ""
            self._modality = "CT"
            self._last_add_time = time.monotonic()
            logger.info("VolumeAccumulator: reset for next series")
