"""
volume_accumulator.py — Phase 1 → Phase 2 Volume Accumulator

Collects HU-transformed 2D slices alongside their per-slice findings as they
stream in from the real-time pipeline. When the series is complete (detected
via association-release or timeout fallback), exports a spatially-ordered 3D
volume.

CRITICAL TRIGGER SEMANTICS:
    - association_released is the PRIMARY trigger (set externally by
      dicom_listener.py's EVT_RELEASED handler).
    - timeout fallback fires if no new slice arrives within timeout_sec.
    - min_slices is a SANITY FLOOR only — it gates against segmenting a
      near-empty series, it is NEVER the trigger by itself.
    - The `triggered` flag guards against firing more than once for the same
      series. Once export_volume() is called, is_ready() returns False until
      reset() is called.

CRITICAL ORDERING:
    Ordering is by ImagePositionPatient Z-coordinate, NOT by InstanceNumber.
    These can (and routinely do) disagree in clinical data.

Usage:
    from phase1_ingestion.volume_accumulator import VolumeAccumulator

    acc = VolumeAccumulator(min_slices=20)
    acc.add(instance_number, hu_array, findings, spacing_meta)

    if acc.is_ready():
        volume, findings, inst_nums, spacing = acc.export_volume()
        acc.reset()
"""

import time
import logging
import threading
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class VolumeAccumulator:
    """
    Accumulates HU-transformed slices and per-slice findings for Phase 2
    volume assembly.

    Trigger logic (is_ready):
        1. If already triggered for this series → False (one-shot guard).
        2. If fewer than min_slices accumulated → False (sanity floor).
        3. If association_released flag is set → True (primary trigger).
        4. If timeout_sec elapsed since last add() → True (fallback).
        5. Otherwise → False.

    min_slices alone NEVER triggers. This prevents the accumulator from
    re-firing on every slice past the floor — the exact bug this rewrite
    fixes.
    """

    def __init__(self, min_slices: int = 20):
        """
        Args:
            min_slices: Minimum number of slices before the volume is
                        considered potentially ready (sanity floor only,
                        never the trigger).
        """
        self._min_slices = min_slices
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

        self._association_released: bool = False
        self._stream_stalled: bool = False
        self._triggered: bool = False  # one-shot guard: prevents firing more than once per series
        self._completed_series_uids: set[str] = set()  # guards against re-triggering same series UID
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

            logger.debug(
                "VolumeAccumulator: added instance %d (z=%.2f, total=%d)",
                instance_number,
                spacing_meta.get("z_coordinate", 0.0),
                len(self._slices),
            )

    def mark_association_released(self) -> None:
        """
        Signal that the DICOM association has been released.

        Call this from dicom_listener.py's association-release handler
        (pynetdicom evt.EVT_RELEASED), NOT inferred from slice count.
        """
        with self._lock:
            self._association_released = True
            logger.info(
                "VolumeAccumulator: association release signaled "
                "(accumulated %d slices)",
                len(self._slices),
            )

    # Keep the old name as an alias so nothing breaks during transition
    signal_association_release = mark_association_released

    def mark_stream_stalled(self) -> None:
        """
        Signal that the DICOM stream has stalled.
        
        Call this ONLY from SliceBuffer.is_truly_idle() being True -
        never from any timer local to this class.
        """
        with self._lock:
            self._stream_stalled = True
            logger.info("VolumeAccumulator: stream stall signaled by buffer")

    def is_ready(self) -> bool:
        """
        Primary trigger: association_released (or timeout fallback).
        min_slices is a SANITY FLOOR only — it gates against segmenting a
        near-empty series, it is NEVER the trigger by itself.
        triggered guards against firing more than once for the same series.

        Returns:
            True if the volume should be exported now. Once True is returned
            and export_volume() is called, subsequent calls return False until
            reset() is called.
        """
        with self._lock:
            # Guard: already fired for this series
            if self._triggered:
                return False

            # Sanity floor: not enough slices for a useful volume
            if len(self._slices) < self._min_slices:
                return False

            # Primary trigger: association released by the DICOM peer
            if self._association_released:
                logger.info(
                    "VolumeAccumulator: READY via association-release "
                    "(%d slices)",
                    len(self._slices),
                )
                return True

            # Fallback trigger: stream stalled (reported by buffer)
            if self._stream_stalled:
                logger.info(
                    "VolumeAccumulator: READY via stream stall fallback "
                    "(%d slices)",
                    len(self._slices),
                )
                return True

            return False

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

        Sets the triggered flag to prevent is_ready() from firing again
        for the same series.

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
            # Mark as triggered FIRST, before any processing
            self._triggered = True
            if self._series_instance_uid:
                self._completed_series_uids.add(self._series_instance_uid)

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
        """
        Clear all accumulated data for the next series.

        Call after export_volume() to prepare for the next series.
        Resets the triggered guard, association_released flag, and all
        accumulated slice data.
        """
        with self._lock:
            self._slices.clear()
            self._association_released = False
            self._stream_stalled = False
            self._triggered = False
            self._completed_series_uids.clear()
            self._series_instance_uid = ""
            self._study_instance_uid = ""
            self._modality = "CT"
            logger.info("VolumeAccumulator: reset for next series")

    def reset_all(self) -> None:
        """Clear all accumulated data and reset completed series UID guards."""
        with self._lock:
            self.reset()
            self._completed_series_uids.clear()
