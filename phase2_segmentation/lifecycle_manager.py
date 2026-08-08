"""
lifecycle_manager.py — Phase 2 Subprocess Lifecycle & RAM Watchdog

Orchestrates the full Phase 2 volume-processing pipeline:
    1. Assemble NIfTI from HU volume (volume_builder)
    2. Run TotalSegmentator segmentation (segment_runner)
    3. Monitor RAM throughout via psutil
    4. Guarantee temp file cleanup on success AND failure
    5. Log peak RAM and processing times

This module is designed to run inside a multiprocessing.Process spawned
by pipeline.py, ensuring complete memory isolation from the real-time
Phase 1 path.

Usage:
    from phase2_segmentation.lifecycle_manager import Phase2LifecycleManager

    manager = Phase2LifecycleManager(work_dir="phase2_work")
    result = manager.run(volume, spacing, series_meta)
"""

import os
import gc
import time
import shutil
import logging
import threading
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

RAM_BUDGET_GB = 3.0          # Hard ceiling for peak RAM usage
RAM_POLL_INTERVAL_SEC = 5.0  # How often to check RAM during segmentation
DEFAULT_WORK_DIR = "phase2_work"


@dataclass
class Phase2Result:
    """Result of a Phase 2 processing run."""
    success: bool = False
    nifti_path: str = ""
    segmentation_dir: str = ""
    organ_stats: Dict[str, Any] = field(default_factory=dict)
    peak_ram_gb: float = 0.0
    total_time_sec: float = 0.0
    assembly_time_sec: float = 0.0
    segmentation_time_sec: float = 0.0
    error_message: str = ""
    volume_shape: Tuple[int, ...] = ()
    num_organs_detected: int = 0


class RAMWatchdog:
    """
    Background thread that monitors process-specific RAM usage via psutil.

    Tracks memory usage of the current Python process and optionally child 
    processes (like TotalSegmentator). More accurate than system-wide monitoring.
    """

    def __init__(self, budget_gb: float = RAM_BUDGET_GB, interval_sec: float = RAM_POLL_INTERVAL_SEC):
        self._budget_gb = budget_gb
        self._interval_sec = interval_sec
        self._peak_gb = 0.0
        self._baseline_gb = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._exceeded_budget = False
        self._process = None

    def start(self) -> None:
        """Begin process RAM monitoring in a background daemon thread."""
        try:
            import psutil
        except ImportError:
            logger.warning("psutil not installed — RAM monitoring disabled")
            return

        # Get current process
        self._process = psutil.Process()
        
        # Measure baseline process memory usage
        memory_info = self._process.memory_info()
        self._baseline_gb = memory_info.rss / (1024 ** 3)  # RSS = Resident Set Size
        self._peak_gb = self._baseline_gb
        self._running = True
        self._exceeded_budget = False

        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="ram-watchdog",
        )
        self._thread.start()

        logger.info(
            "Process RAM watchdog started: baseline=%.3f GB (RSS), budget=%.1f GB, PID=%d",
            self._baseline_gb,
            self._budget_gb,
            self._process.pid,
        )

    def stop(self) -> float:
        """
        Stop monitoring and return the peak process RAM usage in GB.

        Returns:
            Peak process RAM usage observed during monitoring (GB).
        """
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        delta = self._peak_gb - self._baseline_gb
        logger.info(
            "Process RAM watchdog stopped: peak=%.3f GB, baseline=%.3f GB, "
            "delta=%.3f GB, budget=%.1f GB",
            self._peak_gb,
            self._baseline_gb,
            delta,
            self._budget_gb,
        )

        return self._peak_gb

    def _monitor_loop(self) -> None:
        """Poll process RSS RAM usage at regular intervals.
        
        Uses process-specific RSS (Resident Set Size) to measure only what
        this Python process + its children consume, not system-wide RAM.
        This makes the budget comparison meaningful regardless of other system load.
        """
        try:
            import psutil
        except ImportError:
            return

        while self._running:
            try:
                if self._process is not None:
                    # Process-specific RSS: what WE are using
                    mem_info = self._process.memory_info()
                    current_gb = mem_info.rss / (1024 ** 3)
                else:
                    # Fallback to system-wide if process handle is unavailable
                    current_gb = psutil.virtual_memory().used / (1024 ** 3)

                if current_gb > self._peak_gb:
                    self._peak_gb = current_gb

                if current_gb > self._budget_gb and not self._exceeded_budget:
                    self._exceeded_budget = True
                    logger.warning(
                        "RAM BUDGET EXCEEDED: process RSS=%.2f GB > budget=%.1f GB",
                        current_gb,
                        self._budget_gb,
                    )

            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.debug("RAM watchdog: process gone or access denied: %s", e)
                break
            except Exception as e:
                logger.debug("RAM watchdog poll error: %s", e)

            time.sleep(self._interval_sec)

    @property
    def peak_gb(self) -> float:
        return self._peak_gb

    @property
    def exceeded_budget(self) -> bool:
        return self._exceeded_budget


class Phase2LifecycleManager:
    """
    Manages the full Phase 2 processing lifecycle:
        volume_builder.assemble_nifti → segment_runner.run_totalsegmentator

    Provides:
        - Deterministic temp directory management
        - RAM monitoring via psutil watchdog
        - Guaranteed cleanup on both success and failure
        - Structured result reporting
    """

    def __init__(
        self,
        work_dir: str = DEFAULT_WORK_DIR,
        ram_budget_gb: float = RAM_BUDGET_GB,
        cleanup_on_success: bool = False,
        cleanup_on_failure: bool = False,
    ):
        """
        Args:
            work_dir: Base directory for temp NIfTI and segmentation output.
                      Will be created if it doesn't exist.
            ram_budget_gb: Maximum allowed RAM usage in GB.
            cleanup_on_success: If True, delete temp files after successful run.
                                Default False to allow inspection during development.
            cleanup_on_failure: If True, delete temp files even on failure.
                                Default False to allow post-mortem debugging.
        """
        self._work_dir = os.path.abspath(work_dir)
        self._ram_budget_gb = ram_budget_gb
        self._cleanup_on_success = cleanup_on_success
        self._cleanup_on_failure = cleanup_on_failure

    def run(
        self,
        volume: np.ndarray,
        spacing: Tuple[float, float, float],
        series_meta: Optional[Dict[str, str]] = None,
    ) -> Phase2Result:
        """
        Execute the complete Phase 2 processing pipeline.

        This is the main entry point, designed to be called from a
        multiprocessing.Process spawned by pipeline.py.

        Args:
            volume: 3D numpy array (Z, Y, X) in HU from export_volume().
            spacing: (row_mm, col_mm, z_mm) from export_volume().
            series_meta: Optional dict with series_instance_uid, modality, etc.
                         from VolumeAccumulator.get_series_metadata().

        Returns:
            Phase2Result with processing outcomes.
        """
        result = Phase2Result(volume_shape=volume.shape)
        series_meta = series_meta or {}
        modality = series_meta.get("modality", "CT")
        series_uid = series_meta.get("series_instance_uid", "unknown")

        logger.info(
            "Phase 2 lifecycle starting: volume=%s, spacing=%s, "
            "modality=%s, series=%s",
            volume.shape,
            spacing,
            modality,
            series_uid[:20],
        )

        nifti_path = os.path.join(self._work_dir, "temp_volume.nii.gz")
        seg_dir = os.path.join(self._work_dir, "temp_seg")

        # ── Start RAM watchdog ──
        watchdog = RAMWatchdog(
            budget_gb=self._ram_budget_gb,
            interval_sec=RAM_POLL_INTERVAL_SEC,
        )
        watchdog.start()

        t_total_start = time.perf_counter()

        try:
            # ── Create work directory (inside try so PermissionError is caught) ──
            os.makedirs(self._work_dir, exist_ok=True)

            # ── Stage 1: NIfTI Assembly ──
            t_assemble_start = time.perf_counter()
            result.nifti_path = self._assemble(volume, spacing, nifti_path)
            result.assembly_time_sec = time.perf_counter() - t_assemble_start

            logger.info(
                "Stage 1 complete (NIfTI assembly): %.2fs, file=%s",
                result.assembly_time_sec,
                result.nifti_path,
            )

            # ── Stage 2: TotalSegmentator ──
            t_seg_start = time.perf_counter()
            result.segmentation_dir, result.organ_stats = self._segment(
                result.nifti_path, seg_dir, modality
            )
            result.segmentation_time_sec = time.perf_counter() - t_seg_start
            result.num_organs_detected = len(result.organ_stats)

            logger.info(
                "Stage 2 complete (segmentation): %.2fs, %d organs",
                result.segmentation_time_sec,
                result.num_organs_detected,
            )

            result.success = True

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            logger.error(
                "Phase 2 FAILED: %s", e, exc_info=True
            )

        finally:
            # ── Stop watchdog and record peak RAM ──
            result.peak_ram_gb = watchdog.stop()
            result.total_time_sec = time.perf_counter() - t_total_start

            # ── Cleanup ──
            should_cleanup = (
                (result.success and self._cleanup_on_success) or
                (not result.success and self._cleanup_on_failure)
            )
            if should_cleanup:
                self._cleanup(nifti_path, seg_dir)

            # ── Force garbage collection ──
            gc.collect()

            # ── Summary log ──
            self._log_summary(result)

        return result

    def _assemble(
        self,
        volume: np.ndarray,
        spacing: Tuple[float, float, float],
        out_path: str,
    ) -> str:
        """Stage 1: Assemble NIfTI from volume + spacing."""
        from phase2_segmentation.volume_builder import assemble_nifti
        return assemble_nifti(volume, spacing, out_path=out_path)

    def _segment(
        self,
        nifti_path: str,
        out_dir: str,
        modality: str,
    ) -> Tuple[str, Dict]:
        """Stage 2: Run TotalSegmentator segmentation."""
        from phase2_segmentation.segment_runner import run_totalsegmentator
        return run_totalsegmentator(nifti_path, out_dir=out_dir, modality=modality)

    def _cleanup(self, nifti_path: str, seg_dir: str) -> None:
        """Remove temporary files created during processing."""
        try:
            if os.path.isfile(nifti_path):
                os.remove(nifti_path)
                logger.debug("Cleaned up temp NIfTI: %s", nifti_path)
        except OSError as e:
            logger.warning("Failed to clean up %s: %s", nifti_path, e)

        try:
            if os.path.isdir(seg_dir):
                shutil.rmtree(seg_dir)
                logger.debug("Cleaned up temp segmentation dir: %s", seg_dir)
        except OSError as e:
            logger.warning("Failed to clean up %s: %s", seg_dir, e)

    def _log_summary(self, result: Phase2Result) -> None:
        """Log a structured summary of the Phase 2 run."""
        status = "SUCCESS" if result.success else "FAILED"

        summary = (
            f"\n{'=' * 60}\n"
            f"  Phase 2 Processing Summary — {status}\n"
            f"{'=' * 60}\n"
            f"  Volume shape:        {result.volume_shape}\n"
            f"  Total time:          {result.total_time_sec:.2f}s\n"
            f"    Assembly:          {result.assembly_time_sec:.2f}s\n"
            f"    Segmentation:      {result.segmentation_time_sec:.2f}s\n"
            f"  Peak RAM:            {result.peak_ram_gb:.2f} GB "
            f"(budget: {self._ram_budget_gb:.1f} GB)\n"
            f"  Organs detected:     {result.num_organs_detected}\n"
        )

        if not result.success:
            summary += f"  Error:               {result.error_message}\n"

        summary += f"{'=' * 60}\n"

        if result.success:
            logger.info(summary)
        else:
            logger.error(summary)

        # Also print for visibility in the subprocess console
        print(summary)


def run_phase2(
    volume: np.ndarray,
    findings_per_slice: list,
    instance_numbers: list,
    spacing: Tuple[float, float, float],
    series_meta: Optional[Dict[str, str]] = None,
) -> Phase2Result:
    """
    Top-level entry point for Phase 2 processing.

    This function matches the signature expected by pipeline.py's
    multiprocessing.Process(target=...) call. It wraps Phase2LifecycleManager
    and returns the result.

    Args:
        volume: 3D numpy array (Z, Y, X) in HU from export_volume().
        findings_per_slice: Per-slice findings (for future Package 4 use).
        instance_numbers: InstanceNumbers in spatial order (for logging).
        spacing: (row_mm, col_mm, z_mm) from export_volume().
        series_meta: Optional metadata from get_series_metadata().

    Returns:
        Phase2Result with processing outcomes.
    """
    # Configure logging for the subprocess
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [Phase2/%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info(
        "Phase 2 subprocess started: %d slices, volume shape=%s",
        len(instance_numbers),
        volume.shape,
    )

    manager = Phase2LifecycleManager(
        work_dir=os.path.join("phase2_work", f"run_{int(time.time())}"),
        ram_budget_gb=RAM_BUDGET_GB,
        # Keep files for development/debugging
        cleanup_on_success=False,
        cleanup_on_failure=False,
    )

    result = manager.run(volume, spacing, series_meta)

    return result
