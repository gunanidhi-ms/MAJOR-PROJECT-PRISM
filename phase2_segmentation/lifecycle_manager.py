

import os
import sys
import gc
import time
import shutil
import logging
import threading
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
import urllib.request
import json
import numpy as np

# Silence harmless WinError 10054 on Windows asyncio ProactorEventLoop
if sys.platform == "win32":
    try:
        import asyncio
        import asyncio.proactor_events
        _orig_call_connection_lost = asyncio.proactor_events._ProactorBasePipeTransport._call_connection_lost

        def _safe_call_connection_lost(self, exc):
            try:
                _orig_call_connection_lost(self, exc)
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
                pass

        asyncio.proactor_events._ProactorBasePipeTransport._call_connection_lost = _safe_call_connection_lost
    except Exception:
        pass

logger = logging.getLogger(__name__)



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
    region: str = "unknown"
    candidates: list = field(default_factory=list)
    candidates_path: str = ""
    findings_path: str = ""


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

    def _emit_event(self, phase: str, status: str, estimated_time_sec: float = 0.0, study_id: str = ""):
        """Helper to post progress events to the Phase 1 WebSocket server."""
        try:
            payload = json.dumps({
                "type": "pipeline_event",
                "phase": phase,
                "status": status,
                "estimated_time_sec": estimated_time_sec,
                "study_id": study_id
            }).encode('utf-8')
            
            req = urllib.request.Request("http://localhost:8001/api/events", data=payload, headers={'Content-Type': 'application/json'})
            urllib.request.urlopen(req, timeout=1.0)
        except Exception as e:
            logger.debug(f"Failed to emit pipeline event '{status}': {e}")

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
        findings_per_slice: Optional[list] = None,
        instance_numbers: Optional[list] = None,
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
            findings_per_slice: Per-slice Phase 1 findings from VolumeAccumulator.
            instance_numbers: Spatially-ordered DICOM instance numbers (same Z
                order as the assembled volume). Used to build instance→z mapping
                for the organ postfilter.

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
        
        # Broadcast that Phase 2 has started
        self._emit_event(
            phase="segmentation", 
            status="started", 
            estimated_time_sec=45.0, 
            study_id=series_uid
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
            result.segmentation_dir, ts_stats = self._segment(
                result.nifti_path, seg_dir, modality
            )
            result.segmentation_time_sec = time.perf_counter() - t_seg_start

            logger.info(
                "Stage 2 complete (segmentation): %.2fs",
                result.segmentation_time_sec,
            )

            # ── Stage 3: Patient-Specific Baselines & Region Detection ──
            seg_nii_path = os.path.join(self._work_dir, "temp_seg.nii")
            if not os.path.isfile(seg_nii_path):
                seg_nii_path = os.path.join(seg_dir, "temp_seg.nii")

            if os.path.isfile(seg_nii_path):
                import SimpleITK as sitk
                from phase2_segmentation.organ_baseline import compute_organ_baseline
                from phase2_segmentation.region_detector import detect_region
                try:
                    from totalsegmentator.map_to_binary import class_map
                    task_key = "total_mr" if modality.upper() in ("MR", "MRI") else "total"
                    cmap = class_map.get(task_key, {})
                except ImportError:
                    cmap = {}
                    logger.warning("Could not import totalsegmentator class_map")

                seg_img = sitk.ReadImage(seg_nii_path)
                seg_arr = sitk.GetArrayFromImage(seg_img)

                voxel_volume_cc = float(spacing[0] * spacing[1] * spacing[2] / 1000.0)
                
                # Compute patient-specific baselines
                patient_stats = {}
                present_labels = []
                
                unique_vals = np.unique(seg_arr)
                for val in unique_vals:
                    if val == 0:
                        continue
                    label = cmap.get(val, f"organ_{val}")
                    mask = (seg_arr == val)
                    voxels = volume[mask]
                    
                    baseline = compute_organ_baseline(voxels, voxel_volume_cc=voxel_volume_cc)
                    patient_stats[label] = baseline
                    
                    if baseline["voxel_count"] > 0:
                        present_labels.append(label)
                
                result.organ_stats = patient_stats
                result.num_organs_detected = len(patient_stats)
                result.region = detect_region(present_labels)

                # ── Region agreement logging ──
                # Log (phase1_guessed_region, package3_actual_region) per series
                # so logs/region_agreement.csv accumulates a confusion matrix for
                # empirically recalibrating _step_2B_body_region thresholds.
                # phase1_guessed_region was captured at trigger time in pipeline.py
                # (majority-vote across all slices) and threaded through series_meta
                # so it survives the multiprocessing boundary.
                phase1_guess = (series_meta or {}).get("phase1_guessed_region", "")
                if phase1_guess:
                    try:
                        from phase1_ingestion.region_agreement_log import log_region_agreement
                        log_region_agreement(
                            series_uid=series_uid,
                            phase1_guess=phase1_guess,
                            package3_actual=result.region,
                        )
                        logger.info(
                            "Region agreement logged: phase1='%s' vs actual='%s' "
                            "(match=%s)",
                            phase1_guess, result.region,
                            phase1_guess.lower() == result.region.lower(),
                        )
                    except Exception as _log_err:
                        logger.warning("Failed to log region agreement: %s", _log_err)

                # Write rich patient-specific statistics and region to disk exclusively in seg_dir
                import json
                os.makedirs(seg_dir, exist_ok=True)
                stats_path = os.path.join(seg_dir, "statistics.json")
                try:
                    with open(stats_path, "w") as f:
                        json.dump(patient_stats, f, indent=4)
                    logger.info("Saved rich patient-specific statistics.json to %s", stats_path)
                except Exception as e:
                    logger.warning("Failed to save statistics.json to %s: %s", stats_path, e)

                region_path = os.path.join(seg_dir, "region.txt")
                try:
                    with open(region_path, "w") as f:
                        f.write(result.region)
                    logger.info("Saved classified region to %s", region_path)
                except Exception as e:
                    logger.warning("Failed to save region.txt to %s: %s", region_path, e)

                # ── Stage 4: Path A/B Candidate Generation, Merge & Clinical Filter ──
                # json is already imported above (used for statistics.json) — no re-import needed.
                # seg_arr and cmap are already in scope from Stage 3 above.
                if findings_per_slice:
                    from phase1_ingestion.finding_tracker import link_findings_across_slices
                    from phase2_segmentation.seed_clusterer import cluster_phase1_seeds
                    from phase2_segmentation.organ_sweep import sweep_organs
                    from phase2_segmentation.candidate_merger import merge_candidates

                    t_cand_start = time.perf_counter()
                    tracks = link_findings_across_slices(findings_per_slice)
                    path_a = cluster_phase1_seeds(tracks, volume, spacing)
                    path_b = sweep_organs(volume, seg_arr, cmap, patient_stats, spacing)
                    candidates = merge_candidates(
                        path_a, path_b, seg_arr, volume, cmap, patient_stats,
                        spacing, result.region,
                    )
                    result.candidates = candidates

                    candidates_path = os.path.join(seg_dir, "candidates.json")
                    try:
                        os.makedirs(seg_dir, exist_ok=True)
                        with open(candidates_path, "w") as f:
                            json.dump([c.to_internal_dict() for c in candidates], f, indent=2)
                        result.candidates_path = candidates_path
                        logger.info("Saved %d candidates to %s", len(candidates), candidates_path)
                    except Exception as e:
                        logger.warning("Failed to save candidates.json: %s", e)

                    logger.info(
                        "Stage 4 complete (candidate generation): %.2fs, "
                        "%d path_a, %d path_b, %d merged",
                        time.perf_counter() - t_cand_start,
                        len(path_a), len(path_b), len(candidates),
                    )

                    # ── Stage 5: Confidence Fusion, Scoring & Phase 3 Handoff ──
                    from phase2_segmentation.scorer import score_candidates
                    from phase2_segmentation.phase2_api import export_findings, trigger_phase3

                    t_score_start = time.perf_counter()
                    score_candidates(candidates, result.region)
                    result.findings_path = export_findings(candidates, self._work_dir, series_meta or {}, result.region)

                    anatomy = None
                    try:
                        from phase2_segmentation.anatomy_context import build_anatomy_context
                        anatomy = build_anatomy_context(seg_arr, cmap, spacing, candidates)
                    except Exception as _anat_err:
                        logger.warning(
                            "Anatomy context failed non-fatally (report will omit "
                            "sub-localisation / contacts): %s", _anat_err, exc_info=True,
                        )

                    trigger_phase3(
                        candidates,
                        series_meta or {},
                        organ_stats=result.organ_stats,
                        region=result.region,
                        anatomy=anatomy,
                    )

                    logger.info(
                        "Stage 5 complete (scoring + handoff): %.2fs, %d findings exported",
                        time.perf_counter() - t_score_start,
                        sum(1 for c in candidates if not c.suppressed),
                    )

                    # ── Stage 6: Organ Post-Filter (Zero Hallucination Validation) ──
                    # Uses the nnU-Net segmentation mask from Stage 2 to retroactively
                    # validate every Phase 1 triage finding:
                    #   a) Suppress findings outside all organ masks (fat/air/artifact)
                    #   b) Suppress findings within normal HU range for their organ
                    #   c) Relabel survivors with zero-hallucination labels
                    # Broadcast the validated findings to the frontend as a retro update.
                    try:
                        from phase2_segmentation.organ_postfilter import (
                            postfilter_phase1_findings,
                            build_instance_to_z,
                        )

                        # Build {instance_number → z_index} using the spatially-ordered
                        # instance list from export_volume() — this matches seg_arr's Z axis.
                        ordered_instances = list(instance_numbers) if instance_numbers else []
                        instance_to_z = build_instance_to_z(ordered_instances)

                        # Restructure findings_per_slice into {instance_number: [findings]}
                        # findings_per_slice from VolumeAccumulator is a list of
                        # {instance_number, findings} dicts (one per slice).
                        findings_by_inst: Dict[int, list] = {}
                        for fs in (findings_per_slice or []):
                            if isinstance(fs, dict):
                                inst = fs.get("instance_number", 0)
                                findings_by_inst[inst] = fs.get("findings", [])
                            elif isinstance(fs, (list, tuple)) and len(fs) == 2:
                                inst, flist = fs
                                findings_by_inst[int(inst)] = flist

                        validated_findings = postfilter_phase1_findings(
                            findings_per_instance=findings_by_inst,
                            instance_to_z=instance_to_z,
                            seg_arr=seg_arr,
                            cmap=cmap,
                            organ_stats=patient_stats,
                        )

                        # Persist validated findings alongside other Phase 2 outputs
                        import json as _json
                        vf_path = os.path.join(seg_dir, "validated_phase1_findings.json")
                        try:
                            os.makedirs(seg_dir, exist_ok=True)
                            with open(vf_path, "w") as _f:
                                _json.dump(
                                    {
                                        str(inst): flist
                                        for inst, flist in validated_findings.items()
                                    },
                                    _f, indent=2,
                                )
                            logger.info(
                                "Stage 6 complete: validated Phase 1 findings saved to %s",
                                vf_path,
                            )
                        except Exception as _e:
                            logger.warning("Stage 6: failed to save validated findings: %s", _e)

                        # Broadcast retroactive update to frontend
                        _organ_validated_count = sum(
                            1 for flist in validated_findings.values()
                            for fd in flist
                            if fd.get("organ_validated", False)
                        )
                        self._emit_event(
                            phase="organ_postfilter",
                            status="completed",
                            study_id=series_uid,
                        )
                        logger.info(
                            "Stage 6: %d organ-validated findings broadcast to frontend",
                            _organ_validated_count,
                        )

                    except Exception as _pf_err:
                        logger.warning(
                            "Stage 6 (organ_postfilter) failed non-fatally: %s",
                            _pf_err, exc_info=True,
                        )

                else:
                    logger.warning(
                        "Stage 4 skipped — no findings_per_slice provided to run()"
                    )

                # Move volume and segmentation files into temp_seg directory so everything is inside temp_seg
                try:
                    target_vol = os.path.join(seg_dir, "temp_volume.nii.gz")
                    if os.path.isfile(nifti_path) and os.path.abspath(nifti_path) != os.path.abspath(target_vol):
                        shutil.move(nifti_path, target_vol)
                        result.nifti_path = target_vol
                    
                    target_seg = os.path.join(seg_dir, "temp_seg.nii")
                    if os.path.isfile(seg_nii_path) and os.path.abspath(seg_nii_path) != os.path.abspath(target_seg):
                        shutil.move(seg_nii_path, target_seg)

                    loose_stats = os.path.join(self._work_dir, "statistics.json")
                    if os.path.isfile(loose_stats) and os.path.abspath(loose_stats) != os.path.abspath(stats_path):
                        os.remove(loose_stats)
                    loose_region = os.path.join(self._work_dir, "region.txt")
                    if os.path.isfile(loose_region) and os.path.abspath(loose_region) != os.path.abspath(region_path):
                        os.remove(loose_region)
                except Exception as move_err:
                    logger.warning("Failed to consolidate files into temp_seg dir: %s", move_err)
            else:
                logger.warning(
                    "Segmentation label map temp_seg.nii not found at %s. Baseline stats / region detection fallback to TS stats.",
                    seg_nii_path
                )
                result.organ_stats = ts_stats
                result.num_organs_detected = len(ts_stats)
                result.region = "unknown"

            logger.info(
                "Stage 3 complete (baseline & region): detected region=%s, %d organs characterized",
                result.region,
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
            # Broadcast Phase 2 completion
            if result.success:
                self._emit_event(phase="segmentation", status="completed", study_id=series_uid)
            else:
                self._emit_event(phase="segmentation", status="failed", study_id=series_uid)

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
        """Remove temporary files created during processing if cleanup is enabled."""
        if not self._cleanup_on_success and not self._cleanup_on_failure:
            logger.info("Cleanup disabled — retaining all Phase 2 files in %s", self._work_dir)
            return

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

    result = manager.run(
        volume,
        spacing,
        series_meta,
        findings_per_slice=findings_per_slice,
        instance_numbers=instance_numbers,
    )

    return result


def run_phase2_shm(
    shm_name: str,
    shape: tuple,
    dtype_str: str,
    findings_per_slice: list,
    instance_numbers: list,
    spacing: Tuple[float, float, float],
    series_meta: Optional[Dict[str, str]] = None,
) -> Phase2Result:
    """
    Shared-memory entry point for Phase 2 — called by pipeline.py on Windows.

    Windows multiprocessing uses 'spawn' which pickles all Process args.
    Pickling a 200-400 MB numpy array through a named pipe raises:
        OSError: [Errno 22] Invalid argument — when serializing numpy.ndarray

    Instead, pipeline.py writes the volume to a named SharedMemory block and
    passes only the lightweight (shm_name, shape, dtype) metadata here.
    This function attaches to that block, reconstructs the array in-process,
    calls the normal Phase 2 pipeline, then unlinks the shared memory.

    Args:
        shm_name:           Name of the SharedMemory block created by the parent.
        shape:              Volume shape tuple, e.g. (310, 512, 512).
        dtype_str:          numpy dtype string, e.g. 'float32'.
        findings_per_slice: Per-slice Phase 1 findings list.
        instance_numbers:   DICOM InstanceNumbers in slice order.
        spacing:            (row_mm, col_mm, z_mm) voxel spacing.
        series_meta:        Optional DICOM metadata dict.

    Returns:
        Phase2Result with processing outcomes.
    """
    import multiprocessing.shared_memory as _shm_mod

    # Configure logging for this subprocess
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [Phase2/%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    shm = None
    try:
        # Attach to the shared memory block written by the parent
        shm = _shm_mod.SharedMemory(name=shm_name, create=False)
        volume = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
        # Make a contiguous private copy so we can safely unlink shm early
        volume = np.array(volume, copy=True)
    finally:
        if shm is not None:
            shm.close()
            try:
                shm.unlink()   # Release the OS segment — only call once, here
            except Exception:
                pass           # Already unlinked by parent (non-fatal)

    logger.info(
        "Phase 2 subprocess (shm) started: %d slices, volume shape=%s",
        len(instance_numbers),
        volume.shape,
    )

    manager = Phase2LifecycleManager(
        work_dir=os.path.join("phase2_work", f"run_{int(time.time())}"),
        ram_budget_gb=RAM_BUDGET_GB,
        cleanup_on_success=False,
        cleanup_on_failure=False,
    )

    return manager.run(
        volume,
        spacing,
        series_meta,
        findings_per_slice=findings_per_slice,
        instance_numbers=instance_numbers,
    )
