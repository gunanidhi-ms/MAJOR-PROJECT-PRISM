"""
pipeline.py — Phase 1 Orchestrator

Wires together the complete ingestion pipeline:
    DICOM Listener → Slice Buffer → HU Transform → Triage Screen → WebSocket Push

This is the main entry point for Phase 1. It:
  1. Starts the FastAPI/WebSocket server (async)
  2. Starts the DICOM C-STORE SCP listener (background thread)
  3. On each received slice: buffer → HU → triage → push alert
  4. Handles graceful shutdown

Usage:
    python -m phase1_ingestion.pipeline
    
    # Then in another terminal:
    python -m phase1_ingestion.replay_sender
"""

import os
import sys
import time
import signal
import logging
import threading
import asyncio
import multiprocessing
from datetime import datetime, timezone

import numpy as np
import uvicorn

from .slice_buffer import SliceBuffer
from .hu_transform import apply_hu_transform, auto_crop
from .triage_screen import screen_slice
from .dicom_listener import DICOMListener
from .ws_server import app, broadcaster, create_alert_payload
from .volume_accumulator import VolumeAccumulator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Configuration ──────────────────────────────────────────────────────────────

DICOM_PORT = 11112
WS_PORT = 8001
WS_HOST = "0.0.0.0"
BUFFER_FLUSH_TIMEOUT = 5.0  # seconds


class Phase1Pipeline:
    """
    Orchestrates the Phase 1 ingestion and screening pipeline.
    
    Flow:
        C-STORE SCP → SliceBuffer → HU Transform → Triage → WebSocket
    """

    def __init__(
        self,
        dicom_port: int = DICOM_PORT,
        ws_port: int = WS_PORT,
    ):
        self.dicom_port = dicom_port
        self.ws_port = ws_port
        self.buffer = SliceBuffer()
        self.accumulator = VolumeAccumulator(min_slices=20, timeout_sec=5.0)
        self.listener: DICOMListener | None = None
        self._running = False
        self._stats = {
            "slices_received": 0,
            "slices_processed": 0,
            "slices_flagged": 0,
            "total_processing_time_ms": 0.0,
        }

    def _on_slice_received(self, slice_data: dict) -> None:
        """
        Callback from DICOM listener — processes each incoming slice
        through the full pipeline.
        
        This runs in the pynetdicom handler thread.
        """
        t_start = time.perf_counter()
        instance_number = slice_data["instance_number"]

        try:
            # ── Step 1: Buffer the slice ──
            self.buffer.add_slice(instance_number, slice_data)
            self._stats["slices_received"] += 1

            # ── Step 2: Process all ready slices in order ──
            ready_slices = self.buffer.pop_all_ready()

            # If no sequential slices are ready, try stale flush
            if not ready_slices:
                ready_slices = self.buffer.flush_if_stale(BUFFER_FLUSH_TIMEOUT)

            for inst_num, data in ready_slices:
                self._process_single_slice(inst_num, data, t_start)

        except Exception as e:
            logger.error(
                "Pipeline error for slice %d: %s", instance_number, e, exc_info=True
            )

    def _process_single_slice(
        self, instance_number: int, slice_data: dict, t_pipeline_start: float
    ) -> None:
        """Process a single slice through HU → Triage → Alert pipeline."""

        pixel_array = slice_data["pixel_array"]
        slope = slice_data["rescale_slope"]
        intercept = slice_data["rescale_intercept"]

        # ── Step 2: HU Transform (without clip, to preserve padding markers) ──
        hu_array_raw = apply_hu_transform(pixel_array, slope, intercept, clip=False)
        
        # ── Step 2b: Auto-crop scanner padding (e.g. -8192 HU padding) ──
        # Must happen BEFORE clipping, since clip(-1024) makes padding == air
        hu_array = auto_crop(hu_array_raw)
        
        # ── Step 2c: Now clip to valid HU range ──
        import numpy as np
        np.clip(hu_array, -1024.0, 3071.0, out=hu_array)

        # ── Step 3: Store thumbnail for frontend ──
        broadcaster.store_thumbnail(
            instance_number, 
            hu_array, 
            window_center=slice_data.get("window_center"),
            window_width=slice_data.get("window_width")
        )

        # ── Step 4: Triage Screening ──
        body_part_hint = slice_data.get("body_part", "") or slice_data.get("study_description", "")
        triage_result = screen_slice(hu_array, body_part_hint=body_part_hint)

        # ── Step 5: Feed VolumeAccumulator (non-blocking, Package 1) ──
        # Accumulator receives the HU-transformed array and findings
        # for Phase 2 volume assembly. This MUST NOT block the real-time path.
        spacing_meta = {
            "z_coordinate": slice_data.get("z_coordinate", 0.0),
            "pixel_spacing": slice_data.get("pixel_spacing", [1.0, 1.0]),
            "slice_thickness": slice_data.get("slice_thickness", 0.0),
            "image_position_patient": slice_data.get(
                "image_position_patient", [0.0, 0.0, 0.0]
            ),
            "image_orientation_patient": slice_data.get(
                "image_orientation_patient",
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ),
            "series_instance_uid": slice_data.get("series_instance_uid", ""),
            "study_instance_uid": slice_data.get("study_instance_uid", ""),
            "modality": slice_data.get("modality", "CT"),
        }
        self.accumulator.add(
            instance_number,
            hu_array,  # HU-transformed, not raw pixel array
            triage_result.findings,
            spacing_meta,
        )

        # ── Step 6: Create and push alert ──
        total_time_ms = (time.perf_counter() - t_pipeline_start) * 1000
        alert = create_alert_payload(
            slice_id=instance_number,
            sop_uid=slice_data["sop_instance_uid"],
            patient_id=slice_data.get("patient_id", "UNKNOWN"),
            triage_result=triage_result.to_dict(),
            processing_time_ms=total_time_ms,
            window_center=slice_data.get("window_center"),
            window_width=slice_data.get("window_width"),
        )
        broadcaster.push_alert(alert)

        # Update stats
        self._stats["slices_processed"] += 1
        self._stats["total_processing_time_ms"] += total_time_ms
        if triage_result.flagged:
            self._stats["slices_flagged"] += 1

        # Console output
        status = "[!] FLAGGED" if triage_result.flagged else "[OK] clean"
        findings_str = ""
        if triage_result.flagged:
            for f in triage_result.findings:
                findings_str += f"\n     -> {f.anomaly_type}: bbox={f.bbox}, HU={f.mean_hu:.0f}"

        print(
            f"  [{instance_number:3d}] {status}  "
            f"({total_time_ms:.1f}ms total, {triage_result.processing_time_ms:.1f}ms triage)"
            f"{findings_str}"
        )

    def run(self) -> None:
        """
        Start the complete pipeline.
        
        Launches:
          1. FastAPI/WebSocket server on WS_PORT
          2. DICOM SCP listener on DICOM_PORT (background thread)
        
        Blocks until shutdown (Ctrl+C).
        """
        self._running = True

        print("=" * 60)
        print("  PRISM Phase 1 — Live Ingestion & 2D Screening Pipeline")
        print("=" * 60)
        print(f"\n  DICOM SCP port:    {self.dicom_port}")
        print(f"  WebSocket port:    {self.ws_port}")
        print(f"  Frontend URL:      http://localhost:5173")
        print(f"  WebSocket URL:     ws://localhost:{self.ws_port}/ws/alerts")
        print(f"  Health check:      http://localhost:{self.ws_port}/health")
        print(f"\n  To send test data:")
        print(f"    python -m phase1_ingestion.replay_sender")
        print(f"\n{'=' * 60}\n")

        # ── Start DICOM listener in background ──
        self.listener = DICOMListener(
            port=self.dicom_port,
            on_slice_received=self._on_slice_received,
            accumulator=self.accumulator,
        )
        dicom_server = self.listener.start_background()

        # ── Start a background thread for stale buffer flushing ──
        flush_thread = threading.Thread(
            target=self._flush_loop, daemon=True, name="buffer-flush"
        )
        flush_thread.start()

        # ── Run FastAPI server (blocking) ──
        try:
            config = uvicorn.Config(
                app,
                host=WS_HOST,
                port=self.ws_port,
                log_level="warning",
                access_log=False,
            )
            server = uvicorn.Server(config)
            server.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def _flush_loop(self) -> None:
        """Periodically check for stale buffer entries and Phase 2 readiness."""
        while self._running:
            time.sleep(1.0)

            # Flush stale buffer entries (existing behavior)
            stale = self.buffer.flush_if_stale(BUFFER_FLUSH_TIMEOUT)
            t_start = time.perf_counter()
            for inst_num, data in stale:
                self._process_single_slice(inst_num, data, t_start)

            # ── Phase 2 dual-path trigger (Package 1) ──
            # Check if the volume accumulator is ready via any trigger path.
            # The timeout fallback fires here; association-release is signaled
            # externally by the DICOM listener.
            self._check_phase2_trigger()

    def _check_phase2_trigger(self) -> None:
        """
        Check if the volume accumulator is ready and spawn Phase 2.

        Uses multiprocessing.Process (not threading.Thread) for clean
        memory teardown — this matters for TotalSegmentator's C++-backed
        tensor memory in Package 2.
        """
        if self.accumulator.is_ready():
            try:
                volume, findings_per_slice, instance_numbers, spacing = (
                    self.accumulator.export_volume()
                )
                series_meta = self.accumulator.get_series_metadata()

                logger.info(
                    "Phase 2 trigger fired: volume shape=%s, "
                    "%d slices, spacing=%s, series=%s",
                    volume.shape,
                    len(instance_numbers),
                    spacing,
                    series_meta.get("series_instance_uid", "unknown")[:20],
                )

                # Spawn Phase 2 as a separate process.
                # phase2_orchestrator.run is not yet implemented (Package 2+),
                # so we log the trigger and reset for now.
                # When Package 2 is ready, uncomment:
                # multiprocessing.Process(
                #     target=phase2_orchestrator.run,
                #     args=(volume, findings_per_slice, instance_numbers, spacing),
                # ).start()

                print(
                    f"\n  [PHASE2] Trigger fired — "
                    f"volume {volume.shape}, {len(instance_numbers)} slices, "
                    f"spacing {spacing}"
                )

                self.accumulator.reset()

            except Exception as e:
                logger.error(
                    "Phase 2 trigger error: %s", e, exc_info=True
                )

    def shutdown(self) -> None:
        """Graceful shutdown of all components."""
        self._running = False
        if self.listener:
            self.listener.stop()

        print(f"\n{'=' * 60}")
        print(f"  Pipeline Statistics:")
        print(f"    Slices received:  {self._stats['slices_received']}")
        print(f"    Slices processed: {self._stats['slices_processed']}")
        print(f"    Slices flagged:   {self._stats['slices_flagged']}")
        if self._stats["slices_processed"] > 0:
            avg = (
                self._stats["total_processing_time_ms"]
                / self._stats["slices_processed"]
            )
            print(f"    Avg processing:   {avg:.1f}ms per slice")
        print(f"{'=' * 60}\n")


def main():
    """Entry point for the Phase 1 pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy loggers
    logging.getLogger("pynetdicom").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)

    pipeline = Phase1Pipeline()

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n[WARN] Interrupt received, shutting down...")
        pipeline.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    pipeline.run()


if __name__ == "__main__":
    main()
