"""
dicom_listener.py — pynetdicom C-STORE SCP (DICOM Receiver)

Sets up a DICOM Application Entity as a Storage SCP on port 11112.
On each received CT slice, extracts key tags and pixel data, saves
the raw .dcm to disk, and forwards the data to the processing pipeline
via a callback function.

Usage:
    listener = DICOMListener(on_slice_received=my_callback)
    listener.start()  # blocking
    
    # or in a background thread:
    listener.start_background()
    # ... do other work ...
    listener.stop()
"""

import os
import logging
import threading
from pathlib import Path
from typing import Callable

import numpy as np

try:
    from pynetdicom import AE, evt, StoragePresentationContexts
    from pynetdicom.sop_class import CTImageStorage
except ImportError:
    raise ImportError("pynetdicom is required: pip install pynetdicom")

try:
    import pydicom
except ImportError:
    raise ImportError("pydicom is required: pip install pydicom")

logger = logging.getLogger(__name__)


# Default storage directory for incoming DICOM files
INCOMING_DIR = os.path.join(os.path.dirname(__file__), "sample_dicoms", "incoming")


class DICOMListener:
    """
    DICOM C-STORE SCP that receives CT slices on a configurable port.
    
    On each received dataset:
      1. Extracts InstanceNumber, RescaleSlope, RescaleIntercept, pixel array
      2. Saves raw .dcm to incoming/ directory
      3. Calls the on_slice_received callback with extracted data
    """

    def __init__(
        self,
        port: int = 11112,
        ae_title: str = "PRISM_SCP",
        on_slice_received: Callable[[dict], None] | None = None,
        save_to_disk: bool = True,
        incoming_dir: str = INCOMING_DIR,
    ):
        """
        Args:
            port: TCP port to listen on (default 11112, standard DICOM).
            ae_title: Application Entity title for this SCP.
            on_slice_received: Callback invoked for each received slice.
                              Receives a dict with keys:
                                - instance_number: int
                                - rescale_slope: float
                                - rescale_intercept: float
                                - pixel_array: np.ndarray
                                - sop_instance_uid: str
                                - filepath: str (if saved to disk)
            save_to_disk: Whether to save incoming .dcm files.
            incoming_dir: Directory path for saving incoming files.
        """
        self._port = port
        self._ae_title = ae_title
        self._on_slice_received = on_slice_received
        self._save_to_disk = save_to_disk
        self._incoming_dir = incoming_dir
        self._ae: AE | None = None
        self._server = None
        self._thread: threading.Thread | None = None
        self._slice_count = 0

        if self._save_to_disk:
            os.makedirs(self._incoming_dir, exist_ok=True)

    def _handle_store(self, event):
        """
        Handler for C-STORE requests (evt.EVT_C_STORE).
        
        Extracts metadata and pixel data from the incoming DICOM dataset,
        optionally saves to disk, and forwards to the processing callback.
        """
        ds = event.dataset
        ds.file_meta = event.file_meta

        try:
            # ── Extract key DICOM tags ──
            instance_number = int(getattr(ds, "InstanceNumber", 0))
            rescale_slope = float(getattr(ds, "RescaleSlope", 1.0))
            rescale_intercept = float(getattr(ds, "RescaleIntercept", -1024.0))
            sop_uid = str(ds.SOPInstanceUID)

            # ── Extract True Physical Z-Coordinate ──
            image_position_patient = getattr(ds, "ImagePositionPatient", None)
            z_coordinate = 0.0
            if image_position_patient and len(image_position_patient) >= 3:
                z_coordinate = float(image_position_patient[2])

            # ── Extract Window Center/Width ──
            def _extract_first(val, default):
                if val is None: return default
                if isinstance(val, (list, tuple)) or type(val).__name__ == "MultiValue":
                    return float(val[0])
                return float(val)
                
            window_center = _extract_first(getattr(ds, "WindowCenter", None), None)
            window_width = _extract_first(getattr(ds, "WindowWidth", None), None)

            # ── Extract body region hints ──
            body_part = str(getattr(ds, "BodyPartExamined", ""))
            study_desc = str(getattr(ds, "StudyDescription", ""))
            series_desc = str(getattr(ds, "SeriesDescription", ""))
            patient_id = str(getattr(ds, "PatientID", "UNKNOWN_PATIENT"))

            # ── Extract pixel array ──
            pixel_array = ds.pixel_array.copy()

            self._slice_count += 1
            logger.info(
                "Received slice #%d (InstanceNumber=%d, z_coord=%.2f, shape=%s, "
                "slope=%.2f, intercept=%.2f)",
                self._slice_count,
                instance_number,
                z_coordinate,
                pixel_array.shape,
                rescale_slope,
                rescale_intercept,
            )

            # ── Save to disk ──
            filepath = None
            if self._save_to_disk:
                filename = f"incoming_{instance_number:04d}.dcm"
                filepath = os.path.join(self._incoming_dir, filename)
                ds.save_as(filepath)
                logger.debug("Saved to: %s", filepath)

            # ── Forward to processing pipeline ──
            if self._on_slice_received:
                slice_data = {
                    "instance_number": instance_number,
                    "z_coordinate": z_coordinate,
                    "rescale_slope": rescale_slope,
                    "rescale_intercept": rescale_intercept,
                    "window_center": window_center,
                    "window_width": window_width,
                    "body_part": body_part,
                    "study_description": study_desc,
                    "series_description": series_desc,
                    "pixel_array": pixel_array,
                    "sop_instance_uid": sop_uid,
                    "patient_id": patient_id,
                    "filepath": filepath,
                }
                self._on_slice_received(slice_data)

        except Exception as e:
            logger.error("Error processing received DICOM: %s", e, exc_info=True)

        # Return success status
        return 0x0000

    def start(self) -> None:
        """
        Start the DICOM SCP (blocking). 
        
        Listens on the configured port until stop() is called or 
        KeyboardInterrupt is received.
        """
        self._ae = AE(ae_title=self._ae_title)

        # Accept CT Image Storage and common storage SOP classes
        # Using a broad set of storage contexts for compatibility
        for cx in StoragePresentationContexts:
            self._ae.add_supported_context(cx.abstract_syntax)

        handlers = [(evt.EVT_C_STORE, self._handle_store)]

        logger.info(
            "Starting DICOM C-STORE SCP '%s' on port %d...",
            self._ae_title,
            self._port,
        )
        print(f"[DICOM] Listener '{self._ae_title}' listening on port {self._port}")
        print("   Waiting for incoming CT slices...")
        print("   Press Ctrl+C to stop.\n")

        self._server = self._ae.start_server(
            ("0.0.0.0", self._port),
            evt_handlers=handlers,
            block=True,
        )

    def start_background(self) -> threading.Thread:
        """
        Start the DICOM SCP in a background thread.
        
        Returns:
            The background thread running the server.
        """
        self._ae = AE(ae_title=self._ae_title)
        for cx in StoragePresentationContexts:
            self._ae.add_supported_context(cx.abstract_syntax)

        handlers = [(evt.EVT_C_STORE, self._handle_store)]

        logger.info(
            "Starting DICOM SCP '%s' on port %d (background)...",
            self._ae_title,
            self._port,
        )

        self._server = self._ae.start_server(
            ("0.0.0.0", self._port),
            evt_handlers=handlers,
            block=False,
        )

        print(f"[DICOM] Listener '{self._ae_title}' started on port {self._port} (background)")
        return self._server

    def stop(self) -> None:
        """Gracefully shut down the DICOM SCP."""
        if self._server:
            logger.info("Shutting down DICOM SCP...")
            self._server.shutdown()
            self._server = None
        if self._ae:
            self._ae.shutdown()
            self._ae = None
        print(f"\n[STOP] DICOM Listener stopped. Total slices received: {self._slice_count}")

    @property
    def slice_count(self) -> int:
        """Number of slices received so far."""
        return self._slice_count


if __name__ == "__main__":
    # Standalone mode — just receive and save, no processing
    logging.basicConfig(level=logging.INFO)
    listener = DICOMListener()
    try:
        listener.start()
    except KeyboardInterrupt:
        listener.stop()
