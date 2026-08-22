"""
ws_server.py — FastAPI WebSocket Server for Real-time Alert Push

Provides a WebSocket endpoint for streaming triage alerts to the 
frontend viewer. Also serves slice thumbnails as PNG images.

Endpoints:
    WS  /ws/alerts              — Real-time alert stream
    GET /health                 — Health check
    GET /api/slices/{id}/thumbnail  — PNG thumbnail of processed slice
    GET /api/stats              — Pipeline statistics

Architecture:
    The sync DICOM pipeline pushes alerts into a thread-safe queue.
    An async background task drains the queue and broadcasts to all
    connected WebSocket clients.
"""

import io
import json
import time
import asyncio
import logging
import threading
import base64
import shutil
import glob
import os
from collections import deque
from typing import Any
from datetime import datetime, timezone

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from PIL import Image
except ImportError:
    raise ImportError("Pillow is required: pip install Pillow")

from .hu_transform import apply_window

logger = logging.getLogger(__name__)


# ── Global state (shared between sync pipeline and async server) ──────────────

class AlertBroadcaster:
    """
    Thread-safe bridge between the sync DICOM pipeline and async WebSocket.
    
    The pipeline (running in a sync thread) pushes alerts via push_alert().
    The async broadcaster drains the queue and broadcasts to all connected clients.
    """

    def __init__(self, max_history: int = 100):
        self._queue: asyncio.Queue = None  # set when event loop starts
        self._sync_queue: deque = deque(maxlen=500)  # thread-safe fallback
        self._lock = threading.Lock()
        self._clients: set[WebSocket] = set()
        self._history: deque = deque(maxlen=max_history)
        self._thumbnails: dict[int, bytes] = {}  # slice_id → PNG bytes
        self._hu_arrays: dict[int, np.ndarray] = {}  # slice_id → raw HU array
        self._stats = {
            "total_slices": 0,
            "flagged_slices": 0,
            "connected_clients": 0,
            "start_time": datetime.now(timezone.utc).isoformat(),
        }
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the asyncio event loop (called from the async context)."""
        self._loop = loop
        self._queue = asyncio.Queue()

    def push_alert(self, alert: dict) -> None:
        """
        Push an alert from the sync pipeline thread.
        Thread-safe — can be called from any thread.
        """
        with self._lock:
            # Clean thumbnail for history so it doesn't eat memory
            history_alert = alert.copy()
            if "thumbnail" in history_alert:
                del history_alert["thumbnail"]
            self._history.append(history_alert)
            
            self._stats["total_slices"] += 1
            if alert.get("flagged", False):
                self._stats["flagged_slices"] += 1
                
            # Save a preliminary report for this patient
            self._save_preliminary_report(alert.get("patient_id", "unknown"))

        # Push to async queue via event loop
        if self._loop and self._queue:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, alert)
        else:
            self._sync_queue.append(alert)

    def _save_preliminary_report(self, patient_id: str) -> None:
        """Saves a JSON report of all findings for the current patient to disk."""
        reports_dir = os.path.join(os.getcwd(), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        report_path = os.path.join(reports_dir, f"report_{patient_id}.json")
        try:
            with open(report_path, "w") as f:
                json.dump(list(self._history), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save preliminary report: {e}")

    def store_thumbnail(self, slice_id: int, hu_array: np.ndarray, window_center: float = None, window_width: float = None) -> None:
        """
        Generate and store a full-resolution PNG for a processed slice.
        Called from the sync pipeline thread.
        """
        try:
            # Store raw HU array for dynamic windowing
            self._hu_arrays[slice_id] = hu_array.copy()

            # Default: use DICOM values if present, else fallback to soft tissue
            wc = window_center if window_center is not None else 50
            ww = window_width if window_width is not None else 350
            windowed = apply_window(hu_array, window_center=wc, window_width=ww)
            img = Image.fromarray(windowed, mode="L")
            # Keep full resolution (512x512) for proper quality

            buf = io.BytesIO()
            img.save(buf, format="PNG", compress_level=1)  # fast compression
            self._thumbnails[slice_id] = buf.getvalue()
        except Exception as e:
            logger.error("Failed to generate thumbnail for slice %d: %s", slice_id, e)

    def get_thumbnail(self, slice_id: int) -> bytes | None:
        """Get stored PNG thumbnail bytes for a slice."""
        return self._thumbnails.get(slice_id)

    def get_thumbnail_base64(self, slice_id: int) -> str | None:
        """Get stored thumbnail as base64 string for JSON embedding."""
        png_bytes = self._thumbnails.get(slice_id)
        if png_bytes:
            return base64.b64encode(png_bytes).decode("ascii")
        return None

    async def register(self, ws: WebSocket) -> None:
        """Register a new WebSocket client."""
        await ws.accept()
        self._clients.add(ws)
        self._stats["connected_clients"] = len(self._clients)
        logger.info("WebSocket client connected (%d total)", len(self._clients))

        # Send recent history to newly connected client
        # Take a snapshot to avoid "deque mutated during iteration" race
        with self._lock:
            history_snapshot = list(self._history)
        for alert in history_snapshot:
            try:
                await ws.send_json(alert)
            except Exception:
                break

    def unregister(self, ws: WebSocket) -> None:
        """Unregister a disconnected client."""
        self._clients.discard(ws)
        self._stats["connected_clients"] = len(self._clients)
        logger.info("WebSocket client disconnected (%d remaining)", len(self._clients))

    async def broadcast_loop(self) -> None:
        """
        Async loop that drains the alert queue and broadcasts to all clients.
        Runs as a background task in the FastAPI app.
        """
        # First, drain any alerts that arrived before the loop started
        while self._sync_queue:
            alert = self._sync_queue.popleft()
            if self._queue:
                await self._queue.put(alert)

        while True:
            try:
                alert = await self._queue.get()
                dead_clients = set()

                for ws in self._clients.copy():
                    try:
                        await ws.send_json(alert)
                    except Exception:
                        dead_clients.add(ws)

                for ws in dead_clients:
                    self.unregister(ws)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Broadcast error: %s", e)

    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        return {**self._stats, "history_size": len(self._history)}


# ── Global broadcaster instance ───────────────────────────────────────────────
broadcaster = AlertBroadcaster()


# ── FastAPI Application ───────────────────────────────────────────────────────

app = FastAPI(
    title="PRISM Phase 1 — Alert Server",
    description="Real-time WebSocket push for CT slice triage alerts",
    version="1.0.0",
)

# CORS for frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """Initialize the broadcaster and start the broadcast loop."""
    loop = asyncio.get_event_loop()
    broadcaster.set_event_loop(loop)
    asyncio.create_task(broadcaster.broadcast_loop())
    logger.info("Alert broadcaster started")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "PRISM Phase 1 Alert Server",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/stats")
async def get_stats():
    """Pipeline statistics."""
    return JSONResponse(content=broadcaster.get_stats())


class EventPayload(BaseModel):
    type: str
    phase: str
    status: str
    estimated_time_sec: float = 0.0
    study_id: str = ""

@app.post("/api/events")
async def post_event(payload: EventPayload):
    """
    Webhook for Phase 2/3 to push progress events to the frontend.
    """
    event_dict = payload.dict()
    # Tag it as a pipeline event so the frontend can differentiate it from DICOM slices
    event_dict["is_pipeline_event"] = True
    
    # Broadcast to WebSocket clients
    if broadcaster._loop and broadcaster._queue:
        broadcaster._loop.call_soon_threadsafe(broadcaster._queue.put_nowait, event_dict)
    else:
        broadcaster._sync_queue.append(event_dict)
        
    return {"status": "ok", "broadcasted": True}



@app.websocket("/ws/alerts")
async def websocket_alerts(ws: WebSocket):
    """
    WebSocket endpoint for real-time triage alerts.
    
    Clients connect here to receive JSON alert messages:
    {
        "slice_id": 5,
        "flagged": true,
        "timestamp": "2026-07-17T16:30:00Z",
        "findings": [...],
        "thumbnail": "base64..."
    }
    """
    await broadcaster.register(ws)
    try:
        # Keep connection alive — listen for client messages (e.g., pings)
        while True:
            data = await ws.receive_text()
            # Client can send "ping" for keepalive
            if data == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        broadcaster.unregister(ws)
    except Exception:
        broadcaster.unregister(ws)


@app.get("/api/slices/{slice_id}/thumbnail")
async def get_thumbnail(slice_id: int):
    """Serve the PNG thumbnail for a given slice."""
    png_bytes = broadcaster.get_thumbnail(slice_id)
    if not png_bytes:
        return JSONResponse(
            status_code=404,
            content={"error": f"No thumbnail for slice {slice_id}"},
        )
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )

@app.get("/api/slices/{slice_id}/windowed")
async def get_windowed_slice(slice_id: int, wc: float = 50, ww: float = 350):
    """
    Serve a full-resolution PNG with custom window/level.
    
    Query params:
        wc: Window Center (default 50 for soft tissue)
        ww: Window Width (default 350)
    
    Common presets:
        Lung:        wc=-600, ww=1500
        Mediastinum: wc=50,   ww=350
        Bone:        wc=400,  ww=1800
        Brain:       wc=40,   ww=80
    """
    hu_array = broadcaster._hu_arrays.get(slice_id)
    if hu_array is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"No HU data for slice {slice_id}"},
        )
    
    windowed = apply_window(hu_array, window_center=wc, window_width=ww)
    img = Image.fromarray(windowed, mode="L")
    
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=1)
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"Cache-Control": "no-cache"},
    )


# ── Scanner Control Endpoints ──────────────────────────────────────────────────
import subprocess
import threading as _threading
from pydantic import BaseModel

scanner_process = None
scanner_stderr_output = ""  # Captured stderr for debugging
current_patient_id = ""

INCOMING_DIR = os.path.join(os.path.dirname(__file__), "sample_dicoms", "incoming")
EXPORTS_DIR = os.path.join(os.path.dirname(__file__), "sample_dicoms", "exports")

def clear_incoming_dir():
    """Clears all dicom files from the incoming directory."""
    if os.path.exists(INCOMING_DIR):
        for f in glob.glob(os.path.join(INCOMING_DIR, "*.dcm")):
            try:
                os.remove(f)
            except Exception as e:
                logger.error(f"Failed to remove {f}: {e}")

def zip_incoming_dir(patient_id: str):
    """Zips the incoming directory and saves it to exports."""
    if not os.path.exists(INCOMING_DIR) or not os.listdir(INCOMING_DIR):
        logger.info("Incoming directory is empty, nothing to zip.")
        return
        
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"patient_{patient_id}" if patient_id else f"scan_{timestamp}"
    
    zip_path = os.path.join(EXPORTS_DIR, name)
    try:
        shutil.make_archive(zip_path, 'zip', INCOMING_DIR)
        logger.info(f"Successfully zipped incoming slices to {zip_path}.zip")
    except Exception as e:
        logger.error(f"Failed to zip incoming directory: {e}")

def _drain_pipe(pipe, label: str):
    """Background thread to drain subprocess pipe and log output."""
    global scanner_stderr_output
    try:
        for line in iter(pipe.readline, b''):
            decoded = line.decode("utf-8", errors="replace").rstrip()
            if decoded:
                if label == "stderr":
                    scanner_stderr_output += decoded + "\n"
                    logger.error(f"[Scanner {label}] {decoded}")
                else:
                    logger.info(f"[Scanner {label}] {decoded}")
    except Exception:
        pass
    finally:
        pipe.close()

class ScannerConfig(BaseModel):
    dataset_dir: str = r"C:\Users\gunan\Downloads\TCIA files"
    delay_ms: int = 30
    patient_id: str = ""
    patient_name: str = ""
    patient_age: str = ""
    patient_sex: str = ""

@app.post("/api/scanner/start")
async def start_scanner(config: ScannerConfig):
    global scanner_process, scanner_stderr_output, current_patient_id
    
    if scanner_process and scanner_process.poll() is None:
        return {"status": "error", "message": "Scanner is already running"}
    
    # Reset stderr capture and set patient ID
    scanner_stderr_output = ""
    current_patient_id = config.patient_id
    
    # Clear previous incoming slices and broadcaster state
    clear_incoming_dir()
    with broadcaster._lock:
        broadcaster._history.clear()
        broadcaster._thumbnails.clear()
        broadcaster._hu_arrays.clear()
        broadcaster._sync_queue.clear()
        broadcaster._stats["total_slices"] = 0
        broadcaster._stats["flagged_slices"] = 0
        
    try:
        # Launch the CT Machine Emulator as a subprocess
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        
        cmd = [
            "python", "-u", "-m", "phase1_ingestion.ct_machine_emulator",
            "--dir", config.dataset_dir,
            "--delay", str(config.delay_ms)
        ]
        
        if config.patient_id and config.patient_id.strip():
            cmd.extend(["--patient", config.patient_id.strip()])
        if config.patient_name and config.patient_name.strip():
            cmd.extend(["--name", config.patient_name.strip()])
        if config.patient_age and config.patient_age.strip():
            cmd.extend(["--age", config.patient_age.strip()])
        if config.patient_sex and config.patient_sex.strip():
            cmd.extend(["--sex", config.patient_sex.strip()])
        
        logger.info(f"Starting scanner with command: {' '.join(cmd)}")
        
        scanner_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=r"e:\MAJOR-PROJECT-PRISM"
        )
        
        # Start background threads to drain stdout/stderr (prevents pipe blocking)
        _threading.Thread(
            target=_drain_pipe, args=(scanner_process.stdout, "stdout"),
            daemon=True, name="scanner-stdout-drain"
        ).start()
        _threading.Thread(
            target=_drain_pipe, args=(scanner_process.stderr, "stderr"),
            daemon=True, name="scanner-stderr-drain"
        ).start()
        
        # Give the process a moment to fail fast (e.g., import errors)
        import asyncio as _asyncio
        await _asyncio.sleep(1.0)
        
        if scanner_process.poll() is not None:
            # Process already exited
            error_msg = scanner_stderr_output.strip()
            logger.error(f"Scanner process exited immediately: {error_msg}")
            scanner_process = None
            return {"status": "error", "message": f"Scanner crashed at startup: {error_msg[-500:] if error_msg else 'No error output'}"}
        
        # Monitor process in background to zip upon natural completion
        async def _wait_and_zip():
            global scanner_process
            while scanner_process and scanner_process.poll() is None:
                await _asyncio.sleep(1.0)
            logger.info("Scanner process finished naturally. Zipping slices...")
            zip_incoming_dir(current_patient_id)
            
        _asyncio.create_task(_wait_and_zip())
        
        return {"status": "success", "message": "Scanner started"}
    except Exception as e:
        logger.error(f"Failed to start scanner: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/scanner/stop")
async def stop_scanner():
    global scanner_process
    if scanner_process and scanner_process.poll() is None:
        scanner_process.terminate()
        scanner_process = None
        logger.info("Scanner stopped by user. Zipping slices...")
        zip_incoming_dir(current_patient_id)
        return {"status": "success", "message": "Scanner stopped and slices zipped"}
    return {"status": "success", "message": "Scanner was not running"}

@app.get("/api/scanner/status")
async def scanner_status():
    global scanner_process, scanner_stderr_output
    if scanner_process and scanner_process.poll() is None:
        return {"running": True}
    # Include any error info if the process stopped
    error = scanner_stderr_output.strip() if scanner_stderr_output else None
    return {"running": False, "last_error": error}

@app.get("/api/scanner/browse")
def browse_folder():
    """Opens a native OS folder dialog on the server and returns the absolute path."""
    try:
        # We run this in a quick subprocess to avoid any Tkinter main-thread issues in FastAPI
        cmd = [
            "python", "-c",
            "import tkinter as tk, tkinter.filedialog as fd, sys; "
            "root=tk.Tk(); root.withdraw(); root.attributes('-topmost', True); "
            "path=fd.askdirectory(title='Select DICOM Directory'); "
            "print(path); sys.exit(0)"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        path = result.stdout.strip()
        
        if path:
            return {"status": "success", "path": os.path.normpath(path)}
        else:
            return {"status": "cancelled"}
    except Exception as e:
        logger.error(f"Browse dialog error: {e}")
        return {"status": "error", "message": str(e)}

def create_alert_payload(
    slice_id: int,
    sop_uid: str,
    patient_id: str,
    triage_result: dict,
    processing_time_ms: float = 0.0,
    window_center: float = None,
    window_width: float = None,
) -> dict:
    """
    Create a standardized alert payload for WebSocket push.
    
    Args:
        slice_id: InstanceNumber of the slice.
        triage_result: Output of triage_screen.screen_slice().to_dict()
        processing_time_ms: Total pipeline processing time.
    
    Returns:
        Dict ready for JSON serialization and WebSocket push.
    """
    thumbnail_b64 = broadcaster.get_thumbnail_base64(slice_id)

    alert = {
        "slice_id": slice_id,
        "uid": sop_uid,
        "patient_id": patient_id,
        "flagged": triage_result.get("flagged", False),
        "emergency_score": triage_result.get("emergency_score", 0),
        "emergency_action": triage_result.get("emergency_action", "CONTINUE"),
        "score_details": triage_result.get("score_details", []),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "findings": triage_result.get("all_findings", []),
        "processing_time_ms": round(processing_time_ms, 2),
        "triage_time_ms": triage_result.get("processing_time_ms", 0.0),
        "slice_hu_mean": triage_result.get("slice_hu_mean", 0.0),
        "slice_hu_std": triage_result.get("slice_hu_std", 0.0),
        "stats": triage_result.get("stats", {}),
        "detected_region": triage_result.get("detected_region", "generic"),
        "thumbnail": thumbnail_b64,
        "window_center_default": window_center,
        "window_width_default": window_width,
        # "phase2": phase2_result, # Phase 2 detached
    }

    return alert
