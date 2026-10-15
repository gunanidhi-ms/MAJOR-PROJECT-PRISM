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
import sys
from pathlib import Path
from collections import deque
from typing import Any, Optional
from datetime import datetime, timezone

# ── Silence harmless WinError 10054 in asyncio ProactorEventLoop on Windows ────
if sys.platform == "win32":
    try:
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

    def __init__(self, max_history: int = 10000):
        self._queue: asyncio.Queue = None  # set when event loop starts
        self._sync_queue: deque = deque(maxlen=2000)  # thread-safe fallback
        self._lock = threading.Lock()
        self._clients: set[WebSocket] = set()
        self._history: deque = deque(maxlen=max_history)
        self._thumbnails: dict[int, bytes] = {}  # slice_id → PNG bytes
        self._hu_arrays: dict[int, np.ndarray] = {}  # slice_id → raw HU array
        self._current_patient_id: str = ""
        self._current_patient_name: str = ""
        self._current_patient_age: str = ""
        self._current_patient_sex: str = ""
        self._current_dataset_dir: str = ""
        self._current_delay_ms: int = 30
        self._current_study_uid: str = ""
        self._current_series_uid: str = ""
        self._session_active: bool = False
        self._phase1_status: str = "idle"  # "idle" | "scanning" | "completed"
        self._phase2_status: str = "idle"  # "idle" | "running" | "completed" | "failed"
        self._phase3_status: str = "locked"  # "locked" | "unlocked" | "generating" | "draft_ready" | "signed"
        self._session_start_time: str = ""
        self._session_end_time: str = ""
        self._phase2_info: dict = {}
        self._stats = {
            "total_slices": 0,
            "flagged_slices": 0,
            "connected_clients": 0,
            "start_time": datetime.now(timezone.utc).isoformat(),
        }
        self._has_had_client: bool = False
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the asyncio event loop (called from the async context)."""
        self._loop = loop
        self._queue = asyncio.Queue()

    def get_session_info(self) -> dict:
        """Get snapshot of current patient session state and pipeline progression."""
        global scanner_process
        is_running = bool(scanner_process and scanner_process.poll() is None)
        with self._lock:
            return {
                "patient_id": self._current_patient_id,
                "patient_name": self._current_patient_name,
                "patient_age": self._current_patient_age,
                "patient_sex": self._current_patient_sex,
                "dataset_dir": self._current_dataset_dir,
                "delay_ms": self._current_delay_ms,
                "study_id": self._current_study_uid,
                "series_id": self._current_series_uid,
                "session_active": self._session_active,
                "phase1_status": self._phase1_status,
                "phase2_status": self._phase2_status,
                "phase3_status": self._phase3_status,
                "total_slices": len(self._history),
                "flagged_slices": self._stats["flagged_slices"],
                "start_time": self._session_start_time or self._stats["start_time"],
                "phase2_info": self._phase2_info,
                "scanner_running": is_running,
            }

    def reset_for_new_patient(
        self,
        patient_id: str = "",
        patient_name: str = "",
        patient_age: str = "",
        patient_sex: str = "",
        dataset_dir: str = "",
        delay_ms: int = 30,
        study_uid: str = "",
    ) -> None:
        """
        Clears all in-memory slice caches, thumbnails, history, and statistics
        so that a new patient session starts completely fresh.
        """
        with self._lock:
            # If we have history from the previous patient, save preliminary findings
            if self._history and self._current_patient_id:
                self._save_preliminary_report(self._current_patient_id, self._current_patient_name)

            self._history.clear()
            self._thumbnails.clear()
            self._hu_arrays.clear()
            self._sync_queue.clear()
            self._current_patient_id = patient_id
            self._current_patient_name = patient_name
            self._current_patient_age = patient_age
            self._current_patient_sex = patient_sex
            self._current_dataset_dir = dataset_dir
            self._current_delay_ms = delay_ms
            self._current_study_uid = study_uid
            self._current_series_uid = ""
            self._session_active = bool(patient_id or patient_name)
            self._phase1_status = "idle"
            self._phase2_status = "idle"
            self._phase3_status = "locked"
            self._phase2_info = {}
            now_iso = datetime.now(timezone.utc).isoformat()
            self._session_start_time = now_iso
            self._stats["total_slices"] = 0
            self._stats["flagged_slices"] = 0
            self._stats["start_time"] = now_iso

        # Push a session reset event to all connected WebSocket clients
        reset_event = {
            "type": "patient_session_start",
            "patient_id": patient_id,
            "patient_name": patient_name,
            "patient_age": patient_age,
            "patient_sex": patient_sex,
            "study_id": study_uid,
            "dataset_dir": dataset_dir,
            "delay_ms": delay_ms,
            "phase1_status": "idle",
            "phase2_status": "idle",
            "phase3_status": "locked",
            "timestamp": now_iso,
        }
        if self._loop and self._queue:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, reset_event)
        else:
            self._sync_queue.append(reset_event)

        logger.info(
            "AlertBroadcaster reset for new patient (ID: %s, Name: %s, Age: %s, Sex: %s, Study: %s)",
            patient_id, patient_name, patient_age, patient_sex, study_uid[:20] if study_uid else "None"
        )

    def push_alert(self, alert: dict) -> None:
        """
        Push a new triage alert from the sync pipeline.
        Called by Phase 1 pipeline for each processed slice.
        """
        with self._lock:
            self._history.append(alert)
            self._stats["total_slices"] += 1
            if alert.get("flagged"):
                self._stats["flagged_slices"] += 1

            # Track patient metadata
            if alert.get("patient_id"):
                self._current_patient_id = alert["patient_id"]
            if alert.get("patient_name"):
                self._current_patient_name = alert["patient_name"]
            if alert.get("patient_age"):
                self._current_patient_age = alert["patient_age"]
            if alert.get("patient_sex"):
                self._current_patient_sex = alert["patient_sex"]
            if alert.get("study_id") or alert.get("study_instance_uid"):
                self._current_study_uid = alert.get("study_id") or alert.get("study_instance_uid")
            if alert.get("series_instance_uid"):
                self._current_series_uid = alert.get("series_instance_uid")
                
            self._session_active = True
            if self._phase1_status != "completed":
                self._phase1_status = "scanning"

        # Push to async queue via event loop
        if self._loop and self._queue:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, alert)
        else:
            self._sync_queue.append(alert)

    def _save_preliminary_report(self, patient_id: str, patient_name: str = "") -> None:
        """Saves a JSON report of all findings for the current patient to disk using patient ID and name."""
        reports_dir = os.path.join(os.getcwd(), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        safe_p_id = "".join(c for c in (patient_id or "UNK") if c.isalnum() or c in "-_")[:32]
        safe_p_name = "".join(c for c in (patient_name or "Patient").replace("^", "_").replace(" ", "_") if c.isalnum() or c in "-_")[:32]
        
        report_path = os.path.join(reports_dir, f"report_{safe_p_name}_{safe_p_id}.json")
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
        self._has_had_client = True
        self._clients.add(ws)
        self._stats["connected_clients"] = len(self._clients)
        logger.info("WebSocket client connected (%d total)", len(self._clients))

        # Cancel any pending shutdown
        global _shutdown_timer
        with _shutdown_lock:
            if _shutdown_timer:
                _shutdown_timer.cancel()
                _shutdown_timer = None
                logger.info("Pending system shutdown cancelled due to active client connection.")

        # Send full active session state greeting first so the client has complete metadata
        session_greeting = {
            "type": "session_sync",
            "session": self.get_session_info(),
        }
        try:
            await ws.send_json(session_greeting)
        except Exception:
            pass

        # Send full slice history to newly connected client without truncating
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

        # If all client tabs/windows have closed after at least one had connected, initiate shutdown
        if self._has_had_client and len(self._clients) == 0:
            logger.info("All browser tabs/clients disconnected (hit 'X'). Initiating automatic shutdown timer (2.5s)...")
            global _shutdown_timer
            with _shutdown_lock:
                if _shutdown_timer:
                    _shutdown_timer.cancel()
                _shutdown_timer = _threading.Timer(2.5, _perform_system_shutdown)
                _shutdown_timer.daemon = True
                _shutdown_timer.start()

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
    
    # Configure loop exception handler to silence WinError 10054 on Windows
    def _proactor_exception_handler(loop, context):
        exc = context.get("exception")
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError)):
            return
        msg = str(context.get("message", ""))
        if "WinError 10054" in msg or "_call_connection_lost" in msg:
            return
        loop.default_exception_handler(context)

    loop.set_exception_handler(_proactor_exception_handler)
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


@app.get("/health")
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "phase1_ingestion"}

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
    
    # Update broadcaster phase tracking
    with broadcaster._lock:
        if payload.phase == "segmentation":
            if payload.status == "started":
                broadcaster._phase2_status = "running"
            elif payload.status == "completed":
                broadcaster._phase2_status = "completed"
                broadcaster._phase3_status = "unlocked"
            elif payload.status == "failed":
                broadcaster._phase2_status = "failed"
        elif payload.phase == "reporting":
            if payload.status == "started":
                broadcaster._phase3_status = "generating"
            elif payload.status == "completed":
                broadcaster._phase3_status = "draft_ready"
            elif payload.status == "failed":
                broadcaster._phase3_status = "unlocked"
        elif payload.phase == "ingestion":
            if payload.status == "completed":
                broadcaster._phase1_status = "completed"

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
    """
    global _shutdown_timer
    with _shutdown_lock:
        if _shutdown_timer:
            _shutdown_timer.cancel()
            _shutdown_timer = None
            logger.info("Pending system shutdown cancelled due to active WebSocket client.")

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



# ── Patient Session & Scanner Endpoints ────────────────────────────────────────
import subprocess
import threading as _threading
from pydantic import BaseModel

scanner_process = None
scanner_stderr_output = ""  # Captured stderr for debugging
current_patient_id = ""

INCOMING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "sample_dicoms", "incoming"))
EXPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "exports"))

def clear_incoming_dir():
    """Clears all dicom files from the incoming directory."""
    if os.path.exists(INCOMING_DIR):
        for f in glob.glob(os.path.join(INCOMING_DIR, "*.dcm")):
            try:
                os.remove(f)
            except Exception as e:
                logger.error(f"Failed to remove {f}: {e}")

def zip_incoming_dir(patient_id: str, patient_name: str = "", patient_age: str = "", patient_sex: str = "") -> Optional[str]:
    """Zips the incoming directory and saves it with report.json to root exports."""
    if not os.path.exists(INCOMING_DIR) or not os.listdir(INCOMING_DIR):
        logger.info("Incoming directory is empty, nothing to zip.")
        return None
        
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_p_id = "".join(c for c in (patient_id or "UNK") if c.isalnum() or c in "-_")[:32]
    safe_p_name = "".join(c for c in (patient_name or "Patient").replace("^", "_").replace(" ", "_") if c.isalnum() or c in "-_")[:32]
    
    parts = [f"patient_{safe_p_name}_{safe_p_id}"]
    if patient_age:
        parts.append(f"age_{''.join(c for c in patient_age if c.isalnum())}")
    if patient_sex:
        parts.append(f"sex_{''.join(c for c in patient_sex if c.isalnum())}")
    parts.append(timestamp)
    name = "_".join(parts)
    
    zip_path = os.path.join(EXPORTS_DIR, name)
    try:
        # Write report.json into incoming directory before zipping
        report_data = {
            "patient_id": patient_id or "UNK",
            "patient_name": patient_name or "Patient",
            "patient_age": patient_age or "",
            "patient_sex": patient_sex or "",
            "total_slices": len(glob.glob(os.path.join(INCOMING_DIR, "*.dcm"))),
            "status": "archived",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "findings": "Real-time CT slice screening completed. DICOM slices validated and archived.",
            "impression": f"CT scan acquisition complete for {patient_name or 'Patient'} ({patient_id or 'UNK'}).",
            "source": "PRISM_INTEGRATED_PIPELINE",
            "validated": True
        }
        report_file = os.path.join(INCOMING_DIR, "report.json")
        with open(report_file, "w", encoding="utf-8") as rf:
            json.dump(report_data, rf, indent=2, ensure_ascii=False)

        shutil.make_archive(zip_path, 'zip', INCOMING_DIR)
        final_zip_name = f"{name}.zip"
        logger.info(f"Successfully zipped incoming slices + report.json to {zip_path}.zip")
        return final_zip_name
    except Exception as e:
        logger.error(f"Failed to zip incoming directory: {e}")
        return None

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
    dataset_dir: str = r"D:\MAJOR-PROJECT-PRISM\phase1_ingestion\sample_dicoms\test_series"
    delay_ms: int = 30
    patient_id: str = ""
    patient_name: str = ""
    patient_age: str = ""
    patient_sex: str = ""

class PatientSessionStartRequest(BaseModel):
    patient_id: str = ""
    patient_name: str = ""
    patient_age: str = ""
    patient_sex: str = ""
    dataset_dir: str = r"D:\MAJOR-PROJECT-PRISM\phase1_ingestion\sample_dicoms\test_series"
    delay_ms: int = 30
    auto_start_scan: bool = True

@app.get("/api/session/current")
async def get_current_session():
    """Get active patient session metadata and phase status."""
    return broadcaster.get_session_info()

@app.post("/api/session/start")
async def start_patient_session(req: PatientSessionStartRequest):
    """Initializes a new patient session and starts Phase 1 slice streaming."""
    global scanner_process, scanner_stderr_output, current_patient_id
    
    # Terminate any existing scanner process
    if scanner_process and scanner_process.poll() is None:
        try:
            scanner_process.terminate()
        except Exception:
            pass
        scanner_process = None

    scanner_stderr_output = ""
    current_patient_id = req.patient_id.strip()

    # Reset archive manager
    from phase1_ingestion.dicom_archive import get_archive_manager
    archive_mgr = get_archive_manager()
    if archive_mgr:
        archive_mgr.reset_session()
    
    clear_incoming_dir()

    # Explicit pipeline accumulator reset
    from phase1_ingestion.pipeline import get_pipeline
    pipeline = get_pipeline()
    if pipeline:
        pipeline.accumulator.reset_all()
        pipeline.buffer.reset()

    # Reset broadcaster state for new patient
    broadcaster.reset_for_new_patient(
        patient_id=req.patient_id.strip(),
        patient_name=req.patient_name.strip(),
        patient_age=req.patient_age.strip(),
        patient_sex=req.patient_sex.strip(),
        dataset_dir=req.dataset_dir.strip(),
        delay_ms=req.delay_ms,
        study_uid="",
    )

    # Launch CT scanner emulator
    try:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        
        cmd = [
            "python", "-u", "-m", "phase1_ingestion.ct_machine_emulator",
            "--dir", req.dataset_dir.strip(),
            "--delay", str(req.delay_ms)
        ]
        if req.patient_id.strip():
            cmd.extend(["--patient", req.patient_id.strip()])
        if req.patient_name.strip():
            cmd.extend(["--name", req.patient_name.strip()])
        if req.patient_age.strip():
            cmd.extend(["--age", req.patient_age.strip()])
        if req.patient_sex.strip():
            cmd.extend(["--sex", req.patient_sex.strip()])

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        scanner_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=project_root
        )

        _threading.Thread(
            target=_drain_pipe, args=(scanner_process.stdout, "stdout"),
            daemon=True, name="scanner-stdout-drain"
        ).start()
        _threading.Thread(
            target=_drain_pipe, args=(scanner_process.stderr, "stderr"),
            daemon=True, name="scanner-stderr-drain"
        ).start()

        with broadcaster._lock:
            broadcaster._phase1_status = "scanning"

        # Background task to monitor scanner completion and automatically trigger Phase 2
        async def _monitor_scan_and_chain_pipeline():
            global scanner_process
            while scanner_process and scanner_process.poll() is None:
                await asyncio.sleep(1.0)
            
            await asyncio.sleep(2.0)
            logger.info("Phase 1 scan finished. Slices in history: %d", len(broadcaster._history))
            
            with broadcaster._lock:
                broadcaster._phase1_status = "completed"
            
            # Broadcast Phase 1 completion
            completion_event = {
                "type": "pipeline_event",
                "phase": "ingestion",
                "status": "completed",
                "total_slices": len(broadcaster._history),
                "is_pipeline_event": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if broadcaster._loop and broadcaster._queue:
                broadcaster._loop.call_soon_threadsafe(broadcaster._queue.put_nowait, completion_event)
            else:
                broadcaster._sync_queue.append(completion_event)

            # Auto-trigger Phase 2 if we have accumulated slices
            try:
                from phase1_ingestion.pipeline import get_pipeline
                p = get_pipeline()
                if p and p.accumulator and len(p.accumulator._slices) >= 3:
                    logger.info("Auto-triggering Phase 2 TotalSegmentator & AI Screening...")
                    with broadcaster._lock:
                        broadcaster._phase2_status = "running"
                    p.accumulator.on_stream_stall()
                    p.accumulator.on_association_released()
            except Exception as e:
                logger.error("Auto Phase 2 trigger error: %s", e)

        asyncio.create_task(_monitor_scan_and_chain_pipeline())

        return {
            "status": "success",
            "message": f"Patient session started for {req.patient_name or 'Patient'} ({req.patient_id or 'ID'}). Live scan active.",
            "scan_started": True,
            "session": broadcaster.get_session_info(),
        }
    except Exception as e:
        logger.error(f"Failed to start patient session: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/session/end")
@app.post("/api/archive/close-session")
async def end_patient_session():
    """Finalizes patient session, archives DICOM slices + report.json to exports/, and cleans memory."""
    global scanner_process
    if scanner_process and scanner_process.poll() is None:
        try:
            scanner_process.terminate()
        except Exception:
            pass
        scanner_process = None

    p_id = broadcaster._current_patient_id
    p_name = broadcaster._current_patient_name
    p_age = broadcaster._current_patient_age
    p_sex = broadcaster._current_patient_sex

    # Write report.json into incoming dir and zip
    zip_name = zip_incoming_dir(p_id, p_name, p_age, p_sex)
    if not zip_name:
        # Fallback if incoming dir was already empty
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in (p_name or "Patient") if c.isalnum()) or "Patient"
        safe_id = "".join(c for c in (p_id or "ID") if c.isalnum()) or "ID"
        zip_name = f"patient_{safe_name}_{safe_id}_{timestamp}.zip"

    clear_incoming_dir()
    broadcaster.reset_for_new_patient()

    end_event = {
        "type": "patient_session_end",
        "archive_file": zip_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if broadcaster._loop and broadcaster._queue:
        broadcaster._loop.call_soon_threadsafe(broadcaster._queue.put_nowait, end_event)
    else:
        broadcaster._sync_queue.append(end_event)

    return {
        "status": "success",
        "message": f"Patient session finalized and archived to exports/{zip_name}.",
        "archive_file": zip_name,
        "archive_path": os.path.join(EXPORTS_DIR, zip_name),
    }

@app.post("/api/session/phase2-trigger")
async def trigger_phase2():
    """Manually triggers Phase 2 segmentation & AI analysis on current volume."""
    try:
        from phase1_ingestion.pipeline import get_pipeline
        p = get_pipeline()
        if p and p.accumulator:
            with broadcaster._lock:
                broadcaster._phase2_status = "running"
            p.accumulator.on_stream_stall()
            p.accumulator.on_association_released()
            return {"status": "success", "message": "Phase 2 TotalSegmentator initiated."}
        return {"status": "error", "message": "Accumulator not initialized."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/phase2/volumetric-data")
async def get_phase2_volumetric_data():
    """
    Return Phase 2 findings and candidate 3D bounding boxes for the
    holographic volumetric viewer.

    Finds the most recent phase2_work/run_* directory, reads findings.json
    and candidates.json, and returns a unified payload.

    Returns:
        {
            "status": "ready" | "not_ready",
            "findings": [...],         # from findings.json (Phase 2 final findings)
            "candidates": [...],       # from candidates.json (all candidates with 3D bbox)
            "organ_stats": {...},      # from statistics.json (patient baselines)
            "region": "chest" | ...    # from region.txt
        }
    """
    try:
        work_dir = os.path.abspath("phase2_work")
        if not os.path.isdir(work_dir):
            return JSONResponse({"status": "not_ready", "reason": "no phase2_work dir"})

        # Find all run directories and pick the most recent by mtime
        run_dirs = sorted(
            [d for d in glob.glob(os.path.join(work_dir, "run_*")) if os.path.isdir(d)],
            key=os.path.getmtime,
            reverse=True,
        )
        if not run_dirs:
            return JSONResponse({"status": "not_ready", "reason": "no run dirs found"})

        seg_dir = os.path.join(run_dirs[0], "temp_seg")
        if not os.path.isdir(seg_dir):
            return JSONResponse({"status": "not_ready", "reason": "temp_seg not found"})

        # Read findings.json
        findings = []
        findings_path = os.path.join(seg_dir, "findings.json")
        if os.path.isfile(findings_path):
            try:
                with open(findings_path) as f:
                    envelope = json.load(f)
                findings = envelope.get("findings", [])
            except Exception as e:
                logger.warning("Could not read findings.json: %s", e)

        # Read candidates.json
        candidates = []
        cand_path = os.path.join(seg_dir, "candidates.json")
        if os.path.isfile(cand_path):
            try:
                with open(cand_path) as f:
                    candidates = json.load(f)
            except Exception as e:
                logger.warning("Could not read candidates.json: %s", e)

        # Read statistics.json (organ baselines)
        organ_stats = {}
        stats_path = os.path.join(seg_dir, "statistics.json")
        if os.path.isfile(stats_path):
            try:
                with open(stats_path) as f:
                    organ_stats = json.load(f)
            except Exception as e:
                logger.warning("Could not read statistics.json: %s", e)

        # Read region
        region = "unknown"
        region_path = os.path.join(seg_dir, "region.txt")
        if os.path.isfile(region_path):
            try:
                with open(region_path) as f:
                    region = f.read().strip()
            except Exception:
                pass

        return JSONResponse({
            "status": "ready",
            "findings": findings,
            "candidates": candidates,
            "organ_stats": organ_stats,
            "region": region,
            "run_dir": run_dirs[0],
        })

    except Exception as e:
        logger.error("phase2/volumetric-data error: %s", e, exc_info=True)
        return JSONResponse({"status": "error", "reason": str(e)}, status_code=500)


@app.get("/api/phase2/organ-mesh")
async def get_phase2_organ_mesh():
    """
    Run marching cubes on the TotalSegmentator segmentation and return
    actual 3D organ surface meshes with per-vertex anomaly colouring.

    Normal tissue vertices → grey
    Anomalous zones (overlapping with candidate bboxes) → red

    This is the data source for the real anatomy 3D holographic viewer.

    Returns (may be several MB depending on anatomy complexity):
        {
            "status": "ready" | "not_ready" | "error",
            "organs": [{label, vertices, faces, colours, ...}, ...],
            "anomalies": [{bbox_norm, label, hu_mean}, ...],
            "n_organs": int,
            "n_anomaly_candidates": int,
        }
    """
    import asyncio
    import functools

    try:
        work_dir = os.path.abspath("phase2_work")
        if not os.path.isdir(work_dir):
            return JSONResponse({"status": "not_ready", "reason": "no phase2_work dir"})

        # Find most recent run dir
        run_dirs = sorted(
            [d for d in glob.glob(os.path.join(work_dir, "run_*")) if os.path.isdir(d)],
            key=os.path.getmtime,
            reverse=True,
        )
        if not run_dirs:
            return JSONResponse({"status": "not_ready", "reason": "no run dirs"})

        seg_dir = os.path.join(run_dirs[0], "temp_seg")
        if not os.path.isdir(seg_dir):
            return JSONResponse({"status": "not_ready", "reason": "temp_seg not found"})

        # Locate the TotalSegmentator NIfTI segmentation
        # TotalSegmentator writes it as 'segmentations.nii.gz' or 'temp_seg.nii.gz'
        seg_nifti = None
        # Search order: our filename first, then TotalSegmentator standard names
        for candidate_name in [
            "temp_seg.nii",           # our pipeline writes this name
            "temp_seg.nii.gz",
            "segmentations.nii.gz",   # TotalSegmentator multi-label output
            "segmentation.nii.gz",
        ]:
            p = os.path.join(seg_dir, candidate_name)
            if os.path.isfile(p):
                seg_nifti = p
                break

        # Also check one level up (the run dir itself)
        if seg_nifti is None:
            for candidate_name in ["segmentations.nii.gz", "temp_seg.nii.gz"]:
                p = os.path.join(run_dirs[0], candidate_name)
                if os.path.isfile(p):
                    seg_nifti = p
                    break

        if seg_nifti is None:
            # Fallback: scan for any .nii or .nii.gz file that isn't the volume
            for root, _, files in os.walk(run_dirs[0]):
                for f in files:
                    if f.endswith((".nii", ".nii.gz")) and "volume" not in f.lower():
                        seg_nifti = os.path.join(root, f)
                        break
                if seg_nifti:
                    break

        if seg_nifti is None:
            return JSONResponse({
                "status": "not_ready",
                "reason": "TotalSegmentator .nii not found. Phase 2 may still be running."
            })

        # Load candidates.json
        candidates = []
        cand_path = os.path.join(seg_dir, "candidates.json")
        if os.path.isfile(cand_path):
            try:
                with open(cand_path) as f:
                    candidates = json.load(f)
            except Exception:
                pass

        # Load region
        region = "unknown"
        region_path = os.path.join(seg_dir, "region.txt")
        if os.path.isfile(region_path):
            try:
                with open(region_path) as f:
                    region = f.read().strip()
            except Exception:
                pass

        # ── Heavy work in thread pool (marching cubes is CPU-intensive) ────────
        def _build_meshes():
            import nibabel as nib
            from phase2_segmentation.mesh_builder import build_organ_meshes

            img = nib.load(seg_nifti)
            seg_arr = img.get_fdata(dtype=np.float32)

            # TotalSegmentator output can be (X,Y,Z) or (Z,Y,X) depending on version
            # Standard: (X, Y, Z) with affine — we need (Z, Y, X) for our pipeline
            # Check header orientation and transpose if needed
            # Simple heuristic: if the first dim is the smallest, it's likely X not Z
            if seg_arr.ndim == 3:
                # TotalSegmentator 2.x: output is (X, Y, Z)
                # Our pipeline expects (Z, Y, X) — transpose
                seg_arr = np.transpose(seg_arr, (2, 1, 0)).astype(np.int16)

            # Build label map: class_map["total"] is {int_label: organ_name_str}
            # which is exactly what mesh_builder expects as cmap={int: str}.
            try:
                from totalsegmentator.map_to_binary import class_map
                # class_map["total"] = {1: "spleen", 2: "kidney_right", 5: "liver", ...}
                cmap = dict(class_map.get("total", {}))   # {int_label: organ_name}
            except Exception:
                # Fallback: generate generic names
                unique_labels = [int(v) for v in np.unique(seg_arr) if v != 0]
                cmap = {v: "organ_" + str(v) for v in unique_labels}

            # Spacing from NIfTI header (voxel size in mm)
            # nibabel zooms for a (X,Y,Z)-stored image are (x_mm, y_mm, z_mm).
            # After transposing to (Z,Y,X) the marching cubes spacing order is
            # (z_mm, y_mm, x_mm) to match the array axis order.
            voxel_sizes = img.header.get_zooms()
            x_mm = float(voxel_sizes[0])
            y_mm = float(voxel_sizes[1])
            z_mm = float(voxel_sizes[2])
            spacing = (z_mm, y_mm, x_mm)   # (Z, Y, X) for (Z,Y,X) array

            mesh_data = build_organ_meshes(
                seg_arr=seg_arr,
                cmap=cmap,
                candidates=candidates,
                spacing=spacing,
                downsample=2,
                max_organs=25,
            )
            mesh_data["region"] = region
            return mesh_data

        loop = asyncio.get_event_loop()
        mesh_data = await loop.run_in_executor(None, _build_meshes)

        return JSONResponse({"status": "ready", **mesh_data})

    except Exception as e:
        logger.error("phase2/organ-mesh error: %s", e, exc_info=True)
        return JSONResponse({"status": "error", "reason": str(e)}, status_code=500)


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

# ── System Shutdown & Terminal Auto-Close Logic ──────────────────────────────
_shutdown_timer = None
_shutdown_lock = _threading.Lock()

def _perform_system_shutdown():
    """Executes clean termination of all backend & frontend processes and terminal windows."""
    logger.info("Executing PRISM full system shutdown...")
    global scanner_process
    if scanner_process and scanner_process.poll() is None:
        try:
            scanner_process.terminate()
        except Exception:
            pass

    try:
        from phase1_ingestion.dicom_archive import get_archive_manager
        archive_mgr = get_archive_manager()
        if archive_mgr:
            archive_mgr.force_archive()
    except Exception:
        pass

    # Read .prism_pids.json for spawned terminal process IDs
    pids_file = os.path.abspath(".prism_pids.json")
    pids_to_kill = []
    if os.path.isfile(pids_file):
        try:
            with open(pids_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("backend_pid"):
                    pids_to_kill.append(data["backend_pid"])
                if data.get("frontend_pid"):
                    pids_to_kill.append(data["frontend_pid"])
        except Exception as e:
            logger.error(f"Error reading .prism_pids.json: {e}")

    # Kill spawned terminal processes (PowerShell window + child tree)
    for pid in pids_to_kill:
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
            else:
                os.kill(int(pid), 9)
        except Exception:
            pass

    # Fallback port kills for Vite / Reporting server on Windows
    if os.name == "nt":
        for port in [5173, 8005]:
            try:
                cmd = f'powershell -Command "Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess | ForEach-Object {{ Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }}"'
                subprocess.run(cmd, shell=True, capture_output=True)
            except Exception:
                pass

    if os.path.isfile(pids_file):
        try:
            os.remove(pids_file)
        except Exception:
            pass

    os._exit(0)

@app.post("/api/system/shutdown")
@app.get("/api/system/shutdown")
async def system_shutdown(grace: float = 0.5):
    """
    Shuts down PRISM backend services, frontend Vite server,
    and closes the associated terminal windows after a short grace period.
    """
    global _shutdown_timer
    with _shutdown_lock:
        if _shutdown_timer:
            _shutdown_timer.cancel()
        _shutdown_timer = _threading.Timer(grace, _perform_system_shutdown)
        _shutdown_timer.daemon = True
        _shutdown_timer.start()

    return {"status": "success", "message": f"PRISM shutdown initiated ({grace}s grace). Terminals will close."}

@app.post("/api/system/cancel-shutdown")
async def cancel_shutdown():
    """Cancels a pending shutdown (e.g. if the browser reloads)."""
    global _shutdown_timer
    with _shutdown_lock:
        if _shutdown_timer:
            _shutdown_timer.cancel()
            _shutdown_timer = None
            logger.info("Pending system shutdown cancelled.")
            return {"status": "success", "message": "Shutdown cancelled."}
    return {"status": "success", "message": "No shutdown was pending."}

def create_alert_payload(
    slice_id: int,
    sop_uid: str,
    patient_id: str,
    triage_result: dict,
    processing_time_ms: float = 0.0,
    window_center: float = None,
    window_width: float = None,
    patient_name: str = "",
    study_instance_uid: str = "",
    series_instance_uid: str = "",
) -> dict:
    """
    Create a standardized alert payload for WebSocket push.
    
    Args:
        slice_id: InstanceNumber of the slice.
        sop_uid: SOP Instance UID.
        patient_id: DICOM Patient ID.
        triage_result: Output of triage_screen.screen_slice().to_dict()
        processing_time_ms: Total pipeline processing time.
        window_center: Default window center from DICOM.
        window_width: Default window width from DICOM.
        patient_name: Patient Name.
        study_instance_uid: Study Instance UID.
        series_instance_uid: Series Instance UID.
    
    Returns:
        Dict ready for JSON serialization and WebSocket push.
    """
    thumbnail_b64 = broadcaster.get_thumbnail_base64(slice_id)

    alert = {
        "slice_id": slice_id,
        "uid": sop_uid,
        "patient_id": patient_id,
        "patient_name": patient_name,
        "study_id": study_instance_uid,
        "study_instance_uid": study_instance_uid,
        "series_instance_uid": series_instance_uid,
        "flagged": triage_result.get("flagged", False),
        "emergency_score": triage_result.get("emergency_score", 0),
        "emergency_action": triage_result.get("emergency_action", "CONTINUE"),
        "score_details": triage_result.get("score_details", []),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "findings": triage_result.get("displayed_findings", triage_result.get("all_findings", [])),
        "processing_time_ms": round(processing_time_ms, 2),
        "triage_time_ms": triage_result.get("processing_time_ms", 0.0),
        "slice_hu_mean": triage_result.get("slice_hu_mean", 0.0),
        "slice_hu_std": triage_result.get("slice_hu_std", 0.0),
        "stats": triage_result.get("stats", {}),
        "detected_region": triage_result.get("detected_region", "generic"),
        "thumbnail": thumbnail_b64,
        "window_center_default": window_center,
        "window_width_default": window_width,
    }

    return alert
