import os
import sys
import time
import subprocess
import threading
import httpx
import logging
from dotenv import load_dotenv

# Load root .env if present
load_dotenv()

# Configure logging for the orchestrator
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] ORCHESTRATOR | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("orchestrator")

def check_phase3_health(url: str, timeout: int = 60) -> bool:
    """Poll Phase 3 health endpoint until it returns 200 OK or times out."""
    logger.info(f"Waiting for Phase 3 to become healthy at {url}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with httpx.Client(timeout=2.0) as client:
                response = client.get(url)
                if response.status_code == 200:
                    logger.info("Phase 3 is healthy and ready.")
                    return True
        except httpx.RequestError:
            pass
        time.sleep(1.0)
    
    logger.error(f"Phase 3 failed to become healthy within {timeout} seconds.")
    return False

def stream_logs(process: subprocess.Popen, prefix: str):
    """Read logs from a subprocess and print them."""
    for line in iter(process.stdout.readline, b''):
        try:
            print(f"[{prefix}] {line.decode('utf-8', errors='replace').rstrip()}")
        except UnicodeEncodeError:
            # Fallback for Windows consoles (cp1252) that cannot print certain characters
            safe_str = line.decode('utf-8', errors='replace').encode('ascii', errors='replace').decode('ascii')
            print(f"[{prefix}] {safe_str.rstrip()}")

def main():
    logger.info("Starting PRISM Autonomous End-to-End Pipeline")
    
    # --- PHASE 3 ---
    phase3_port = os.environ.get("PHASE3_PORT", "8000")
    phase3_health_url = f"http://localhost:{phase3_port}/api/v1/health"
    
    logger.info("Starting Phase 3 (Reporting Service)...")
    
    # Phase 3 has its own venv — use its Python interpreter
    phase3_dir = os.path.abspath("phase3_reporting")
    phase3_python = os.path.join(phase3_dir, "venv", "Scripts", "python.exe")
    if not os.path.isfile(phase3_python):
        # Linux/macOS fallback
        phase3_python = os.path.join(phase3_dir, "venv", "bin", "python")
    if not os.path.isfile(phase3_python):
        # Last resort: global python (user must have deps installed)
        phase3_python = sys.executable
        logger.warning("Phase 3 venv not found, using global Python: %s", phase3_python)
    
    phase3_process = subprocess.Popen(
        [phase3_python, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", phase3_port],
        cwd=phase3_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ}  # Pass environment down
    )
    
    # Start log streamer thread
    threading.Thread(target=stream_logs, args=(phase3_process, "PHASE3"), daemon=True).start()
    
    # Wait for Phase 3 health
    if not check_phase3_health(phase3_health_url):
        logger.error("Aborting startup due to Phase 3 health check failure.")
        phase3_process.terminate()
        sys.exit(1)
        
    # --- PHASE 1 & 2 ---
    # Phase 2 is triggered by Phase 1, so we just set the environment variables
    # to tell Phase 2 how to talk to Phase 3.
    os.environ["PHASE3_ENABLED"] = "true"
    os.environ["PHASE3_URL"] = f"http://localhost:{phase3_port}/api/v1/generate-report"
    
    logger.info("Starting Phase 1 (Ingestion & Real-Time Triage)...")
    
    # Import Phase 1 here so we can run it in the main thread
    # Needs to be done after setting env vars so its imports see them if needed
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "phase1_ingestion")))
    try:
        from phase1_ingestion.pipeline import Phase1Pipeline
    except ImportError as e:
        logger.error(f"Failed to import Phase 1 pipeline: {e}")
        phase3_process.terminate()
        sys.exit(1)
        
    pipeline = Phase1Pipeline()
    
    try:
        pipeline.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down PRISM pipeline...")
    finally:
        # Cleanup
        logger.info("Terminating Phase 3...")
        phase3_process.terminate()
        try:
            phase3_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            phase3_process.kill()
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    main()
