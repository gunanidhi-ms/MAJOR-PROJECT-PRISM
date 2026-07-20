"""
main.py
-------
FastAPI application entry point for Phase 3 – PRISM Radiology Report Generator.

Start the server
~~~~~~~~~~~~~~~~
    # from the phase3_reporting/ directory:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

    # or via the helper script:
    python -m app.main
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.utils import setup_logging
from api.report_routes import router as report_router

# ── Logging ──────────────────────────────────────────────────────────── #
setup_logging()
logger = logging.getLogger(__name__)


# ====================================================================== #
#  Application lifespan
# ====================================================================== #


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Run startup and shutdown logic.

    Startup
    ~~~~~~~
    * Log configuration summary.
    * Ensure the reports directory exists.

    Shutdown
    ~~~~~~~~
    * Log clean exit.
    """
    settings = get_settings()
    logger.info("=" * 60)
    logger.info("PRISM Phase 3 – Radiology Report Generator")
    logger.info("Ollama host  : %s", settings.ollama_host)
    logger.info("Ollama model : %s", settings.ollama_model)
    logger.info("Reports dir  : %s", settings.reports_dir)
    logger.info("Skip LLM     : %s", settings.skip_llm)
    logger.info("=" * 60)

    import pathlib
    pathlib.Path(settings.reports_dir).mkdir(parents=True, exist_ok=True)

    # ── Model warm-up ────────────────────────────────────────────────── #
    # Fire a tiny /api/chat request at startup so Ollama loads the model
    # into memory NOW, before the first real request arrives.
    # Without this, the first POST /generate-report bears the full
    # cold-load penalty (~12-15 s), which can exceed the httpx timeout.
    if not settings.skip_llm:
        warmup_url = f"{settings.ollama_host.rstrip('/')}/api/chat"
        warmup_payload = {
            "model": settings.ollama_model,
            "stream": False,
            "options": {"num_predict": 1},
            "messages": [{"role": "user", "content": "hi"}],
        }
        try:
            logger.info("Warming up Ollama model '%s' …", settings.ollama_model)
            async with httpx.AsyncClient(timeout=180) as client:
                await client.post(warmup_url, json=warmup_payload)
            logger.info("Ollama warm-up complete – model is loaded and ready.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama warm-up failed (server may be offline): %s", exc)

    yield  # Application runs here

    logger.info("PRISM Phase 3 – shutting down cleanly.")


# ====================================================================== #
#  FastAPI application
# ====================================================================== #

app = FastAPI(
    title="PRISM – Phase 3: Radiology Report Generator",
    description=(
        "Converts structured radiology findings JSON (Phase 2 output) into "
        "professional draft reports using a deterministic template engine "
        "backed by an Ollama LLM for grammar refinement.  "
        "Zero hallucination guaranteed by a multi-axis validator."
    ),
    version="1.0.0",
    contact={
        "name": "PRISM Radiology AI Team",
    },
    license_info={
        "name": "Proprietary – PRISM Project",
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ── CORS ─────────────────────────────────────────────────────────────── #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global exception handler ─────────────────────────────────────────── #


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "Unhandled exception on %s %s: %s",
        request.method,
        request.url.path,
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred.", "code": "INTERNAL_ERROR"},
    )


# ── Routes ───────────────────────────────────────────────────────────── #

app.include_router(report_router)


# ── Root ─────────────────────────────────────────────────────────────── #


@app.get("/", tags=["Root"], summary="API info")
async def root():
    return {
        "service": "PRISM Phase 3 – Radiology Report Generator",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


# ====================================================================== #
#  Direct execution  (python -m app.main)
# ====================================================================== #

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.fastapi_host,
        port=settings.fastapi_port,
        reload=settings.fastapi_debug,
    )
