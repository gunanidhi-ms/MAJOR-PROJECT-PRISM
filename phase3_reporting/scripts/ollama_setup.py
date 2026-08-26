"""Prepare the configured Ollama server and model without machine-specific paths.

Run from the repository root:
    python phase3_reporting/scripts/ollama_setup.py

The helper only manages a server when OLLAMA_HOST points to this computer. A
remote Ollama host is left untouched; its administrator is responsible for
starting it and making the configured model available.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx


PHASE3_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE3_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE3_ROOT))

from app.config import get_settings  # noqa: E402


def _is_local(host: str) -> bool:
    """Return whether *host* addresses the current computer."""
    parsed = urlparse(host)
    return parsed.hostname in {"localhost", "127.0.0.1", "::1"}


def _find_ollama(command: str) -> str | None:
    """Find the configured CLI, including Ollama's normal Windows location."""
    configured = Path(command).expanduser()
    if configured.is_file():
        return str(configured)
    found = shutil.which(command)
    if found:
        return found

    # The fallback is discovery only. Configuration never relies on this path.
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidate = Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe"
        if candidate.is_file():
            return str(candidate)
    return None


def _healthy(host: str, timeout: float = 2.0) -> bool:
    try:
        return httpx.get(f"{host.rstrip('/')}/api/tags", timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


def _start_server(cli: str) -> None:
    kwargs: dict[str, object] = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen([cli, "serve"], **kwargs)  # noqa: S603


def main() -> int:
    settings = get_settings()
    host = settings.ollama_host.rstrip("/")
    print(f"Ollama host: {host}")
    print(f"Ollama model: {settings.ollama_model}")

    if _healthy(host):
        print("Ollama server is ready.")
    elif not _is_local(host):
        print("Remote Ollama server is unavailable. Check OLLAMA_HOST and its server.", file=sys.stderr)
        return 1
    else:
        cli = _find_ollama(settings.ollama_command)
        if not cli:
            print(
                "Ollama is not installed or is not on PATH. Install it from https://ollama.com/download, "
                "then run this helper again.",
                file=sys.stderr,
            )
            return 1
        print("Starting local Ollama server...")
        _start_server(cli)
        deadline = time.monotonic() + settings.ollama_start_timeout
        while time.monotonic() < deadline:
            if _healthy(host):
                print("Ollama server is ready.")
                break
            time.sleep(1)
        else:
            print("Ollama did not become ready in time.", file=sys.stderr)
            return 1

    if not settings.ollama_auto_pull:
        print("Model auto-pull is disabled (OLLAMA_AUTO_PULL=false).")
        return 0

    cli = _find_ollama(settings.ollama_command)
    if not cli:
        print("Ollama CLI is required to pull the configured model.", file=sys.stderr)
        return 1
    print(f"Ensuring model '{settings.ollama_model}' is available (this may download several GB)...")
    env = os.environ.copy()
    env["OLLAMA_HOST"] = host
    completed = subprocess.run([cli, "pull", settings.ollama_model], env=env, check=False)  # noqa: S603
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
