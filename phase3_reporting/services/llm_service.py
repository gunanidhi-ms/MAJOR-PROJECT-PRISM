"""
llm_service.py
--------------
Ollama LLM integration for grammar / fluency refinement.

Responsibilities
~~~~~~~~~~~~~~~~
* Send the deterministic template text to Ollama with the immutable
  system prompt.
* Return the refined text.
* On ANY failure (network, model not loaded, timeout, etc.) gracefully
  fall back to the original template text.

The LLM is NEVER the source of factual content – that role belongs
exclusively to ``template_engine.py``.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from app.config import get_settings
from app.prompts import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)


class LLMUnavailableError(Exception):
    """Raised when Ollama cannot be reached and we want to signal that explicitly."""


class LLMService:
    """
    Thin wrapper around the Ollama ``/api/chat`` endpoint.

    Parameters
    ----------
    host : str, optional
        Override the Ollama host (e.g. in tests).
    model : str, optional
        Override the model name.
    temperature : float, optional
        Sampling temperature.  Keep low (< 0.3) for determinism.
    timeout : int, optional
        HTTP timeout in seconds.
    """

    _refinement_cache: dict[str, str] = {}
    _consecutive_failures: int = 0
    _circuit_open: bool = False

    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
    ) -> None:
        cfg = get_settings()
        self.host = (host or cfg.ollama_host).rstrip("/")
        self.model = model or cfg.ollama_model
        self.temperature = temperature if temperature is not None else cfg.ollama_temperature
        self.timeout = timeout or cfg.ollama_timeout

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    _consecutive_failures: int = 0
    _circuit_open: bool = False

    async def refine(self, template_text: str) -> tuple[str, str]:
        """
        Ask the LLM to refine the template text for grammar / fluency.

        Parameters
        ----------
        template_text : str
            The deterministic findings text from the template engine.

        Returns
        -------
        tuple[str, str]
            ``(refined_text, source)`` where *source* is ``"llm"`` on
            success or ``"template"`` on fallback.
        """
        settings = get_settings()
        if settings.skip_llm or self._circuit_open:
            if self._circuit_open:
                logger.info("LLM skipped (Circuit Breaker OPEN) – returning template text.")
            else:
                logger.info("LLM skipped (SKIP_LLM=true) – returning template text.")
            return template_text, "template"

        if template_text in self._refinement_cache:
            logger.info("LLM refinement served from cache.")
            return self._refinement_cache[template_text], "llm"

        try:
            refined = await self._call_ollama(template_text)
            
            # Add to cache and keep max 1000 items
            if len(self._refinement_cache) >= 1000:
                self._refinement_cache.pop(next(iter(self._refinement_cache)))
            self._refinement_cache[template_text] = refined
            
            self.__class__._consecutive_failures = 0
            logger.info("LLM refinement succeeded (%d chars).", len(refined))
            return refined, "llm"
        except Exception as exc:
            self.__class__._consecutive_failures += 1
            if self.__class__._consecutive_failures >= 2:
                self.__class__._circuit_open = True
                logger.warning("Circuit breaker OPEN - LLM disabled for this session. (Failures: %d)", self.__class__._consecutive_failures)
            logger.warning("Ollama unavailable or failed – falling back to template: %s", exc)
            return template_text, "template"

    async def warmup(self) -> None:
        """
        Pre-load the model into GPU memory by sending a tiny dummy prompt.

        Called once at startup so the first real report doesn't incur the
        cold-start penalty (can be 30-60s for mistral:7b on first load).
        Falls back silently if Ollama is unavailable.
        """
        settings = get_settings()
        if settings.skip_llm:
            return
        logger.info("Warming up Ollama model '%s'...", self.model)
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 1},
            "messages": [{"role": "user", "content": "Hi"}],
        }
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    logger.info("Ollama warmup complete — model '%s' is loaded.", self.model)
                else:
                    logger.warning("Ollama warmup returned HTTP %s: %s", resp.status_code, resp.text[:100])
        except Exception as exc:
            logger.warning("Ollama warmup failed (will retry on first report): %s", exc)

    async def health_check(self) -> bool:
        """
        Return True if the Ollama server is reachable.

        Used by the /health endpoint and startup checks.
        """
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.host}/api/tags")
                return response.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #

    async def _call_ollama(self, template_text: str) -> str:
        """
        Make the actual HTTP call to Ollama's ``/api/chat`` endpoint.

        Uses the non-streaming interface (``"stream": false``) so we get a
        single JSON response rather than a chunked stream.

        Raises
        ------
        LLMUnavailableError
            If the server cannot be reached or returns a non-2xx status.
        """
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": 512,
            },
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(template_text)},
            ],
        }

        logger.debug("Calling Ollama at %s  model=%s", url, self.model)
        logger.debug("Payload lengths -> System: %d chars, User: %d chars", len(SYSTEM_PROMPT), len(payload["messages"][1]["content"]))

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
        except httpx.ConnectError as exc:
            raise LLMUnavailableError(
                f"Cannot connect to Ollama at {self.host}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError(
                f"Ollama request timed out after {self.timeout}s: {exc}"
            ) from exc
        except httpx.RequestError as exc:
            raise LLMUnavailableError(f"HTTP request error: {exc}") from exc

        if response.status_code != 200:
            raise LLMUnavailableError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:200]}"
            )

        try:
            data = response.json()
            content: str = data["message"]["content"]
        except (KeyError, ValueError) as exc:
            raise LLMUnavailableError(
                f"Unexpected Ollama response format: {exc}"
            ) from exc

        return content.strip()
    