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
        if settings.skip_llm:
            logger.info("LLM skipped (SKIP_LLM=true) – returning template text.")
            return template_text, "template"

        try:
            refined = await self._call_ollama(template_text)
            logger.info("LLM refinement succeeded (%d chars).", len(refined))
            return refined, "llm"
        except LLMUnavailableError as exc:
            logger.warning("Ollama unavailable – falling back to template: %s", exc)
            return template_text, "template"
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Unexpected LLM error – falling back to template: %s", exc, exc_info=True
            )
            return template_text, "template"

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
    