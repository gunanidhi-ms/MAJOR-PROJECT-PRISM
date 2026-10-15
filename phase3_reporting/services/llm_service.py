from __future__ import annotations

import logging
from typing import Optional

import httpx

from app.config import get_settings
from app.prompts import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)


class LLMUnavailableError(Exception):
    pass


class LLMService:
    """Thin wrapper around the Ollama /api/chat endpoint."""

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
        self.timeout = timeout if timeout is not None else cfg.ollama_timeout

    async def refine(self, template_text: str) -> tuple[str, str]:
        settings = get_settings()
        if settings.skip_llm or self._circuit_open:
            reason = "Circuit Breaker OPEN" if self._circuit_open else "SKIP_LLM=true"
            logger.info("LLM skipped (%s) – returning template text.", reason)
            return template_text, "template"

        if template_text in self._refinement_cache:
            logger.info("LLM refinement served from cache.")
            return self._refinement_cache[template_text], "llm"

        try:
            refined = await self._call_ollama(template_text)
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
                logger.warning("Circuit breaker OPEN after %d failures.", self.__class__._consecutive_failures)
            logger.warning("Ollama fallback to template: %s", exc)
            return template_text, "template"

    async def warmup(self) -> None:
        settings = get_settings()
        if settings.skip_llm:
            return
        if not await self.health_check():
            logger.info("Ollama not reachable — skipping warmup.")
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
            async with httpx.AsyncClient(timeout=min(self.timeout, 30.0)) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    logger.info("Ollama warmup complete — model '%s' loaded.", self.model)
                else:
                    logger.warning("Ollama warmup HTTP %s: %s", resp.status_code, resp.text[:100])
        except Exception as exc:
            logger.warning("Ollama warmup failed: %s", exc)

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=min(self.timeout, 5.0)) as client:
                response = await client.get(f"{self.host}/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    async def _call_ollama(self, template_text: str) -> str:
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": 2048,
            },
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(template_text)},
            ],
        }

        logger.debug("Calling Ollama at %s model=%s", url, self.model)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
        except httpx.ConnectError as exc:
            raise LLMUnavailableError(f"Cannot connect to Ollama at {self.host}: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError(f"Ollama request timed out after {self.timeout}s: {exc}") from exc
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
            raise LLMUnavailableError(f"Unexpected Ollama response format: {exc}") from exc

        return content.strip()
