"""
HTTP client for the PRISM Phase 3 FastAPI backend.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from django.conf import settings

logger = logging.getLogger(__name__)


class PrismAPIError(Exception):
    """Raised when the FastAPI backend returns an error."""

    def __init__(self, message: str, status_code: int = 500, detail: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail or message


class PrismAPIClient:
    """Thin wrapper around the Phase 3 reporting REST API."""

    def __init__(self, base_url: str | None = None, timeout: float = 60.0):
        self.base_url = (base_url or settings.PRISM_API_BASE_URL).rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.request(method, url, **kwargs)
        except httpx.ConnectError as exc:
            logger.error("Cannot connect to PRISM API at %s", self.base_url)
            raise PrismAPIError(
                "Backend unavailable. Start the FastAPI server on port 8000.",
                status_code=503,
            ) from exc
        except httpx.TimeoutException as exc:
            raise PrismAPIError("Backend request timed out.", status_code=504) from exc

        if response.status_code >= 400:
            detail = response.text
            try:
                payload = response.json()
                detail = payload.get("detail", detail)
                if isinstance(detail, list):
                    detail = "; ".join(str(item) for item in detail)
            except Exception:
                pass
            raise PrismAPIError(str(detail), status_code=response.status_code, detail=str(detail))

        if response.status_code == 204 or not response.content:
            return {}
        return response.json()

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def list_reports(self) -> dict[str, Any]:
        return self._request("GET", "/reports")

    def get_report(self, study_id: str) -> dict[str, Any]:
        return self._request("GET", f"/report/{study_id}")

    def generate_report(self, findings: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/generate-report", json=findings)

    def update_report(
        self,
        study_id: str,
        findings: str | None = None,
        impression: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, str] = {}
        if findings is not None:
            payload["findings"] = findings
        if impression is not None:
            payload["impression"] = impression
        return self._request("PUT", f"/report/{study_id}", json=payload)

    def sign_report(self, study_id: str) -> dict[str, Any]:
        return self._request("POST", "/sign-report", json={"study_id": study_id})


def get_api_client() -> PrismAPIClient:
    return PrismAPIClient()
