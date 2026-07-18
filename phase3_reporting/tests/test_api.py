"""
tests/test_api.py
-----------------
Integration tests for the FastAPI endpoints.

Uses TestClient (synchronous) so no running Ollama is required.
The LLM service is patched to return the template text directly
(``skip_llm`` mode) ensuring tests are deterministic and offline.

Run with:
    pytest tests/test_api.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Ensure phase3_reporting is on sys.path when running from any CWD
import sys, os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app
from storage.storage import ReportStorage

# ====================================================================== #
#  Fixtures
# ====================================================================== #

SAMPLE_DIR = Path(__file__).parent.parent / "sample_data"


def load_sample(name: str) -> dict:
    with (SAMPLE_DIR / name).open() as f:
        return json.load(f)


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """TestClient with LLM skipped for determinism."""
    with patch.dict(os.environ, {"SKIP_LLM": "true", "REPORTS_DIR": "reports_test"}):
        with TestClient(app) as c:
            yield c


@pytest.fixture(scope="module", autouse=True)
def cleanup_test_reports():
    """Remove test report files after the test module completes."""
    yield
    import shutil
    test_dir = Path("reports_test")
    if test_dir.exists():
        shutil.rmtree(test_dir)


# ====================================================================== #
#  POST /api/v1/generate-report
# ====================================================================== #


class TestGenerateReport:
    def test_normal_case_returns_200(self, client):
        payload = load_sample("normal_case.json")
        resp = client.post("/api/v1/generate-report", json=payload)
        assert resp.status_code == 200

    def test_normal_case_response_schema(self, client):
        payload = load_sample("normal_case.json")
        data = client.post("/api/v1/generate-report", json=payload).json()
        assert "study_id" in data
        assert "findings" in data
        assert "impression" in data
        assert "validated" in data
        assert "status" in data
        assert "source" in data

    def test_normal_case_status_is_draft(self, client):
        payload = load_sample("normal_case.json")
        data = client.post("/api/v1/generate-report", json=payload).json()
        assert data["status"] == "draft"

    def test_normal_case_impression_is_empty(self, client):
        payload = load_sample("normal_case.json")
        data = client.post("/api/v1/generate-report", json=payload).json()
        assert data["impression"] == ""

    def test_single_lesion_contains_measurements(self, client):
        payload = load_sample("single_lesion.json")
        data = client.post("/api/v1/generate-report", json=payload).json()
        assert "22.4" in data["findings"]
        assert "14.1" in data["findings"]

    def test_multi_lesion_returns_200(self, client):
        payload = load_sample("multi_lesion.json")
        resp = client.post("/api/v1/generate-report", json=payload)
        assert resp.status_code == 200

    def test_multi_organ_returns_200(self, client):
        payload = load_sample("multi_organ.json")
        resp = client.post("/api/v1/generate-report", json=payload)
        assert resp.status_code == 200

    def test_invalid_json_returns_422(self, client):
        resp = client.post(
            "/api/v1/generate-report",
            content=b"not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_missing_organs_field_returns_422(self, client):
        resp = client.post(
            "/api/v1/generate-report",
            json={"study_id": "X", "modality": "CT"},
        )
        assert resp.status_code == 422

    def test_empty_organs_returns_422(self, client):
        resp = client.post(
            "/api/v1/generate-report",
            json={"study_id": "X", "modality": "CT", "organs": []},
        )
        assert resp.status_code == 422


# ====================================================================== #
#  GET /api/v1/report/{study_id}
# ====================================================================== #


class TestGetReport:
    def test_get_existing_report(self, client):
        payload = load_sample("normal_case.json")
        study_id = payload["study_id"]
        client.post("/api/v1/generate-report", json=payload)  # ensure exists
        resp = client.get(f"/api/v1/report/{study_id}")
        assert resp.status_code == 200

    def test_get_existing_report_schema(self, client):
        payload = load_sample("normal_case.json")
        study_id = payload["study_id"]
        client.post("/api/v1/generate-report", json=payload)
        data = client.get(f"/api/v1/report/{study_id}").json()
        for field in ("study_id", "findings", "impression", "validated", "status"):
            assert field in data, f"Missing field: {field}"

    def test_get_nonexistent_report_returns_404(self, client):
        resp = client.get("/api/v1/report/DOES_NOT_EXIST_ZZZZZ")
        assert resp.status_code == 404


# ====================================================================== #
#  PUT /api/v1/report/{study_id}
# ====================================================================== #


class TestUpdateReport:
    def test_update_impression(self, client):
        payload = load_sample("single_lesion.json")
        study_id = payload["study_id"]
        client.post("/api/v1/generate-report", json=payload)

        update_resp = client.put(
            f"/api/v1/report/{study_id}",
            json={"impression": "Likely benign cyst. Follow-up in 12 months."},
        )
        assert update_resp.status_code == 200
        assert update_resp.json()["success"] is True

    def test_update_is_persisted(self, client):
        payload = load_sample("single_lesion.json")
        study_id = payload["study_id"]
        client.post("/api/v1/generate-report", json=payload)

        impression = "Test impression value."
        client.put(f"/api/v1/report/{study_id}", json={"impression": impression})

        data = client.get(f"/api/v1/report/{study_id}").json()
        assert data["impression"] == impression

    def test_update_findings(self, client):
        payload = load_sample("single_lesion.json")
        study_id = payload["study_id"]
        client.post("/api/v1/generate-report", json=payload)

        new_findings = "Manually edited findings text."
        resp = client.put(f"/api/v1/report/{study_id}", json={"findings": new_findings})
        assert resp.status_code == 200

    def test_update_nonexistent_returns_404(self, client):
        resp = client.put(
            "/api/v1/report/NONEXISTENT_XYZ",
            json={"impression": "test"},
        )
        assert resp.status_code == 404


# ====================================================================== #
#  POST /api/v1/sign-report
# ====================================================================== #


class TestSignReport:
    def test_sign_report_returns_200(self, client):
        payload = load_sample("multi_lesion.json")
        study_id = payload["study_id"]
        client.post("/api/v1/generate-report", json=payload)

        resp = client.post("/api/v1/sign-report", json={"study_id": study_id})
        assert resp.status_code == 200

    def test_sign_report_response_schema(self, client):
        payload = load_sample("multi_lesion.json")
        study_id = payload["study_id"]
        client.post("/api/v1/generate-report", json=payload)

        data = client.post("/api/v1/sign-report", json={"study_id": study_id}).json()
        assert data["saved"] is True
        assert data["status"] == "signed"

    def test_sign_report_status_persisted(self, client):
        payload = load_sample("multi_organ.json")
        study_id = payload["study_id"]
        client.post("/api/v1/generate-report", json=payload)
        client.post("/api/v1/sign-report", json={"study_id": study_id})

        data = client.get(f"/api/v1/report/{study_id}").json()
        assert data["status"] == "signed"

    def test_signed_report_cannot_be_edited(self, client):
        payload = load_sample("multi_organ.json")
        study_id = payload["study_id"]
        client.post("/api/v1/generate-report", json=payload)
        client.post("/api/v1/sign-report", json={"study_id": study_id})

        resp = client.put(f"/api/v1/report/{study_id}", json={"impression": "Late edit"})
        assert resp.status_code == 409

    def test_sign_nonexistent_returns_404(self, client):
        resp = client.post("/api/v1/sign-report", json={"study_id": "DOES_NOT_EXIST"})
        assert resp.status_code == 404

    def test_sign_missing_study_id_returns_422(self, client):
        resp = client.post("/api/v1/sign-report", json={})
        assert resp.status_code == 422


# ====================================================================== #
#  GET /api/v1/health
# ====================================================================== #


class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_has_status_ok(self, client):
        data = client.get("/api/v1/health").json()
        assert data["status"] == "ok"


# ====================================================================== #
#  GET /
# ====================================================================== #


class TestRoot:
    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_has_service_name(self, client):
        data = client.get("/").json()
        assert "PRISM" in data["service"]
