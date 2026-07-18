"""
tests/test_storage.py
----------------------
Unit tests for the JSON storage layer.

Run with:
    pytest tests/test_storage.py -v
"""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.models import Report
from storage.json_storage import JSONReportStorage


@pytest.fixture
def tmp_storage(tmp_path) -> JSONReportStorage:
    """Return a JSONReportStorage instance backed by a temporary directory."""
    return JSONReportStorage(reports_dir=tmp_path)


def make_report(study_id: str = "TEST001", status: str = "draft") -> Report:
    now = datetime.now(timezone.utc)
    return Report(
        study_id=study_id,
        modality="CT",
        protocol="CT KUB",
        findings="The left kidney appears normal.",
        impression="",
        validated=True,
        status=status,
        source="llm",
        created_at=now,
        updated_at=now,
    )


# ====================================================================== #
#  save_report
# ====================================================================== #


class TestSaveReport:
    def test_save_creates_file(self, tmp_storage, tmp_path):
        report = make_report()
        tmp_storage.save_report(report)
        files = list(tmp_path.glob("TEST001_*.json"))
        assert len(files) == 1

    def test_saved_file_is_valid_json(self, tmp_storage, tmp_path):
        tmp_storage.save_report(make_report())
        file = next(tmp_path.glob("TEST001_*.json"))
        data = json.loads(file.read_text(encoding="utf-8"))
        assert data["study_id"] == "TEST001"

    def test_save_multiple_creates_multiple_files(self, tmp_storage, tmp_path):
        import time
        tmp_storage.save_report(make_report("STUDY_A"))
        time.sleep(0.01)
        tmp_storage.save_report(make_report("STUDY_A"))
        files = list(tmp_path.glob("STUDY_A_*.json"))
        assert len(files) == 2


# ====================================================================== #
#  get_report
# ====================================================================== #


class TestGetReport:
    def test_get_returns_correct_study_id(self, tmp_storage):
        tmp_storage.save_report(make_report("STUDY_X"))
        result = tmp_storage.get_report("STUDY_X")
        assert result is not None
        assert result.study_id == "STUDY_X"

    def test_get_returns_none_for_missing(self, tmp_storage):
        assert tmp_storage.get_report("NONEXISTENT") is None

    def test_get_returns_latest_after_multiple_saves(self, tmp_storage):
        import time
        tmp_storage.save_report(make_report("STUDY_Y", status="draft"))
        time.sleep(0.05)
        updated = make_report("STUDY_Y", status="signed")
        tmp_storage.save_report(updated)
        result = tmp_storage.get_report("STUDY_Y")
        assert result is not None
        assert result.status == "signed"

    def test_get_preserves_findings_text(self, tmp_storage):
        report = make_report()
        report.findings = "Unique findings string 12345."
        tmp_storage.save_report(report)
        loaded = tmp_storage.get_report(report.study_id)
        assert loaded.findings == "Unique findings string 12345."


# ====================================================================== #
#  list_reports
# ====================================================================== #


class TestListReports:
    def test_list_empty_storage(self, tmp_storage):
        assert tmp_storage.list_reports() == []

    def test_list_returns_correct_count(self, tmp_storage):
        tmp_storage.save_report(make_report("S1"))
        tmp_storage.save_report(make_report("S2"))
        tmp_storage.save_report(make_report("S3"))
        reports = tmp_storage.list_reports()
        assert len(reports) == 3

    def test_list_deduplicates_by_study_id(self, tmp_storage):
        import time
        tmp_storage.save_report(make_report("SAME"))
        time.sleep(0.05)
        tmp_storage.save_report(make_report("SAME"))
        reports = tmp_storage.list_reports()
        study_ids = [r.study_id for r in reports]
        assert study_ids.count("SAME") == 1


# ====================================================================== #
#  delete_report
# ====================================================================== #


class TestDeleteReport:
    def test_delete_existing(self, tmp_storage, tmp_path):
        tmp_storage.save_report(make_report("DEL001"))
        deleted = tmp_storage.delete_report("DEL001")
        assert deleted is True
        files = list(tmp_path.glob("DEL001_*.json"))
        assert len(files) == 0

    def test_delete_nonexistent_returns_false(self, tmp_storage):
        assert tmp_storage.delete_report("GHOST") is False

    def test_get_after_delete_returns_none(self, tmp_storage):
        tmp_storage.save_report(make_report("WILL_BE_DELETED"))
        tmp_storage.delete_report("WILL_BE_DELETED")
        assert tmp_storage.get_report("WILL_BE_DELETED") is None
