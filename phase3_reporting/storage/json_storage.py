"""
json_storage.py
---------------
JSON file-based implementation of ``ReportStorage``.

Each report is stored as an individual JSON file named:

    reports/{study_id}_{timestamp}.json

The most recent file for a given study_id is treated as the canonical
version (``get_report`` returns it; ``save_report`` creates a new
timestamped file).

This makes the storage append-only by default, giving a natural audit
trail.  The ``delete_report`` method removes ALL files for a study_id.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from app.models import Report
from .storage import ReportStorage

logger = logging.getLogger(__name__)


class JSONReportStorage(ReportStorage):
    """
    Stores reports as pretty-printed JSON files on the local file system.

    Parameters
    ----------
    reports_dir : str | Path
        Directory where report JSON files are written.
        Created automatically if it does not exist.
    """

    def __init__(self, reports_dir: str | Path = "reports") -> None:
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        logger.info("JSONReportStorage initialised: dir=%s", self.reports_dir.resolve())

    # ------------------------------------------------------------------ #
    #  ReportStorage interface
    # ------------------------------------------------------------------ #

    def save_report(self, report: Report) -> None:
        """
        Write the report to a new timestamped JSON file.

        The file name format is ``{study_id}_{YYYYMMDD_HHMMSS}.json``,
        providing a human-readable audit trail.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        p_id   = (report.patient_id   or "UNK").replace(" ", "_")[:16]
        p_name = (report.patient_name or "Patient").replace(" ", "_").replace("^", "_")[:20]
        # Use only the last 8 chars of the DICOM UID — long enough to be unique,
        # short enough to keep the filename human-readable on disk.
        uid_short = report.study_id[-8:] if len(report.study_id) > 8 else report.study_id

        filename = f"{p_name}_{p_id}_{timestamp}_{uid_short}.json"
        filepath = self.reports_dir / filename

        payload = report.model_dump(mode="json")
        # Convert datetime objects to ISO strings (model_dump mode="json" handles this)

        with filepath.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str, ensure_ascii=False)

        logger.info("Report saved: %s", filepath)

    def get_report(self, study_id: str) -> Optional[Report]:
        """
        Return the most recently saved report matching *study_id*
        (Patient ID, Patient Name, Study UID, or filename stem).
        """
        if not study_id:
            return None

        clean_id = study_id.strip().lower()
        uid_short = study_id[-8:] if len(study_id) > 8 else study_id

        # Gather all candidate report JSON files sorted by newest first
        candidates: List[Path] = sorted(
            set(
                list(self.reports_dir.glob(f"*{study_id}*.json"))
                + list(self.reports_dir.glob(f"*_{uid_short}.json"))
                + list(self.reports_dir.glob("*.json"))
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for p in candidates:
            report = self._load(p)
            if not report:
                continue

            p_id = str(report.patient_id or "").strip().lower()
            p_name = str(report.patient_name or "").strip().lower()
            s_id = str(report.study_id or "").strip().lower()
            stem = p.stem.lower()

            # Match on Patient ID (exact or clean match)
            if p_id and (p_id == clean_id or clean_id in p_id or p_id in clean_id):
                logger.info("Found report for Patient ID '%s' in %s", study_id, p.name)
                return report

            # Match on Patient Name
            if p_name and clean_id in p_name:
                logger.info("Found report for Patient Name '%s' in %s", study_id, p.name)
                return report

            # Match on Study ID
            if s_id and (s_id == clean_id or s_id.endswith(clean_id) or clean_id in s_id):
                logger.info("Found report for Study ID '%s' in %s", study_id, p.name)
                return report

            # Match on filename stem
            if clean_id in stem:
                logger.info("Found report for filename '%s' in %s", study_id, p.name)
                return report

        logger.warning("No report found for identifier=%s in %s", study_id, self.reports_dir)
        return None

    def list_reports(self) -> List[Report]:
        """
        Return all reports (deduplicated by study_id – latest per study).
        """
        all_files = sorted(
            self.reports_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        seen: set[str] = set()
        reports: List[Report] = []

        for fp in all_files:
            report = self._load(fp)
            if report and report.study_id not in seen:
                seen.add(report.study_id)
                reports.append(report)

        return reports

    def delete_report(self, study_id: str) -> bool:
        """
        Remove all JSON files for *study_id*.

        Returns True if at least one file was deleted, False otherwise.
        """
        count = 0
        all_files = list(self.reports_dir.glob("*.json"))
        for p in all_files:
            report = self._load(p)
            if report and report.study_id == study_id:
                try:
                    p.unlink()
                    count += 1
                except FileNotFoundError:
                    pass
        return count > 0

    def update_report(self, report: Report) -> None:
        """
        Updates an existing report.
        For JSON storage, this just creates a new timestamped file (append-only log).
        """
        self.save_report(report)

    def archive_report(self, study_id: str) -> bool:
        """
        Archives a report.
        Stub implementation for JSON storage.
        """
        # A real implementation might move it to an archive folder.
        # For now, if the report exists, we just say it's archived.
        return self.get_report(study_id) is not None

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load(filepath: Path) -> Optional[Report]:
        """Parse a JSON file into a Report, swallowing errors gracefully."""
        try:
            with filepath.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            return Report.model_validate(data)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to parse report file %s: %s", filepath, exc)
            return None
