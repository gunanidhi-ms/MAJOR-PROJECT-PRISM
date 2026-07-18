"""
storage.py
----------
Abstract storage layer for radiology reports.

Architecture note
~~~~~~~~~~~~~~~~~
All persistence goes through ``ReportStorage``.  No API or service code
imports ``JSONReportStorage`` directly; they depend only on the abstract
base class.

To swap in MySQL later:

    1.  Create ``MySQLReportStorage(ReportStorage)`` in a new module.
    2.  Update the ``get_storage()`` factory below.
    3.  No changes required in ``report_routes.py`` or any service.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from app.models import Report


class ReportStorage(ABC):
    """
    Abstract base class for the report persistence layer.

    Every concrete implementation must honour this contract.
    """

    @abstractmethod
    def save_report(self, report: Report) -> None:
        """
        Persist a report.

        Must be idempotent: calling with the same ``study_id`` replaces
        the previously stored version.

        Parameters
        ----------
        report : Report
            The fully-populated report model to save.
        """
        ...

    @abstractmethod
    def get_report(self, study_id: str) -> Optional[Report]:
        """
        Retrieve a report by study ID.

        Parameters
        ----------
        study_id : str

        Returns
        -------
        Report or None
            ``None`` if no report with this study_id exists.
        """
        ...

    @abstractmethod
    def list_reports(self) -> List[Report]:
        """
        Return all stored reports, newest first.

        Returns
        -------
        list[Report]
        """
        ...

    @abstractmethod
    def update_report(self, report: Report) -> None:
        """
        Update an existing report.

        Parameters
        ----------
        report : Report
        """
        ...

    @abstractmethod
    def archive_report(self, study_id: str) -> bool:
        """
        Archive a report (move to cold storage/mark as archived).

        Returns
        -------
        bool
            True if archived; False if not found.
        """
        ...

    @abstractmethod
    def delete_report(self, study_id: str) -> bool:
        """
        Delete a report.

        Returns
        -------
        bool
            True if deleted; False if not found.
        """
        ...


# ====================================================================== #
#  Factory
# ====================================================================== #

def get_storage() -> ReportStorage:
    """
    Return the active storage implementation.

    Reads ``STORAGE_BACKEND`` from settings (default: "json").
    Extend this function when adding new backends (e.g. "mysql").
    """
    from app.config import get_settings
    from storage.json_storage import JSONReportStorage

    settings = get_settings()

    # Future: if settings.storage_backend == "mysql": return MySQLReportStorage(...)
    return JSONReportStorage(reports_dir=settings.reports_dir)
