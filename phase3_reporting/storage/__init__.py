"""Storage package – abstract layer + JSON implementation."""

from .storage import ReportStorage, get_storage
from .json_storage import JSONReportStorage

__all__ = ["ReportStorage", "get_storage", "JSONReportStorage"]
