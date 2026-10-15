"""
prior_study_lookup.py — Prior-study comparator lookup for the COMPARISON section.

Scans the DICOM archive (exports/*.zip, written by dicom_archive.py) for earlier
archived studies of the same patient. It reports only what can be established
from archive filenames: that a prior study exists and when it was archived.
PRISM does NOT automatically compare imaging between studies, and the report
text says so explicitly.

Archive naming (see DicomArchiveManager._archive_current_session):
    patient_{name}_{patientid}[_age_X][_sex_Y]_study_{uid16}_{YYYYmmdd_HHMMSS}.zip
"""

import os
import re
import logging
from datetime import datetime
from typing import List

logger = logging.getLogger(__name__)

_TS_RE = re.compile(r"(\d{8})_(\d{6})\.zip$")


def _safe(s: str, extra: str = "-_") -> str:
    return "".join(c for c in (s or "") if c.isalnum() or c in extra)


def find_prior_studies(
    exports_dir: str,
    patient_id: str,
    current_study_uid: str = "",
    max_results: int = 3,
) -> List[dict]:
    """
    Return prior archived studies for ``patient_id`` (newest first).

    Each entry: {"archive": filename, "archived_on": "YYYY-MM-DD"}.
    The current study (matched on StudyInstanceUID suffix) is excluded, so
    re-sending the same dataset is never reported as its own comparator.
    """
    if not patient_id or patient_id in ("UNKNOWN_PATIENT", "UNK", "PATIENT_INCOMING"):
        return []
    if not os.path.isdir(exports_dir):
        return []

    token = f"_{_safe(patient_id)[:32]}_"
    current_suffix = _safe(current_study_uid)[-16:] if current_study_uid else ""

    priors = []
    try:
        for name in os.listdir(exports_dir):
            if not name.lower().endswith(".zip") or name.startswith(".tmp_"):
                continue
            if token not in name:
                continue
            if current_suffix and f"study_{current_suffix}" in name:
                continue
            m = _TS_RE.search(name)
            if not m:
                continue
            try:
                dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            except ValueError:
                continue
            priors.append({"archive": name, "archived_on": dt.strftime("%Y-%m-%d"), "_dt": dt})
    except OSError as e:
        logger.warning("prior_study_lookup: could not scan %s: %s", exports_dir, e)
        return []

    priors.sort(key=lambda p: p["_dt"], reverse=True)
    for p in priors:
        p.pop("_dt", None)
    return priors[:max_results]
