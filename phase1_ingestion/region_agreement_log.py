"""
region_agreement_log.py — Phase 1 vs Phase 2 Region Agreement Logger

Records (series_uid, phase1_guessed_region, package3_actual_region) per
series to logs/region_agreement.csv.  Over enough series this becomes a
confusion matrix for empirically recalibrating the Step 2B histogram
heuristics in triage_screen.py.

Called from lifecycle_manager.py after Phase 2 determines the actual region
via region_detector.py.  The phase1_guessed_region is captured into
series_meta at export_volume() trigger time (pipeline.py) so it survives the
multiprocessing boundary without any shared-memory IPC.

Usage:
    from phase1_ingestion.region_agreement_log import log_region_agreement

    log_region_agreement(
        series_uid="1.2.3...",
        phase1_guess="chest",
        package3_actual="cardiothoracic",
    )
"""

import csv
import os
import threading
from datetime import datetime, timezone

# ── Module-level thread lock ───────────────────────────────────────────────────
# Serialises concurrent writes from multiple Phase 2 processes that may run
# back-to-back on fast scanners.
_lock = threading.Lock()

LOG_PATH = os.path.join("logs", "region_agreement.csv")

_HEADER = [
    "timestamp_utc",
    "series_uid",
    "phase1_guess",
    "package3_actual",
    "match",
]


def log_region_agreement(
    series_uid: str,
    phase1_guess: str,
    package3_actual: str,
) -> None:
    """
    Append one row to logs/region_agreement.csv.

    Creates the logs/ directory and writes the CSV header on first call.
    Thread-safe for concurrent writers.

    Args:
        series_uid:       DICOM SeriesInstanceUID for this series.
        phase1_guess:     Body region guessed by Phase 1's majority vote
                          (e.g. "chest", "abdomen", "unknown").
        package3_actual:  Body region determined by region_detector.py from
                          TotalSegmentator labels (e.g. "cardiothoracic",
                          "abdomen_pelvis").
    """
    os.makedirs("logs", exist_ok=True)

    match = phase1_guess.lower() == package3_actual.lower()
    row = [
        datetime.now(timezone.utc).isoformat(),
        series_uid,
        phase1_guess,
        package3_actual,
        str(match),
    ]

    with _lock:
        write_header = not os.path.isfile(LOG_PATH)
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if write_header:
                writer.writerow(_HEADER)
            writer.writerow(row)
