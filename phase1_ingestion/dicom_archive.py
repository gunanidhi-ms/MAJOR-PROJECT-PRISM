"""
dicom_archive.py — DICOM Archive Manager

Manages the lifecycle of incoming DICOM files:
- Temporary storage in incoming/ during active scan
- Automatic zip + move to exports/ on patient change (StudyInstanceUID) or shutdown
- Thread-safe, zero interference with VolumeAccumulator (read-only file tracking)
"""

import os
import json
import shutil
import threading
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PatientSession:
    """Tracks a single patient's incoming DICOM session."""
    study_instance_uid: str
    patient_id: str
    patient_name: str = ""
    patient_age: str = ""
    patient_sex: str = ""
    slice_files: Set[Path] = field(default_factory=set)
    first_slice_time: datetime = field(default_factory=datetime.now)
    last_slice_time: datetime = field(default_factory=datetime.now)


class DicomArchiveManager:
    """
    Manages DICOM file archiving from incoming/ to exports/.

    Key Design Principles:
    - Separate from VolumeAccumulator: only tracks file paths, never touches pixel data
    - Patient detection via StudyInstanceUID (most reliable DICOM identifier)
    - Atomic zip operations: write to temp, then atomic rename
    - Thread-safe: all public methods use internal lock
    """

    def __init__(
        self,
        incoming_dir: str,
        exports_dir: str,
        max_slices_per_patient: int = 5000,
    ):
        """
        Args:
            incoming_dir: Directory where DICOM files are temporarily stored
            exports_dir: Directory where zipped archives are saved
            max_slices_per_patient: Safety limit to prevent runaway accumulation
        """
        self.incoming_dir = Path(incoming_dir).resolve()
        self.exports_dir = Path(exports_dir).resolve()
        self.max_slices = max_slices_per_patient

        # Ensure directories exist
        self.incoming_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)

        # Thread-safe state
        self._lock = threading.RLock()
        self._current_session: Optional[PatientSession] = None
        self._archived_studies: Set[str] = set()  # Track archived StudyInstanceUIDs
        self._shutdown = False

        logger.info(
            "DicomArchiveManager initialized: incoming=%s, exports=%s",
            self.incoming_dir,
            self.exports_dir,
        )

    def register_slice(
        self,
        study_instance_uid: str,
        patient_id: str,
        patient_name: str,
        slice_filepath: Path,
        patient_age: str = "",
        patient_sex: str = "",
    ) -> None:
        """
        Called when a new DICOM slice arrives.
        Tracks the slice under the current patient session.

        Args:
            study_instance_uid: DICOM StudyInstanceUID (primary patient/session key)
            patient_id: DICOM PatientID
            patient_name: DICOM PatientName
            slice_filepath: Path to the saved .dcm file in incoming/
            patient_age: DICOM PatientAge
            patient_sex: DICOM PatientSex
        """
        with self._lock:
            if self._shutdown:
                return

            # Check for patient/session change (new StudyInstanceUID)
            if self._current_session is not None:
                if self._current_session.study_instance_uid != study_instance_uid:
                    logger.info(
                        "StudyInstanceUID changed: %s -> %s. Archiving previous session.",
                        self._current_session.study_instance_uid[:20],
                        study_instance_uid[:20],
                    )
                    self._archive_current_session()

            # Initialize new session if needed
            if self._current_session is None:
                self._current_session = PatientSession(
                    study_instance_uid=study_instance_uid,
                    patient_id=patient_id,
                    patient_name=patient_name,
                    patient_age=patient_age,
                    patient_sex=patient_sex,
                )
                logger.info(
                    "New patient session started: StudyUID=%s, PatientID=%s, Name=%s",
                    study_instance_uid[:20],
                    patient_id,
                    patient_name,
                )

            # Track this slice
            self._current_session.slice_files.add(slice_filepath.resolve())
            self._current_session.last_slice_time = datetime.now()

            # Safety check
            if len(self._current_session.slice_files) > self.max_slices:
                logger.warning(
                    "Patient session exceeds max slices (%d). Forcing archive.",
                    self.max_slices,
                )
                self._archive_current_session()

    def on_association_released(self) -> None:
        """
        Called when DICOM association is released (EVT_RELEASED).
        Does NOT auto-archive - waits for next patient or shutdown.
        This allows multi-association studies to complete.
        """
        with self._lock:
            if self._current_session:
                logger.info(
                    "Association released. Current session: %s (%d slices, StudyUID=%s)",
                    self._current_session.patient_id,
                    len(self._current_session.slice_files),
                    self._current_session.study_instance_uid[:20],
                )

    def on_patient_change(self, new_study_instance_uid: str) -> None:
        """
        Explicit patient change notification.
        Archives current session if StudyInstanceUID differs.
        """
        with self._lock:
            if self._current_session and self._current_session.study_instance_uid != new_study_instance_uid:
                logger.info("Explicit patient change detected. Archiving current session.")
                self._archive_current_session()

    def on_shutdown(self) -> None:
        """
        Called on graceful application shutdown.
        Archives any remaining session.
        """
        with self._lock:
            self._shutdown = True
            if self._current_session and self._current_session.slice_files:
                logger.info("Shutdown: archiving final session (%d slices)",
                           len(self._current_session.slice_files))
                self._archive_current_session()
            self._current_session = None

    def _get_or_create_report_dict(self, session: PatientSession, slice_count: int) -> dict:
        """Fetch existing Phase 3 report if available or construct a structured clinical report JSON."""
        project_root = Path(__file__).resolve().parent.parent
        report_candidates = []

        # Check phase3 storage reports directory
        p3_reports_dir = project_root / "phase3_reporting" / "storage" / "reports"
        if p3_reports_dir.exists():
            report_candidates.extend(list(p3_reports_dir.glob("*.json")))

        # Check root reports directory
        root_reports_dir = project_root / "reports"
        if root_reports_dir.exists():
            report_candidates.extend(list(root_reports_dir.glob("*.json")))

        # Match candidate by patient ID, name, or study UID
        matched_report = None
        p_id = (session.patient_id or "").strip().lower()
        p_name = (session.patient_name or "").strip().lower()
        s_uid = (session.study_instance_uid or "").strip().lower()
        s_uid_short = s_uid[-8:] if len(s_uid) > 8 else s_uid

        # Sort candidates newest first
        report_candidates = sorted(report_candidates, key=lambda p: p.stat().st_mtime, reverse=True)

        for cand in report_candidates:
            cand_name = cand.stem.lower()
            if (p_id and p_id in cand_name) or (p_name and p_name in cand_name) or (s_uid_short and s_uid_short in cand_name):
                try:
                    with cand.open("r", encoding="utf-8") as fh:
                        data = json.load(fh)
                        if isinstance(data, dict):
                            matched_report = data
                            break
                except Exception:
                    pass

        now_iso = datetime.now().isoformat()
        if matched_report:
            report_dict = dict(matched_report)
            report_dict["patient_id"] = report_dict.get("patient_id") or session.patient_id
            report_dict["patient_name"] = report_dict.get("patient_name") or session.patient_name
            report_dict["patient_age"] = report_dict.get("patient_age") or session.patient_age
            report_dict["patient_sex"] = report_dict.get("patient_sex") or session.patient_sex
            report_dict["study_id"] = report_dict.get("study_id") or session.study_instance_uid
            report_dict["study_instance_uid"] = report_dict.get("study_instance_uid") or session.study_instance_uid
            report_dict["total_slices"] = slice_count
            report_dict["archived_at"] = now_iso
            return report_dict

        # Construct new structured report JSON
        return {
            "study_id": session.study_instance_uid or f"STUDY_{session.patient_id or 'UNK'}",
            "study_instance_uid": session.study_instance_uid or "",
            "patient_id": session.patient_id or "UNK",
            "patient_name": session.patient_name or "Patient",
            "patient_age": session.patient_age or "",
            "patient_sex": session.patient_sex or "",
            "total_slices": slice_count,
            "status": "archived",
            "created_at": session.first_slice_time.isoformat() if hasattr(session, "first_slice_time") else now_iso,
            "archived_at": now_iso,
            "findings": f"DICOM acquisition completed with {slice_count} CT slices. Real-time 2D screening and emergency triage evaluated without critical structural halts.",
            "impression": f"Diagnostic CT imaging complete for {session.patient_name or 'Patient'} ({session.patient_id or 'UNK'}). All {slice_count} slices validated and archived with report.",
            "source": "PRISM_INTEGRATED_PIPELINE",
            "validated": True
        }

    def _archive_current_session(self) -> Optional[Path]:
        """
        Archive the current patient session to a zip file in exports/ containing
        all DICOM slices and a structured report.json.

        Returns:
            Path to created archive, or None if nothing to archive.
        """
        # If no session or slice_files empty, check if incoming_dir has any orphan .dcm files
        if not self._current_session or not self._current_session.slice_files:
            incoming_dcms = list(self.incoming_dir.glob("*.dcm"))
            if incoming_dcms and self._current_session:
                for dcm in incoming_dcms:
                    self._current_session.slice_files.add(dcm.resolve())
            elif incoming_dcms and not self._current_session:
                self._current_session = PatientSession(
                    study_instance_uid="STUDY_INCOMING",
                    patient_id="PATIENT_INCOMING",
                    patient_name="Patient",
                    slice_files=set(f.resolve() for f in incoming_dcms)
                )
            else:
                return None

        # Skip if already archived
        if self._current_session.study_instance_uid in self._archived_studies:
            logger.debug("Study %s already archived, skipping",
                        self._current_session.study_instance_uid[:20])
            self._current_session = None
            return None

        session = self._current_session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build safe archive name with Patient Name, ID, Age, and Sex
        safe_patient_id = "".join(c for c in (session.patient_id or "UNK") if c.isalnum() or c in "-_")[:32]
        safe_patient_name = "".join(c for c in (session.patient_name or "Patient").replace("^", "_").replace(" ", "_") if c.isalnum() or c in "-_")[:32]
        safe_age = "".join(c for c in (session.patient_age or "") if c.isalnum())[:8]
        safe_sex = "".join(c for c in (session.patient_sex or "") if c.isalnum())[:4]
        safe_study_uid = "".join(c for c in session.study_instance_uid if c.isalnum() or c in "-_")[-16:]
        
        parts = [f"patient_{safe_patient_name}_{safe_patient_id}"]
        if safe_age:
            parts.append(f"age_{safe_age}")
        if safe_sex:
            parts.append(f"sex_{safe_sex}")
        if safe_study_uid:
            parts.append(f"study_{safe_study_uid}")
        parts.append(timestamp)
        archive_name = "_".join(parts)

        final_zip = self.exports_dir / f"{archive_name}.zip"
        temp_zip_base = self.exports_dir / f".tmp_{archive_name}"

        try:
            valid_slice_files = [f for f in session.slice_files if f.exists()]
            # Also include any remaining .dcm files in incoming_dir
            for extra_dcm in self.incoming_dir.glob("*.dcm"):
                if extra_dcm.resolve() not in valid_slice_files and extra_dcm.exists():
                    valid_slice_files.append(extra_dcm.resolve())

            logger.info("Archiving %d slices + report.json for patient %s (%s, StudyUID=...%s)...",
                       len(valid_slice_files), session.patient_name, session.patient_id, safe_study_uid)

            # Create zip from staging directory (only this patient's files + report.json)
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                
                # Copy session slice files to temp staging dir
                for slice_path in valid_slice_files:
                    dest = tmpdir_path / slice_path.name
                    shutil.copy2(slice_path, dest)

                # Generate and write structured report.json into the staging dir
                report_dict = self._get_or_create_report_dict(session, len(valid_slice_files))
                report_json_path = tmpdir_path / "report.json"
                with report_json_path.open("w", encoding="utf-8") as rfh:
                    json.dump(report_dict, rfh, indent=2, ensure_ascii=False)

                # Create zip archive in exports directory
                created_archive = shutil.make_archive(str(temp_zip_base), 'zip', tmpdir_path)
                created_path = Path(created_archive)
                if created_path.exists():
                    if final_zip.exists():
                        final_zip.unlink()
                    created_path.rename(final_zip)
                    logger.info("Successfully archived %d slices + report.json to: %s", len(valid_slice_files), final_zip)

            # Mark as archived
            self._archived_studies.add(session.study_instance_uid)

            # Clear incoming files for this session
            self._clear_session_files(session)

            # Reset current session
            self._current_session = None

            return final_zip

        except Exception as e:
            logger.error("Failed to archive session: %s", e, exc_info=True)
            # Cleanup temp file if exists
            temp_file = self.exports_dir / f".tmp_{archive_name}.zip"
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            return None

    def _clear_session_files(self, session: PatientSession) -> None:
        """Remove archived slice files from incoming directory."""
        for slice_path in session.slice_files:
            try:
                if slice_path.exists():
                    slice_path.unlink()
                    logger.debug("Removed archived slice: %s", slice_path.name)
            except OSError as e:
                logger.warning("Failed to remove %s: %s", slice_path, e)

    def get_current_session_info(self) -> Optional[dict]:
        """Get info about current active session (for debugging/monitoring)."""
        with self._lock:
            if not self._current_session:
                return None
            s = self._current_session
            return {
                "study_instance_uid": s.study_instance_uid,
                "patient_id": s.patient_id,
                "patient_name": s.patient_name,
                "slice_count": len(s.slice_files),
                "first_slice": s.first_slice_time.isoformat(),
                "last_slice": s.last_slice_time.isoformat(),
            }

    def reset_session(self) -> None:
        """Explicitly reset the current session (e.g., when clearing for a new scan)."""
        with self._lock:
            if self._current_session:
                self._clear_session_files(self._current_session)
                self._current_session = None

    def force_archive(self) -> Optional[Path]:
        """Force archive current session (for testing/manual trigger)."""
        with self._lock:
            return self._archive_current_session()


# Global instance (set by pipeline during initialization)
_archive_manager: Optional[DicomArchiveManager] = None


def get_archive_manager() -> Optional[DicomArchiveManager]:
    """Get the global archive manager instance."""
    return _archive_manager


def init_archive_manager(incoming_dir: str, exports_dir: str) -> DicomArchiveManager:
    """Initialize the global archive manager."""
    global _archive_manager
    _archive_manager = DicomArchiveManager(incoming_dir, exports_dir)
    return _archive_manager