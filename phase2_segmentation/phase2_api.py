"""
phase2_api.py — Package 7: Findings Serialization & Phase 3 Handoff

Serializes surviving candidates to findings.json and optionally HTTP-POSTs
to Phase 3's /generate-report endpoint.
"""

import os
import json
import time
import logging
import httpx
from datetime import datetime, timezone

from phase2_segmentation.findings_adapter import to_structured_findings

logger = logging.getLogger(__name__)

def export_findings(candidates: list, run_dir: str, series_meta: dict, region: str) -> str:
    """
    Serializes non-suppressed candidates to findings.json matching the
    schemas/findings_output.json envelope.

    Args:
        candidates: List of Candidate objects.
        run_dir: The phase2_work/run_<timestamp> directory.
        series_meta: Dictionary of DICOM series metadata.

    Returns:
        Absolute path to the generated findings.json file.
    """
    survivors = [c for c in candidates if not c.suppressed]
    
    envelope = {
        "series_instance_uid": series_meta.get("series_instance_uid", "unknown"),
        "region": region,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(survivors),
        "findings": [c.to_dict() for c in survivors]
    }
    
    seg_dir = os.path.join(run_dir, "temp_seg")
    os.makedirs(seg_dir, exist_ok=True)
    
    out_path = os.path.abspath(os.path.join(seg_dir, "findings.json"))
    
    with open(out_path, "w") as f:
        json.dump(envelope, f, indent=2)
        
    return out_path

def trigger_phase3(
    candidates: list, 
    series_meta: dict, 
    phase3_url: str = None,
    target_organs: list = None
) -> bool:
    """
    HTTP POST the structured findings to Phase 3's /generate-report endpoint.
    Guarded by PHASE3_ENABLED environment variable.

    Args:
        candidates: List of Candidate objects.
        series_meta: Dictionary of DICOM series metadata.
        phase3_url: Phase 3 REST API endpoint. Overrides env var if provided.

    Returns:
        True if POST succeeded, False otherwise (never raises exceptions).
    """
    if os.environ.get("PHASE3_ENABLED", "false").lower() != "true":
        logger.info("PHASE3_ENABLED is false; skipping HTTP POST to Phase 3.")
        return False
        
    url = phase3_url or os.environ.get("PHASE3_URL", "http://localhost:8000/api/v1/generate-report")
    
    try:
        payload = to_structured_findings(candidates, series_meta, target_organs=target_organs)
        
        # Exponential backoff retry logic for cold-start (max 3 retries, 60s timeout)
        max_retries = 3
        retry_delay = 5.0
        
        with httpx.Client(timeout=180.0) as client:
            for attempt in range(max_retries):
                try:
                    response = client.post(url, json=payload)
                    if response.status_code in (200, 201):
                        logger.info("Successfully triggered Phase 3 report generation.")
                        return True
                    else:
                        logger.warning(
                            "Phase 3 returned unexpected status: %d - %s", 
                            response.status_code, 
                            response.text
                        )
                        return False
                except (httpx.ConnectError, httpx.ReadTimeout) as e:
                    if attempt < max_retries - 1:
                        logger.warning("Failed to reach Phase 3 (attempt %d/%d): %s. Retrying in %.1fs...", 
                                       attempt + 1, max_retries, e, retry_delay)
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        logger.error("Failed to reach Phase 3 after %d attempts: %s", max_retries, e)
                        return False
                        
    except Exception as e:
        logger.warning("Unexpected error triggering Phase 3: %s", e, exc_info=True)
        return False
