import os
import json
import time
import logging
import httpx
from datetime import datetime, timezone

from phase2_segmentation.findings_adapter import to_structured_findings

logger = logging.getLogger(__name__)


def export_findings(candidates: list, run_dir: str, series_meta: dict, region: str) -> str:
    survivors = [c for c in candidates if not c.suppressed]

    envelope = {
        "series_instance_uid": series_meta.get("series_instance_uid", "unknown"),
        "study_instance_uid": series_meta.get("study_instance_uid", ""),
        "patient_id": series_meta.get("patient_id", "UNKNOWN_PATIENT"),
        "patient_name": series_meta.get("patient_name", "Unknown Patient"),
        "region": region,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(survivors),
        "findings": [c.to_dict() for c in survivors],
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
    target_organs: list = None,
    organ_stats: dict = None,
    region: str = "",
    anatomy: dict = None,
) -> bool:
    if os.environ.get("PHASE3_ENABLED", "false").lower() != "true":
        logger.info("PHASE3_ENABLED is false; skipping Phase 3 handoff.")
        return False

    url = phase3_url or os.environ.get("PHASE3_URL", "http://localhost:8000/api/v1/generate-report")

    # Inject region into series_meta so findings_adapter can include it in the payload
    enriched_meta = dict(series_meta)
    if region:
        enriched_meta["phase2_region"] = region

    try:
        payload = to_structured_findings(
            candidates,
            enriched_meta,
            target_organs=target_organs,
            organ_stats=organ_stats,
            anatomy=anatomy,
        )

        max_retries = 3
        retry_delay = 5.0

        with httpx.Client(timeout=180.0) as client:
            for attempt in range(max_retries):
                try:
                    response = client.post(url, json=payload)
                    if response.status_code in (200, 201):
                        logger.info("Phase 3 report generation triggered successfully.")
                        return True
                    else:
                        logger.warning(
                            "Phase 3 returned unexpected status: %d - %s",
                            response.status_code,
                            response.text,
                        )
                        return False
                except (httpx.ConnectError, httpx.ReadTimeout) as e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            "Failed to reach Phase 3 (attempt %d/%d): %s. Retrying in %.1fs...",
                            attempt + 1, max_retries, e, retry_delay,
                        )
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        logger.error("Failed to reach Phase 3 after %d attempts: %s", max_retries, e)
                        return False

    except Exception as e:
        logger.warning("Unexpected error triggering Phase 3: %s", e, exc_info=True)
        return False
