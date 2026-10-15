from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging

from schemas.candidate import Candidate

logger = logging.getLogger(__name__)

MIN_CONFIDENCE: float = 0.45
MIN_VOLUME_CC: float = 0.3
MAX_ANOMALIES_PER_ORGAN: int = 5

# Minimum coverage threshold per organ class (cc).
# TotalSegmentator always attempts all 104 organ labels.  When an organ is only
# partially within the scan FOV its volume is far below this threshold — we
# treat it as "not meaningfully present" and exclude it from the report.
# Rule of thumb: ~20 % of the lower bound of the normal adult range.
# Organs with None are always included (no minimum, e.g. lesions, GI tract).
_MIN_COVERAGE_CC: Dict[str, float] = {
    # Solid abdominal
    "liver":                  150.0,  # normal 1000–2200 cc
    "spleen":                  15.0,  # normal 60–350 cc
    "pancreas":                 8.0,  # normal 30–120 cc
    "gallbladder":              2.0,  # normal 5–60 cc
    # Renal
    "kidney_right":            25.0,  # normal 100–250 cc
    "kidney_left":             25.0,
    "adrenal_gland_right":      1.0,  # normal 2–15 cc
    "adrenal_gland_left":       1.0,
    # GI (highly variable; just require a non-trivial slab)
    "stomach":                 20.0,
    "small_bowel":             30.0,
    "duodenum":                 3.0,
    "colon":                   30.0,
    "esophagus":                2.0,
    "rectum":                   5.0,
    # Cardiovascular (only meaningful if chest CT covers them fully)
    "heart":                   80.0,  # normal 400–900 cc
    "aorta":                    5.0,  # normal 30–120 cc
    "inferior_vena_cava":       3.0,
    "portal_vein_and_splenic_vein": 3.0,
    "pulmonary_vein":           1.5,
    # Pulmonary (each lobe ~ 300–800 cc when fully imaged)
    "lung_upper_lobe_left":    60.0,
    "lung_lower_lobe_left":    60.0,
    "lung_upper_lobe_right":   60.0,
    "lung_middle_lobe_right":  30.0,
    "lung_lower_lobe_right":   60.0,
    "trachea":                  2.0,
    # Musculoskeletal — vertebrae ~30–60 cc each when fully imaged
    "vertebrae_l1":            10.0,
    "vertebrae_l2":            10.0,
    "vertebrae_l3":            10.0,
    "vertebrae_l4":            10.0,
    "vertebrae_l5":            10.0,
    "vertebrae_t1":             8.0,
    "vertebrae_t2":             8.0,
    "vertebrae_t3":             8.0,
    "vertebrae_t4":             8.0,
    "vertebrae_t5":             8.0,
    "vertebrae_t6":             8.0,
    "vertebrae_t7":             8.0,
    "vertebrae_t8":             8.0,
    "vertebrae_t9":             8.0,
    "vertebrae_t10":            8.0,
    "vertebrae_t11":            8.0,
    "vertebrae_t12":            8.0,
    # Ribs — a complete rib segment is 10–25 cc
    "rib_left_1":               3.0,
    "rib_left_2":               3.0,
    "rib_left_3":               3.0,
    "rib_left_4":               3.0,
    "rib_left_5":               3.0,
    "rib_left_6":               3.0,
    "rib_left_7":               3.0,
    "rib_left_8":               3.0,
    "rib_left_9":               3.0,
    "rib_left_10":              3.0,
    "rib_left_11":              3.0,
    "rib_left_12":              3.0,
    "rib_right_1":              3.0,
    "rib_right_2":              3.0,
    "rib_right_3":              3.0,
    "rib_right_4":              3.0,
    "rib_right_5":              3.0,
    "rib_right_6":              3.0,
    "rib_right_7":              3.0,
    "rib_right_8":              3.0,
    "rib_right_9":              3.0,
    "rib_right_10":             3.0,
    "rib_right_11":             3.0,
    "rib_right_12":             3.0,
    # Shoulder / upper extremity — only visible on chest CTs
    "humerus_left":             5.0,
    "humerus_right":            5.0,
    "scapula_left":             8.0,
    "scapula_right":            8.0,
    "clavicula_left":           3.0,
    "clavicula_right":          3.0,
    # Sternum
    "sternum":                  5.0,
    "costal_cartilages":       10.0,
    # Neurological
    "spinal_cord":              3.0,   # normal ~7–15 cc for full thoracolumbar
    # Soft tissue
    "autochthon_left":          5.0,
    "autochthon_right":         5.0,
    "iliopsoas_left":           8.0,
    "iliopsoas_right":          8.0,
    # Pelvis (only present on abdomen-pelvis CTs)
    "urinary_bladder":          5.0,
    "prostate":                 2.0,
    "uterus":                   2.0,
    "sacrum":                  10.0,
    "hip_left":                15.0,
    "hip_right":               15.0,
    "femur_left":              10.0,
    "femur_right":             10.0,
}

_HU_TYPE_TABLE = [
    (-float("inf"), -900,  "air collection"),
    (-900,          -200,  "hypodense region"),
    (-200,           -20,  "low-density lesion"),
    (-20,             20,  "fluid collection"),
    (20,             100,  "soft-tissue lesion"),
    (100,            200,  "hyperdense lesion"),
    (200,   float("inf"), "calcification"),
]


def _infer_anomaly_type(candidate: "Candidate") -> str:
    """
    Classify anomaly type from TotalSegmentator HU measurements only.
    Phase 1 triage labels are NOT used — they are heuristic 2D labels
    and must never appear in the report.
    """
    if candidate.density_hu and candidate.density_hu.mean is not None:
        hu = candidate.density_hu.mean
        for lo, hi, label in _HU_TYPE_TABLE:
            if lo <= hu < hi:
                return label
    # No HU measurement available — report HU-neutral descriptor.
    # Do NOT fall back to phase1_anomaly_type (Phase 1 heuristic labels).
    return "density abnormality"


def to_structured_findings(
    candidates: List[Candidate],
    series_meta: Dict[str, Any],
    target_organs: List[str] = None,
    organ_stats: Optional[Dict[str, Any]] = None,
    anatomy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    organ_stats = organ_stats or {}
    anatomy = anatomy or {}
    anat_candidates = anatomy.get("candidates", {})
    anat_organs = anatomy.get("organs", {})
    # Directional wording (left/right, anterior/posterior, upper/lower) is only
    # emitted when orientation was verified from the anatomy itself.
    directions_trusted = bool(anatomy) and all(
        (anatomy.get("orientation", {}).get("verified") or {}).values() or [False]
    )

    # Step 1: Filter candidates by significance thresholds
    significant: List[Candidate] = []
    noise_count = 0

    for candidate in candidates:
        if candidate.suppressed:
            continue
        volume_cc = candidate.shape.volume_cc if candidate.shape else 0.0
        confidence = candidate.fused_confidence or 0.0
        if confidence >= MIN_CONFIDENCE and volume_cc >= MIN_VOLUME_CC:
            significant.append(candidate)
        else:
            noise_count += 1

    logger.info(
        "findings_adapter: %d candidates → %d significant, %d noise",
        len([c for c in candidates if not c.suppressed]),
        len(significant),
        noise_count,
    )

    # Step 2: Group significant candidates by organ, cap per organ
    organ_groups: Dict[str, List[Candidate]] = {}
    for candidate in significant:
        organ_groups.setdefault(candidate.organ_label, []).append(candidate)

    if target_organs:
        target_set = {o.lower() for o in target_organs}
        filtered = {k: v for k, v in organ_groups.items() if k.lower() in target_set}
        if filtered:
            organ_groups = filtered
        else:
            logger.warning("No candidates match target_organs=%s; using all significant", target_organs)

    for organ_name in organ_groups:
        organ_groups[organ_name].sort(key=lambda c: c.fused_confidence or 0.0, reverse=True)
        organ_groups[organ_name] = organ_groups[organ_name][:MAX_ANOMALIES_PER_ORGAN]

    # Step 3: Build organ findings — iterate over ALL segmented organs from Phase 2
    # organ_stats is the authoritative list of what TotalSegmentator actually saw.
    # Candidates are joined on top; organs not in organ_groups get status=normal.
    organs = []
    all_organ_names_raw = [
        n for n in (list(organ_stats.keys()) if organ_stats else list(organ_groups.keys()))
        if n.lower() != "unclassified"
    ]

    # Filter out organs that TotalSegmentator only partially captured.
    # When a scan doesn't cover an organ, TS still returns a tiny fragment.
    # We suppress any organ whose measured volume is below the per-class
    # minimum-coverage threshold in _MIN_COVERAGE_CC.
    def _sufficient_coverage(organ_name: str) -> bool:
        vol_cc = organ_stats.get(organ_name, {}).get("volume_cc")
        if vol_cc is None:
            return True  # no volume data → include (candidate-only organ)
        min_cc = _MIN_COVERAGE_CC.get(organ_name.lower())
        if min_cc is None:
            return True  # no threshold defined → always include
        sufficient = float(vol_cc) >= min_cc
        if not sufficient:
            logger.info(
                "findings_adapter: excluding %s (volume %.1f cc < min %.1f cc — "
                "organ not sufficiently covered by this scan)",
                organ_name, float(vol_cc), min_cc,
            )
        return sufficient

    all_organ_names = [n for n in all_organ_names_raw if _sufficient_coverage(n)]
    excluded_count = len(all_organ_names_raw) - len(all_organ_names)
    if excluded_count:
        logger.info(
            "findings_adapter: %d organ(s) excluded for insufficient scan coverage",
            excluded_count,
        )

    for organ_name in all_organ_names:
        baseline = organ_stats.get(organ_name, {})
        organ_candidates = organ_groups.get(organ_name, [])

        organ_finding: Dict[str, Any] = {
            "organ": organ_name,
            "status": "anomaly_detected" if organ_candidates else "normal",
        }

        # Whole-organ metrics from TotalSegmentator
        trimmed_mean = baseline.get("trimmed_mean")
        if trimmed_mean is not None:
            organ_finding["mean_density_hu"] = round(float(trimmed_mean), 1)

        vol_cc = baseline.get("volume_cc")
        if vol_cc and float(vol_cc) > 0:
            organ_finding["dimensions_mm"] = {"volume_cc": round(float(vol_cc), 2)}

        # Spatial context: surface contacts + partial-coverage flag
        a_org = anat_organs.get(organ_name, {})
        if a_org.get("contacts"):
            organ_finding["adjacent_structures"] = list(a_org["contacts"])
        if a_org.get("touches_fov_edge"):
            organ_finding["touches_fov_edge"] = True

        # Anomaly details for flagged organs
        if organ_candidates:
            anomalies = []
            for candidate in organ_candidates:
                shape = {}
                if candidate.shape:
                    if candidate.shape.volume_cc > 0:
                        shape["volume_cc"] = round(candidate.shape.volume_cc, 3)
                    if candidate.shape.long_axis_mm > 0:
                        shape["long_axis_mm"] = round(candidate.shape.long_axis_mm, 2)
                    if candidate.shape.short_axis_mm > 0:
                        shape["short_axis_mm"] = round(candidate.shape.short_axis_mm, 2)
                    if candidate.shape.sphericity > 0:
                        shape["sphericity"] = round(candidate.shape.sphericity, 4)
                    if candidate.shape.margin_curvature_variance > 0:
                        if candidate.shape.margin_curvature_variance > 50:
                            shape["margin"] = "spiculated or irregular"
                        elif candidate.shape.margin_curvature_variance > 20:
                            shape["margin"] = "lobulated"
                        else:
                            shape["margin"] = "well-defined"

                density_hu = {}
                if candidate.density_hu:
                    density_hu["mean"] = round(candidate.density_hu.mean, 1)
                    if candidate.density_hu.min != 0 or candidate.density_hu.max != 0:
                        density_hu["min"] = round(candidate.density_hu.min, 1)
                        density_hu["max"] = round(candidate.density_hu.max, 1)
                    if candidate.density_hu.std > 0:
                        density_hu["std"] = round(candidate.density_hu.std, 1)

                anomaly = {
                    "anomaly_id": candidate.candidate_id,
                    "type": _infer_anomaly_type(candidate),
                    "location": organ_name.replace("_", " ").title(),
                }
                sub_loc = (anat_candidates.get(candidate.candidate_id) or {}).get("sub_location")
                if sub_loc and directions_trusted:
                    anomaly["sub_location"] = sub_loc
                if shape:
                    anomaly["shape"] = shape
                if density_hu:
                    anomaly["density_hu"] = density_hu

                anomalies.append(anomaly)

            organ_finding["anomalies"] = anomalies

        organs.append(organ_finding)

    # Step 4: Also include any candidate organs not in organ_stats (edge case).
    # This is where Path-B lesions with no organ label land ("unclassified"):
    # give each one an anatomical description instead of a repeated bare label.
    already_emitted = {o["organ"] for o in organs}
    for organ_name, organ_candidates in organ_groups.items():
        if organ_name not in organ_stats and organ_name not in already_emitted:
            unassigned = organ_name.lower() == "unclassified"
            anomalies_out = []
            for c in organ_candidates:
                a_ctx = anat_candidates.get(c.candidate_id) or {}
                location = organ_name.replace("_", " ").title()
                if unassigned:
                    if directions_trusted and a_ctx.get("description"):
                        location = a_ctx["description"]
                    elif a_ctx.get("level") or a_ctx.get("nearby"):
                        bits = []
                        if a_ctx.get("level"):
                            bits.append("position " + a_ctx["level"])
                        if a_ctx.get("nearby"):
                            bits.append("closest segmented structures: " + ", ".join(
                                n.replace("_", " ") for n in a_ctx["nearby"]))
                        location = "; ".join(bits)
                entry = {
                    "anomaly_id": c.candidate_id,
                    "type": _infer_anomaly_type(c),
                    "location": location,
                }
                if c.shape:
                    sh = {}
                    if c.shape.volume_cc > 0:
                        sh["volume_cc"] = round(c.shape.volume_cc, 3)
                    if c.shape.long_axis_mm > 0:
                        sh["long_axis_mm"] = round(c.shape.long_axis_mm, 2)
                    if c.shape.short_axis_mm > 0:
                        sh["short_axis_mm"] = round(c.shape.short_axis_mm, 2)
                    if sh:
                        entry["shape"] = sh
                if c.density_hu:
                    d = {"mean": round(c.density_hu.mean, 1)}
                    if c.density_hu.min != 0 or c.density_hu.max != 0:
                        d["min"] = round(c.density_hu.min, 1)
                        d["max"] = round(c.density_hu.max, 1)
                    entry["density_hu"] = d
                anomalies_out.append(entry)
            finding = {
                "organ": organ_name,
                "status": "anomaly_detected",
                "anomalies": anomalies_out,
            }
            if unassigned:
                finding["unassigned"] = True
            organs.append(finding)

    if not organs:
        organs.append({"organ": "whole_body", "status": "not_visualised"})
        logger.info("findings_adapter: no organs detected — reporting as normal study.")

    structured_findings = {
        "study_id": series_meta.get("study_instance_uid") or series_meta.get("series_instance_uid", "unknown_study"),
        "modality": series_meta.get("modality", "CT"),
        "patient_id": series_meta.get("patient_id", "UNKNOWN_PATIENT"),
        "patient_name": series_meta.get("patient_name", "Unknown Patient"),
        "patient_age": series_meta.get("patient_age", ""),
        "patient_sex": series_meta.get("patient_sex", ""),
        "organs": organs,
    }

    protocol = series_meta.get("protocol")
    if protocol:
        structured_findings["protocol"] = protocol

    region = series_meta.get("phase2_region") or series_meta.get("region", "")
    if region:
        structured_findings["region"] = region

    # ── Phase 1 context for INDICATION / TECHNIQUE / COMPARISON ──
    for key in ("study_date", "referring_physician", "accession_number"):
        if series_meta.get(key):
            structured_findings[key] = series_meta[key]
    if series_meta.get("clinical_indication"):
        structured_findings["indication"] = series_meta["clinical_indication"]
    technique = {k: v for k, v in (series_meta.get("technique") or {}).items() if v not in ("", None, 0, 0.0)}
    if technique:
        structured_findings["technique"] = technique
    structured_findings["comparison"] = {"prior_studies": series_meta.get("prior_studies") or []}
    if series_meta.get("volume_qc"):
        structured_findings["volume_qc"] = series_meta["volume_qc"]
    if anatomy:
        structured_findings["orientation_verified"] = directions_trusted

    logger.info(
        "findings_adapter: built StructuredFindings with %d organ sections (%d anomalous)",
        len(organs),
        sum(1 for o in organs if o.get("status") == "anomaly_detected"),
    )

    return structured_findings
