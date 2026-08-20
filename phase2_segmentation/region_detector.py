"""
region_detector.py — Scan Body Region Classifier

Determines the broad body region covered by a CT or MR scan based on the set
of segmented organ/structure labels output by TotalSegmentator.

Includes majority-vote logic and a specialized clinical override for KUB
(Kidneys-Ureters-Bladder) scans.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

# Base organ mapping for exact matching. Wildcard patterns are checked programmatically.
REGION_ORGAN_MAP = {
    # Neuro / Head & Neck
    "brain": "neuro",
    "skull": "neuro",
    
    # Cardiothoracic
    "lung_upper_lobe_left": "cardiothoracic",
    "lung_lower_lobe_left": "cardiothoracic",
    "lung_upper_lobe_right": "cardiothoracic",
    "lung_middle_lobe_right": "cardiothoracic",
    "lung_lower_lobe_right": "cardiothoracic",
    "heart": "cardiothoracic",
    "aorta": "cardiothoracic",
    "pulmonary_artery": "cardiothoracic",
    "pulmonary_vein": "cardiothoracic",
    "trachea": "cardiothoracic",
    "esophagus": "cardiothoracic",
    "thyroid_gland": "cardiothoracic",
    "sternum": "cardiothoracic",
    "costal_cartilages": "cardiothoracic",
    "brachiocephalic_trunk": "cardiothoracic",
    "subclavian_artery_right": "cardiothoracic",
    "subclavian_artery_left": "cardiothoracic",
    "common_carotid_artery_right": "cardiothoracic",
    "common_carotid_artery_left": "cardiothoracic",
    "brachiocephalic_vein_left": "cardiothoracic",
    "brachiocephalic_vein_right": "cardiothoracic",
    "atrial_appendage_left": "cardiothoracic",
    "superior_vena_cava": "cardiothoracic",
    
    # Abdomen / Pelvis
    "liver": "abdomen_pelvis",
    "spleen": "abdomen_pelvis",
    "kidney_left": "abdomen_pelvis",
    "kidney_right": "abdomen_pelvis",
    "pancreas": "abdomen_pelvis",
    "stomach": "abdomen_pelvis",
    "gallbladder": "abdomen_pelvis",
    "urinary_bladder": "abdomen_pelvis",
    "prostate": "abdomen_pelvis",
    "adrenal_gland_right": "abdomen_pelvis",
    "adrenal_gland_left": "abdomen_pelvis",
    "small_bowel": "abdomen_pelvis",
    "duodenum": "abdomen_pelvis",
    "colon": "abdomen_pelvis",
    "kidney_cyst_left": "abdomen_pelvis",
    "kidney_cyst_right": "abdomen_pelvis",
    "gluteus_maximus_left": "abdomen_pelvis",
    "gluteus_maximus_right": "abdomen_pelvis",
    "gluteus_medius_left": "abdomen_pelvis",
    "gluteus_medius_right": "abdomen_pelvis",
    "gluteus_minimus_left": "abdomen_pelvis",
    "gluteus_minimus_right": "abdomen_pelvis",
    "iliopsoas_left": "abdomen_pelvis",
    "iliopsoas_right": "abdomen_pelvis",
    "autochthon_left": "abdomen_pelvis",
    "autochthon_right": "abdomen_pelvis",
    "portal_vein_and_splenic_vein": "abdomen_pelvis",
    "iliac_artery_left": "abdomen_pelvis",
    "iliac_artery_right": "abdomen_pelvis",
    "iliac_vena_left": "abdomen_pelvis",
    "iliac_vena_right": "abdomen_pelvis",
    "inferior_vena_cava": "abdomen_pelvis",
    
    # Ortho / Extremities
    "femur_left": "ortho",
    "femur_right": "ortho",
    "humerus_left": "ortho",
    "humerus_right": "ortho",
    "scapula_left": "ortho",
    "scapula_right": "ortho",
    "clavicula_left": "ortho",
    "clavicula_right": "ortho",
    "hip_left": "ortho",
    "hip_right": "ortho",
    
    # Spine / Spinal Cord
    "sacrum": "spine",
    "spinal_cord": "spine",
}

def _resolve_label_region(label: str) -> str:
    """Helper to map a TotalSegmentator label to its broad body region."""
    # Programmatic wildcard handling
    if label.startswith("vertebrae_"):
        return "spine"
    if label.startswith("rib_"):
        return "ortho"
        
    return REGION_ORGAN_MAP.get(label, "unknown")

def detect_region(present_labels: List[str]) -> str:
    """
    Classifies a scan's broad body region based on the present organ labels.

    Args:
        present_labels: List of label name strings.

    Returns:
        str: One of "neuro", "cardiothoracic", "abdomen_pelvis", "kub", "ortho", "spine", "unknown".
    """
    if not present_labels:
        return "unknown"

    # Normalized list of non-empty labels
    labels_set = {label.strip().lower() for label in present_labels}

    # ── KUB Clinical Override ──
    # Kidneys + Bladder present, but Liver + Spleen absent (focused renal scan)
    kidney_present = ("kidney_left" in labels_set or "kidney_right" in labels_set)
    bladder_present = ("urinary_bladder" in labels_set)
    liver_absent = ("liver" not in labels_set)
    spleen_absent = ("spleen" not in labels_set)

    if kidney_present and bladder_present and liver_absent and spleen_absent:
        logger.info(
            "KUB clinical criteria met: kidney(s)+bladder present, liver+spleen absent. Region resolved to: kub"
        )
        return "kub"

    # ── Structural Group Normalization ──
    # Prevent 24 individual rib labels or 24 individual vertebrae labels from
    # artificially inflating vote counts and distorting region classification.
    effective_labels = set()
    has_ribs = False
    has_vertebrae = False

    for label in labels_set:
        if label.startswith("rib_"):
            has_ribs = True
        elif label.startswith("vertebrae_"):
            has_vertebrae = True
        else:
            effective_labels.add(label)

    if has_ribs:
        effective_labels.add("rib_group")
    if has_vertebrae:
        effective_labels.add("vertebrae_group")

    # ── Majority-Vote Tally ──
    tallies = {
        "neuro": 0,
        "cardiothoracic": 0,
        "abdomen_pelvis": 0,
        "ortho": 0,
        "spine": 0
    }

    for label in effective_labels:
        if label == "rib_group":
            region = "cardiothoracic"  # Rib cage belongs to thoracic wall
        elif label == "vertebrae_group":
            region = "spine"
        else:
            region = _resolve_label_region(label)
            
        if region in tallies:
            tallies[region] += 1

    # Filter out regions with zero votes
    active_tallies = {r: count for r, count in tallies.items() if count > 0}
    if not active_tallies:
        return "unknown"

    # Check for mixed region scans (more than one active region category)
    if len(active_tallies) > 1:
        distribution_str = ", ".join(f"{r}: {c}" for r, c in sorted(active_tallies.items()))
        logger.info("Mixed body regions detected. Vote distribution: %s", distribution_str)

    # Resolve majority region (pick alphabetically in case of equal vote count tie)
    max_count = max(active_tallies.values())
    candidates = [r for r, count in active_tallies.items() if count == max_count]
    selected_region = sorted(candidates)[0]

    logger.info("Region majority vote selected: %s (votes: %d)", selected_region, max_count)
    return selected_region
