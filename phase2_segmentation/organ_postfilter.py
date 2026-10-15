"""
organ_postfilter.py — Phase 1 Finding Post-Filter Using Phase 2 Organ Masks

ZERO HALLUCINATION POLICY
──────────────────────────
This module NEVER asserts a diagnosis or disease name.
Every output label is strictly derived from:
  1. The organ name reported by TotalSegmentator (nnU-Net) — a measured fact.
  2. The measured mean HU value of the candidate region — from the CT data.
  3. The patient's own organ baseline HU statistics computed in Phase 2 — not
     from any population table; purely from THIS patient's segmented voxels.

Label format (the only label this module produces):
  "Attenuation anomaly in {organ} — {hu:.0f} HU
   (organ baseline: {baseline_mean:.0f} ± {baseline_std:.0f} HU,
   deviation: {z:.1f}σ) — requires radiologist review"

No disease name is ever emitted. No population HU ranges are used as
diagnostic cutoffs. The label is a density descriptor, not a diagnosis.

What this module fixes
───────────────────────
Problem 1 — "Marks outside the organ":
  Phase 1's body mask is dilated 15px into subcutaneous fat. Statistical
  outliers in that fat band produce bboxes that land outside any real organ.
  This module suppresses those findings by checking the bbox against the
  TotalSegmentator organ mask: if the bbox does not overlap a real organ by
  at least ORGAN_OVERLAP_FRACTION_MIN, it is suppressed.

Problem 2 — "Normal bowel gas flagged as critical":
  Phase 1 compares each finding to the WHOLE-SLICE HU mean. Small bowel gas
  at -800 HU is a statistical outlier vs soft tissue — but it is physically
  inside the bowel lumen, which is expected anatomy.
  This module checks the finding against the organ's patient-specific baseline.
  If the deviation is < INTRA_ORGAN_ZSCORE_MIN sigma, it is suppressed as
  within normal variation for that organ in this patient.

Lifecycle
─────────
This runs after Stage 5 in lifecycle_manager.py, once TotalSegmentator has
completed and seg_arr + organ_stats are in memory.
It does NOT run during Phase 1 real-time streaming (that would block).
The validated finding list is broadcast via WebSocket as a retroactive update.
"""

import logging
from typing import Dict, List, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────────

# Minimum z-score against the organ's own patient baseline for a finding to be
# considered a true intra-organ deviation.
# Below this the HU is within normal variation for this organ in this patient.
INTRA_ORGAN_ZSCORE_MIN = 2.5

# Fraction of bbox pixels that must fall inside the same organ for the finding
# to be considered "inside" that organ.  A corner of the bbox barely touching
# an organ boundary is not sufficient evidence.
ORGAN_OVERLAP_FRACTION_MIN = 0.25


# ── Main Entry Point ───────────────────────────────────────────────────────────

def postfilter_phase1_findings(
    findings_per_instance: Dict[int, List[Any]],
    instance_to_z: Dict[int, int],
    seg_arr: np.ndarray,
    cmap: Dict[int, str],
    organ_stats: Dict[str, Dict],
) -> Dict[int, List[Dict]]:
    """
    Retroactively validate and relabel Phase 1 findings using the Phase 2
    TotalSegmentator (nnU-Net) organ segmentation mask.

    Args:
        findings_per_instance:
            {instance_number: [Finding objects]} from VolumeAccumulator.
            Each Finding must have .bbox ([x, y, w, h]) and .mean_hu.

        instance_to_z:
            {instance_number: z_index_in_seg_arr}.
            Built via build_instance_to_z() from the spatially ordered
            instance list returned by VolumeAccumulator.export_volume().

        seg_arr:
            3D integer label array (Z, Y, X) from TotalSegmentator. 0=background.

        cmap:
            {int label_value: str organ_name} from TotalSegmentator class_map.

        organ_stats:
            {organ_name: {trimmed_mean, trimmed_std, mad, ...}} — patient-
            specific baselines from organ_baseline.py (Stage 3 of lifecycle).

    Returns:
        {instance_number: [enriched_finding_dict]} where each dict includes:
            organ_label              : str  — TotalSegmentator organ name or ""
            organ_validated          : bool — True if a genuine intra-organ deviation
            suppressed_outside_organ : bool — True if bbox outside all organ masks
            intra_organ_zscore       : float
            validated_label          : str  — zero-hallucination display string
    """
    if seg_arr is None or seg_arr.ndim != 3:
        logger.warning(
            "organ_postfilter: seg_arr unavailable — skipping postfilter"
        )
        return _passthrough(findings_per_instance)

    z_depth = seg_arr.shape[0]
    result: Dict[int, List[Dict]] = {}

    total = validated = outside = intra_normal = 0

    for instance_num, findings in findings_per_instance.items():
        if not findings:
            result[instance_num] = []
            continue

        z_idx = instance_to_z.get(instance_num)
        if z_idx is None or z_idx < 0 or z_idx >= z_depth:
            logger.debug(
                "organ_postfilter: instance %d has no z mapping — passthrough",
                instance_num,
            )
            result[instance_num] = [_passthrough_finding(f) for f in findings]
            continue

        seg_slice = seg_arr[z_idx]          # (Y, X)
        slice_h, slice_w = seg_slice.shape

        enriched = []
        for finding in findings:
            total += 1
            fd = finding.to_dict() if hasattr(finding, "to_dict") else dict(finding)
            bbox = fd.get("bbox", [0, 0, 0, 0])
            mean_hu = float(fd.get("mean_hu", fd.get("hu_mean", 0.0)))

            # ── Step A: Spatial containment check ─────────────────────────────
            organ_label, overlap_frac = _find_organ_at_bbox(
                bbox, seg_slice, cmap, slice_h, slice_w
            )

            if not organ_label or overlap_frac < ORGAN_OVERLAP_FRACTION_MIN:
                # Finding is in fat / air / skin / artifact — suppress
                fd["organ_label"] = ""
                fd["organ_validated"] = False
                fd["suppressed_outside_organ"] = True
                fd["intra_organ_zscore"] = 0.0
                fd["validated_label"] = (
                    "Outside organ boundary — suppressed by anatomy validator"
                )
                outside += 1
                enriched.append(fd)
                continue

            # ── Step B: Intra-organ deviation check ───────────────────────────
            baseline = organ_stats.get(organ_label, {})
            intra_z, bm, bs = _compute_intra_organ_zscore(mean_hu, baseline)

            if intra_z < INTRA_ORGAN_ZSCORE_MIN and bm != 0.0:
                # HU is within normal variation for THIS organ in THIS patient.
                # E.g. bowel gas at -700 HU is expected inside the bowel lumen;
                # it should not be flagged as an anomaly.
                fd["organ_label"] = organ_label
                fd["organ_validated"] = False
                fd["suppressed_outside_organ"] = False
                fd["intra_organ_zscore"] = round(intra_z, 2)
                fd["validated_label"] = (
                    f"Within normal attenuation range for {_pretty(organ_label)} "
                    f"({mean_hu:.0f} HU, organ baseline: {bm:.0f} \u00b1 {bs:.0f} HU) "
                    f"\u2014 not flagged"
                )
                intra_normal += 1
                enriched.append(fd)
                continue

            # ── Step C: Genuine intra-organ deviation ─────────────────────────
            # Relabel with zero-hallucination format.
            fd["organ_label"] = organ_label
            fd["organ_validated"] = True
            fd["suppressed_outside_organ"] = False
            fd["intra_organ_zscore"] = round(intra_z, 2)
            fd["validated_label"] = _zero_hallucination_label(
                organ_label, mean_hu, bm, bs, intra_z
            )
            validated += 1
            enriched.append(fd)

        result[instance_num] = enriched

    logger.info(
        "organ_postfilter: %d total \u2192 %d organ-validated, "
        "%d suppressed (outside organ), %d suppressed (normal organ range)",
        total, validated, outside, intra_normal,
    )
    return result


# ── Helpers ────────────────────────────────────────────────────────────────────

def _find_organ_at_bbox(
    bbox: List[int],
    seg_slice: np.ndarray,
    cmap: Dict[int, str],
    slice_h: int,
    slice_w: int,
) -> Tuple[str, float]:
    """
    Return the organ label with greatest pixel overlap within the bbox region
    of the 2D segmentation slice.

    Returns (organ_name, overlap_fraction). organ_name is "" if none found.
    """
    x, y, w, h = bbox
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(slice_w, int(x + w))
    y2 = min(slice_h, int(y + h))

    if x2 <= x1 or y2 <= y1:
        return "", 0.0

    region = seg_slice[y1:y2, x1:x2]
    total_px = region.size
    if total_px == 0:
        return "", 0.0

    unique_vals, counts = np.unique(region, return_counts=True)
    best_label = ""
    best_overlap = 0.0

    for val, count in zip(unique_vals, counts):
        if int(val) == 0:
            continue  # background
        organ_name = cmap.get(int(val), "")
        if not organ_name:
            continue
        overlap = count / total_px
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = organ_name

    return best_label, float(best_overlap)


def _compute_intra_organ_zscore(
    mean_hu: float,
    baseline: Dict,
) -> Tuple[float, float, float]:
    """
    Compute |mean_hu - organ_baseline_mean| / organ_baseline_std.

    Uses the patient-specific trimmed statistics from organ_baseline.py —
    NOT any population table.

    Returns (z_score, baseline_mean, baseline_std).
    """
    bm = float(baseline.get("trimmed_mean", 0.0))
    bs = float(baseline.get("trimmed_std", 0.0))

    if bs < 1.0:
        # Narrow std — use MAD-derived sigma as fallback (more robust)
        mad = float(baseline.get("mad", 0.0))
        bs = max(mad / 0.6745, 1.0)

    z = abs(mean_hu - bm) / bs
    return round(z, 3), round(bm, 1), round(bs, 1)


def _zero_hallucination_label(
    organ_label: str,
    mean_hu: float,
    baseline_mean: float,
    baseline_std: float,
    intra_z: float,
) -> str:
    """
    Produce a display label that contains ONLY measured facts.

    "hyperdense" / "hypodense" are density descriptors derived purely from
    the sign of (mean_hu - baseline_mean). They are NOT diagnostic terms.
    The label ends with "requires radiologist review" to make the human-in-
    the-loop requirement explicit.
    """
    direction = "hyperdense" if mean_hu > baseline_mean else "hypodense"
    organ_pretty = _pretty(organ_label)
    return (
        f"Attenuation anomaly in {organ_pretty} \u2014 "
        f"{mean_hu:.0f} HU ({direction}, organ baseline: "
        f"{baseline_mean:.0f} \u00b1 {baseline_std:.0f} HU, "
        f"deviation: {intra_z:.1f}\u03c3) \u2014 requires radiologist review"
    )


def _pretty(organ_label: str) -> str:
    """Convert TotalSegmentator snake_case label to Title Case display string."""
    return organ_label.replace("_", " ").title()


def _passthrough_finding(finding: Any) -> Dict:
    """Passthrough path when seg_arr is unavailable."""
    fd = finding.to_dict() if hasattr(finding, "to_dict") else dict(finding)
    fd.setdefault("organ_label", "")
    fd.setdefault("organ_validated", False)
    fd.setdefault("suppressed_outside_organ", False)
    fd.setdefault("intra_organ_zscore", 0.0)
    fd.setdefault("validated_label", "")
    return fd


def _passthrough(findings_per_instance: Dict[int, List[Any]]) -> Dict[int, List[Dict]]:
    """Return all findings unchanged when seg_arr is unavailable (graceful degradation)."""
    return {
        inst: [_passthrough_finding(f) for f in findings]
        for inst, findings in findings_per_instance.items()
    }


# ── Instance-to-Z Mapping Builder ─────────────────────────────────────────────

def build_instance_to_z(ordered_instance_numbers: List[int]) -> Dict[int, int]:
    """
    Build {instance_number: z_index} from the spatially ordered instance list
    returned by VolumeAccumulator.export_volume().

    The ordered list is sorted by ImagePositionPatient Z-coordinate, so
    index 0 = bottommost Z slice, index N-1 = topmost Z slice.
    This matches seg_arr's Z axis (TotalSegmentator preserves NIfTI orientation).
    """
    return {inst: z for z, inst in enumerate(ordered_instance_numbers)}
