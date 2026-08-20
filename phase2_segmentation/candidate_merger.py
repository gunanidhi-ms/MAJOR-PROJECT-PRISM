"""
candidate_merger.py — Package 6: Technical Suppression Cascade, Merge & 3D Re-validation

Merges the two raw candidate lists produced by Package 4 (Path A, phase1_seed)
and Package 5 (Path B, organ_sweep) into a single unified, fully-attributed,
and clinically-filtered candidate list ready for Package 7 confidence fusion.

Lifecycle annotations from schemas/candidate.py:
  [MERGE]   corroborated, detected_by     — set here in _deduplicate_and_tag
  [FILTER]  organ_label, organ_local_zscore, organ_overlap_fraction,
            persistence_ok (read-only), suppressed, suppression_reason
                                          — set here in backfill + clinical_filter
  [SCORE]   fused_confidence, gate, emergency_score
                                          — NOT touched here; Package 7 only

All public and internal functions are standalone (no lifecycle_manager.py
dependency) — same synthetic-data-testable pattern as Package 4 and Package 5.

Usage:
    from phase2_segmentation.candidate_merger import merge_candidates, clinical_filter
"""

import math
import logging

import numpy as np
from scipy import ndimage

from schemas.candidate import Candidate, ShapeFeatures, DensityHU

logger = logging.getLogger(__name__)


# ── Clinical-filter rule constants ─────────────────────────────────────────────

LUNG_LABELS: set = {
    "lung_upper_lobe_left",
    "lung_upper_lobe_right",
    "lung_lower_lobe_left",
    "lung_lower_lobe_right",
    "lung_middle_lobe_right",
}

RENAL_LABELS: set = {
    "kidney_left",
    "kidney_right",
}

# Module-level default for the population_common z-score suppression threshold.
# Passed as a keyword argument to clinical_filter() so tests can override it
# without patching module state — same pattern as min_slices in finding_tracker.py
# and min_component_voxels in organ_sweep.py.
Z_SUPPRESS_THRESHOLD: float = 2.5


# ── Public entry point ─────────────────────────────────────────────────────────

def merge_candidates(
    path_a: list,
    path_b: list,
    seg_arr: np.ndarray,
    volume: np.ndarray,
    value_to_label: dict,
    organ_stats: dict,
    spacing: tuple,
    region: str,
    iou_threshold: float = 0.3,
) -> list:
    """
    Top-level Package 6 orchestrator.

    Executes four steps in order and returns the complete, fully-attributed
    candidate list — suppressed and unsuppressed alike.

    Steps:
      1. _deduplicate_and_tag   — merge Path A + Path B by 3D IoU; set corroborated.
      2. _backfill_organ_context — fill organ_label/zscore/overlap for "unclassified"
                                   candidates (primarily Path A survivors).
      3. _compute_shape_geometry — fill sphericity + margin_curvature_variance
                                   (deferred from P4/P5 by design).
      4. clinical_filter         — apply organ/region suppression rules; mark
                                   suppressed=True + suppression_reason on rejects.

    IMPORTANT — Package 6 never writes to fused_confidence, gate, or
    emergency_score. Those are [SCORE] lifecycle fields owned by Package 7.

    Args:
        path_a:         Candidates from seed_clusterer.cluster_phase1_seeds().
        path_b:         Candidates from organ_sweep.sweep_organs().
        seg_arr:        (Z, Y, X) integer label mask from TotalSegmentator.
        volume:         (Z, Y, X) HU volume — same array passed to Package 4/5.
        value_to_label: {int → str} e.g. {5: "liver"} from lifecycle_manager.
        organ_stats:    {str → baseline dict} from organ_baseline.py (Package 3).
        spacing:        (row_mm, col_mm, z_mm) — same tuple as Package 4/5.
        region:         Scan body region from region_detector.py (Package 3),
                        e.g. "cardiothoracic", "abdomen_pelvis".
        iou_threshold:  Minimum 3D bounding-box IoU to call two candidates
                        corroborated. Confirmed at 0.3 in design review.

    Returns:
        Unified list of Candidate objects with all [MERGE] and [FILTER] fields
        populated. [SCORE] fields remain at their schema defaults (0.0 / "" / 0).
        Every input candidate is represented in the output — none are silently
        dropped. Suppressed candidates carry suppressed=True + suppression_reason.
    """
    candidates = _deduplicate_and_tag(path_a, path_b, iou_threshold)
    _backfill_organ_context(candidates, seg_arr, value_to_label, organ_stats, spacing)
    _compute_shape_geometry(candidates, volume, spacing)
    clinical_filter(candidates, region)

    n_corroborated = sum(1 for c in candidates if c.corroborated)
    n_suppressed = sum(1 for c in candidates if c.suppressed)
    logger.info(
        "merge_candidates: %d path_a + %d path_b → %d unified "
        "(%d corroborated, %d suppressed)",
        len(path_a), len(path_b), len(candidates),
        n_corroborated, n_suppressed,
    )
    return candidates


# ── Step 1: deduplication and corroboration tagging ───────────────────────────

def _deduplicate_and_tag(
    path_a: list,
    path_b: list,
    iou_threshold: float,
) -> list:
    """
    Merge Path A and Path B into one deduplicated list.

    Algorithm:
      - Compute 3D IoU for every (Path A, Path B) pair.
      - Greedily match highest-IoU pairs first (sort descending, skip
        already-matched candidates).
      - For each matched pair with IoU >= iou_threshold, build one survivor:
          Spatial/density fields  → Path B wins (mask-derived, more accurate):
              centroid_3d, bbox_3d, slice_indices, shape, density_hu,
              organ_label, organ_local_zscore, organ_overlap_fraction,
              persistence_ok.
          Provenance fields       → Path A always preserved:
              candidate_id, phase1_confidence, phase1_severity,
              phase1_anomaly_type.
          Merge fields            → corroborated=True,
                                    detected_by=["phase1_seed","organ_sweep"].
          Score fields            → left at schema defaults (Package 7 only).
      - Unmatched Path A candidates → appended as-is (corroborated=False).
      - Unmatched Path B candidates → appended as-is (corroborated=False).

    Return order: merged survivors, then unmatched Path A, then unmatched Path B.
    """
    if not path_a:
        return list(path_b)
    if not path_b:
        return list(path_a)

    # Build all (iou, a_idx, b_idx) pairs where IoU > 0
    pairs: list = []
    for a_idx, a in enumerate(path_a):
        for b_idx, b in enumerate(path_b):
            iou = _iou_3d(a.bbox_3d, b.bbox_3d)
            if iou > 0.0:
                pairs.append((iou, a_idx, b_idx))

    # Greedy assignment: highest IoU first
    pairs.sort(key=lambda x: x[0], reverse=True)

    used_a: set = set()
    used_b: set = set()
    merged: list = []

    for iou, a_idx, b_idx in pairs:
        if a_idx in used_a or b_idx in used_b:
            continue
        if iou < iou_threshold:
            continue

        a = path_a[a_idx]
        b = path_b[b_idx]

        # Build survivor: Path B spatial fields win, Path A provenance preserved.
        # [SCORE] fields are explicitly left at their schema defaults here to make
        # the lifecycle boundary unambiguous — Package 7 is the only writer.
        survivor = Candidate(
            # Identity: keep Path A's upstream-assigned ID
            candidate_id=a.candidate_id,
            corroborated=True,
            detected_by=["phase1_seed", "organ_sweep"],

            # Spatial — Path B wins (mask-derived from actual connected component)
            organ_label=b.organ_label,
            bbox_3d=list(b.bbox_3d),
            centroid_3d=list(b.centroid_3d),
            slice_indices=list(b.slice_indices),

            # Shape — Path B wins; sphericity/margin_curvature_variance will be
            # filled by _compute_shape_geometry in the next step
            shape=ShapeFeatures(
                volume_cc=b.shape.volume_cc,
                long_axis_mm=b.shape.long_axis_mm,
                short_axis_mm=b.shape.short_axis_mm,
                sphericity=b.shape.sphericity,
                elongation=b.shape.elongation,
                margin_curvature_variance=b.shape.margin_curvature_variance,
            ),

            # Density — Path B wins
            density_hu=DensityHU(
                mean=b.density_hu.mean,
                min=b.density_hu.min,
                max=b.density_hu.max,
                std=b.density_hu.std,
            ),

            # Organ context — Path B wins
            organ_local_zscore=b.organ_local_zscore,
            organ_overlap_fraction=b.organ_overlap_fraction,

            # Persistence — Path B wins (computed from actual mask Z-span)
            persistence_ok=b.persistence_ok,

            # Phase 1 provenance — always Path A (unique to Path A candidates)
            phase1_confidence=a.phase1_confidence,
            phase1_severity=a.phase1_severity,
            phase1_anomaly_type=a.phase1_anomaly_type,

            # [FILTER] fields start at defaults; filled later in this pipeline
            suppressed=False,
            suppression_reason="",

            # [SCORE] fields — Package 7 ONLY, intentionally left at defaults
            fused_confidence=0.0,
            gate="",
            emergency_score=0,

            # texture_features: radiomics not running (see README §976 note)
            texture_features={},
        )
        merged.append(survivor)
        used_a.add(a_idx)
        used_b.add(b_idx)

    # Unmatched Path A candidates — corroborated=False already from constructor
    for a_idx, a in enumerate(path_a):
        if a_idx not in used_a:
            merged.append(a)

    # Unmatched Path B candidates — corroborated=False already from constructor
    for b_idx, b in enumerate(path_b):
        if b_idx not in used_b:
            merged.append(b)

    return merged


def _iou_3d(a: list, b: list) -> float:
    """
    3D bounding-box Intersection over Union.

    Both bboxes use the PRISM convention:
        [x_min, y_min, z_min, x_max, y_max, z_max]
    with **inclusive** integer voxel bounds — both endpoints are inside the box.

    Volume of an inclusive-bounds box:
        (x_max - x_min + 1) * (y_max - y_min + 1) * (z_max - z_min + 1)

    The +1 on every dimension is mandatory (H3 fix from audit). Without it, a
    single-voxel-wide box returns dimension=0, causing IoU=0 for perfectly
    coincident candidates and breaking corroboration for the most common case
    where a Path A seed and a Path B component occupy the same voxel extent.

    Args:
        a: bbox_3d of candidate A — [x_min, y_min, z_min, x_max, y_max, z_max].
        b: bbox_3d of candidate B — same format.

    Returns:
        IoU in [0.0, 1.0]. Returns 0.0 if the boxes do not overlap.
    """
    inter_x = max(0, min(a[3], b[3]) - max(a[0], b[0]) + 1)
    inter_y = max(0, min(a[4], b[4]) - max(a[1], b[1]) + 1)
    inter_z = max(0, min(a[5], b[5]) - max(a[2], b[2]) + 1)
    inter_vol = inter_x * inter_y * inter_z

    if inter_vol == 0:
        return 0.0

    vol_a = (a[3] - a[0] + 1) * (a[4] - a[1] + 1) * (a[5] - a[2] + 1)
    vol_b = (b[3] - b[0] + 1) * (b[4] - b[1] + 1) * (b[5] - b[2] + 1)
    union_vol = vol_a + vol_b - inter_vol
    return inter_vol / union_vol if union_vol > 0 else 0.0


# ── Step 2: backfill organ context for Path A candidates ──────────────────────

def _backfill_organ_context(
    candidates: list,
    seg_arr: np.ndarray,
    value_to_label: dict,
    organ_stats: dict,
    spacing: tuple,  # (row_mm, col_mm, z_mm) — accepted for API consistency
) -> None:
    """
    In-place: assign organ_label, organ_local_zscore, and organ_overlap_fraction
    to every candidate still carrying organ_label == "unclassified".

    This corrects the Path A asymmetry: seed_clusterer.py has no organ membership
    information at track-clustering time (see README §13 Package 4 note) and sets
    organ_label="unclassified" unconditionally. Path B (organ_sweep) always sets
    organ_label at creation, so merged corroborated survivors already carry the
    correct Path B context and are skipped by this function.

    Coordinate note: seg_arr is (Z, Y, X). bbox_3d is [x_min, y_min, z_min,
    x_max, y_max, z_max] — X is first in the list but Z must be first when
    indexing into the numpy array.

    Args:
        candidates:     Unified candidate list (mutated in-place).
        seg_arr:        (Z, Y, X) TotalSegmentator label mask.
        value_to_label: {int label value → str organ name}.
        organ_stats:    {str organ name → baseline dict from organ_baseline.py}.
        spacing:        (row_mm, col_mm, z_mm) — accepted for API consistency.
    """
    for c in candidates:
        if c.organ_label != "unclassified":
            continue

        x_min, y_min, z_min, x_max, y_max, z_max = c.bbox_3d

        # Clamp bbox to seg_arr bounds (guards against out-of-range Path A seeds)
        z_lo = max(0, z_min)
        z_hi = min(seg_arr.shape[0] - 1, z_max)
        y_lo = max(0, y_min)
        y_hi = min(seg_arr.shape[1] - 1, y_max)
        x_lo = max(0, x_min)
        x_hi = min(seg_arr.shape[2] - 1, x_max)

        # Empty bbox after clamping (candidate entirely outside seg_arr extent)
        if z_lo > z_hi or y_lo > y_hi or x_lo > x_hi:
            logger.debug(
                "backfill: candidate %s bbox [%s] outside seg_arr, skipping",
                c.candidate_id, c.bbox_3d,
            )
            continue

        # seg_arr is (Z, Y, X) — index accordingly
        region_labels = seg_arr[z_lo:z_hi + 1, y_lo:y_hi + 1, x_lo:x_hi + 1]
        non_zero = region_labels[region_labels != 0]

        if non_zero.size == 0:
            # No organ mask covers this bbox — leave as "unclassified".
            # clinical_filter will NOT suppress these; Package 7 routes them
            # to gate="MANUAL_REVIEW".
            logger.debug(
                "backfill: candidate %s has no organ coverage in seg_arr bbox",
                c.candidate_id,
            )
            continue

        # Majority-vote: most frequent non-zero label in the bbox region.
        # np.bincount requires non-negative integers; TotalSegmentator labels >= 1
        # after filtering out background (0), so argmax() always returns >= 1.
        counts = np.bincount(non_zero.flatten())
        majority_val = int(counts.argmax())
        label = value_to_label.get(majority_val, "unclassified")
        c.organ_label = label

        # organ_overlap_fraction: fraction of clamped bbox voxels belonging to
        # the majority organ. Same formula as organ_sweep.py line 183.
        bbox_volume = (z_hi - z_lo + 1) * (y_hi - y_lo + 1) * (x_hi - x_lo + 1)
        voxel_count = int(np.sum(region_labels == majority_val))
        c.organ_overlap_fraction = (
            round(voxel_count / bbox_volume, 4) if bbox_volume > 0 else 0.0
        )

        # organ_local_zscore using the same formula pattern as organ_sweep.py
        # lines 177-179. NOTE: this is a scalar point-estimate approximation —
        # it uses density_hu.mean (the bounding-box HU average) rather than the
        # per-voxel mean of the actual connected-component mask. Path B candidates
        # use the per-voxel calculation. These will agree for uniform lesions and
        # diverge for heterogeneous ones. The approximation is unavoidable here
        # because Path A candidates carry no component mask, only bbox + scalars.
        if label != "unclassified":
            baseline = organ_stats.get(label)
            if baseline and baseline.get("trimmed_std", 0.0) > 1e-6:
                z_score = (
                    abs(c.density_hu.mean - baseline["trimmed_mean"])
                    / baseline["trimmed_std"]
                )
                c.organ_local_zscore = round(float(z_score), 4)
            # else: no reliable baseline for this organ — leave at 0.0


# ── Step 3: shape geometry computation ────────────────────────────────────────

def _compute_shape_geometry(
    candidates: list,
    volume: np.ndarray,
    spacing: tuple,
) -> None:
    """
    In-place: compute sphericity and margin_curvature_variance for every
    candidate. Both are 0.0 from Package 4 and Package 5 (intentionally
    deferred to this stage — see README §13 Package 4 design note, M1 decision).

    APPROXIMATION CONTRACT (M1 audit decision):
        Candidate stores no binary voxel mask — only bbox_3d and scalar density
        statistics. sphericity and margin_curvature_variance are therefore
        computed from an HU-threshold-derived mask approximation within the bbox
        region of the HU volume, NOT from a true segmentation mask. This is a
        known, intentional approximation documented at design time. Downstream
        code must not treat these values as equivalent to mask-derived
        morphometrics.

    VOLUME_CC SEMANTICS NOTE (M2 audit decision):
        volume_cc differs between paths and is NOT corrected here:
          - Path A (seed_clusterer): whole bounding-box voxel count × voxel_volume_cc
          - Path B (organ_sweep):   connected-component voxel count × voxel_volume_cc
        The approximation mask constructed below is independent and does not
        retroactively fix volume_cc. Clinical size thresholds must account for
        this distinction explicitly.

    Sphericity formula (Wadell, 1935):
        sphericity = π^(1/3) * (6V)^(2/3) / A
    where V = volume of the approximate mask in mm³ and A = surface area in mm².

    Surface area is approximated using the mean of all three voxel face
    orientations: face_area_mm2 = (row_mm*col_mm + row_mm*z_mm + col_mm*z_mm) / 3
    This correctly weights Z-faces for anisotropic CT spacing (e.g. thick-slice
    acquisitions where z_mm >> col_mm), unlike an axial-only approximation which
    would underestimate surface area and overestimate sphericity in such cases.

    margin_curvature_variance: standard deviation of gradient magnitude on
    surface voxels — higher values indicate a more irregular, spiculated margin.

    volume is (Z, Y, X). bbox_3d is [x_min, y_min, z_min, x_max, y_max, z_max].

    Args:
        candidates: Unified candidate list (shape fields mutated in-place).
        volume:     (Z, Y, X) HU volume array.
        spacing:    (row_mm, col_mm, z_mm) physical voxel spacing.
    """
    row_mm, col_mm, z_mm = spacing

    # Mean of the three distinct voxel face areas — accounts for all face
    # orientations equally, critical for anisotropic spacing (correction 3).
    face_area_mm2 = (row_mm * col_mm + row_mm * z_mm + col_mm * z_mm) / 3.0

    for c in candidates:
        x_min, y_min, z_min, x_max, y_max, z_max = c.bbox_3d

        # Clamp to volume bounds
        z_lo = max(0, z_min)
        z_hi = min(volume.shape[0] - 1, z_max)
        y_lo = max(0, y_min)
        y_hi = min(volume.shape[1] - 1, y_max)
        x_lo = max(0, x_min)
        x_hi = min(volume.shape[2] - 1, x_max)

        if z_lo > z_hi or y_lo > y_hi or x_lo > x_hi:
            logger.debug(
                "shape_geometry: candidate %s bbox outside volume, skipping",
                c.candidate_id,
            )
            continue

        # volume is (Z, Y, X) — index accordingly
        bbox_region = volume[z_lo:z_hi + 1, y_lo:y_hi + 1, x_lo:x_hi + 1]

        if bbox_region.size == 0:
            continue

        # Build HU-threshold approximation mask using the candidate's own density.
        # half_window = max(2σ, 30 HU) — floor of 30 HU prevents an all-False mask
        # for nearly-uniform regions (e.g. homogeneous cysts where σ ≈ 0).
        mu = c.density_hu.mean
        sigma = c.density_hu.std
        half_window = max(2.0 * sigma, 30.0)
        approx_mask = np.abs(bbox_region - mu) <= half_window

        n_filled = int(np.sum(approx_mask))
        if n_filled == 0:
            continue

        V_mm3 = n_filled * row_mm * col_mm * z_mm

        # Surface voxels: mask minus its 1-voxel erosion
        eroded = ndimage.binary_erosion(approx_mask)
        surface_mask = approx_mask & ~eroded
        n_surface = int(np.sum(surface_mask))

        if n_surface == 0:
            continue

        # Surface area: n_surface_voxels × mean face area across all orientations
        A_mm2 = n_surface * face_area_mm2

        # Wadell sphericity — clamped to [0, 1] to guard against approximation
        # overshoot in degenerate geometries
        if A_mm2 > 0.0 and V_mm3 > 0.0:
            sphericity = (
                math.pi ** (1.0 / 3.0) * (6.0 * V_mm3) ** (2.0 / 3.0)
            ) / A_mm2
            c.shape.sphericity = round(min(float(sphericity), 1.0), 4)

        # Margin curvature variance: std of Sobel gradient magnitude on surface
        if np.any(surface_mask):
            gx = ndimage.sobel(approx_mask.astype(float), axis=2)
            gy = ndimage.sobel(approx_mask.astype(float), axis=1)
            gz = ndimage.sobel(approx_mask.astype(float), axis=0)
            grad_mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
            surface_grad = grad_mag[surface_mask]
            c.shape.margin_curvature_variance = (
                round(float(np.std(surface_grad)), 4)
                if surface_grad.size > 1
                else 0.0
            )


# ── Step 4 (public): clinical suppression filter ──────────────────────────────

def clinical_filter(
    candidates: list,
    region: str,
    z_suppress_threshold: float = Z_SUPPRESS_THRESHOLD,
) -> list:
    """
    Apply organ- and region-aware suppression rules to each candidate.

    Sets suppressed=True and suppression_reason=<rule_key> on rejected
    candidates. Returns ALL candidates — suppressed and unsuppressed alike.
    Never silently drops any candidate (master spec requirement: every rejected
    candidate must carry a logged suppression reason for audit traceability).

    Unclassified candidates (organ_label == "unclassified") are never
    auto-suppressed regardless of other criteria. Package 7 routes them to
    gate="MANUAL_REVIEW" based on the unresolved organ attribution.

    Rule evaluation order — first matching rule wins:

      "not_persistent"     persistence_ok == False AND NOT corroborated
                           Single/double-slice blips not confirmed by both paths.

      "lung_nodule_<6mm"   organ_label in LUNG_LABELS
                           AND shape.long_axis_mm < 6.0
                           AND region == "cardiothoracic"
                           Sub-centimetre pulmonary nodules below the Fleischner
                           reporting threshold. Applies to corroborated candidates.

      "simple_renal_cyst"  organ_label in RENAL_LABELS
                           AND density_hu.mean < 20.0 HU  (near-water density)
                           AND density_hu.std  < 15.0 HU  (homogeneous)
                           AND shape.sphericity > 0.7    (computed in step 3)
                           Bosniak I cyst-equivalent. Applies to corroborated.

      "population_common"  organ_local_zscore < z_suppress_threshold
                           AND NOT corroborated
                           Statistically unremarkable, unseen by Path B.

    Corroboration protection:
        corroborated=True candidates are exempt from "not_persistent" and
        "population_common". Organ-specific size/density rules still apply —
        a 4 mm lung nodule is still 4 mm regardless of corroboration.

    Args:
        candidates:           Unified candidate list (mutated in-place).
        region:               Body region string from region_detector.py.
        z_suppress_threshold: organ_local_zscore below this → "population_common".
                              Keyword argument with module-level default (2.5).

    Returns:
        The same list (identity return — all candidates, suppressed or not).
    """
    for c in candidates:
        # Never overwrite a suppression already set (defensive; shouldn't happen)
        if c.suppressed:
            continue

        # Unclassified candidates are never auto-suppressed. Package 7 routes
        # them to MANUAL_REVIEW. This applies regardless of corroboration status.
        if c.organ_label == "unclassified":
            continue

        # Rule 1: not persistent — single/double-slice blip, uncorroborated
        if not c.persistence_ok and not c.corroborated:
            c.suppressed = True
            c.suppression_reason = "not_persistent"
            continue

        # Rule 2: lung nodule below Fleischner reporting threshold
        if (
            c.organ_label in LUNG_LABELS
            and region == "cardiothoracic"
            and c.shape.long_axis_mm < 6.0
        ):
            c.suppressed = True
            c.suppression_reason = "lung_nodule_<6mm"
            continue

        # Rule 3: simple renal cyst (Bosniak I equivalent)
        if (
            c.organ_label in RENAL_LABELS
            and c.density_hu.mean < 20.0
            and c.density_hu.std < 15.0
            and c.shape.sphericity > 0.7
        ):
            c.suppressed = True
            c.suppression_reason = "simple_renal_cyst"
            continue

        # Rule 4: population-common — statistically unremarkable, uncorroborated
        if c.organ_local_zscore < z_suppress_threshold and not c.corroborated:
            c.suppressed = True
            c.suppression_reason = "population_common"
            continue

    return candidates
