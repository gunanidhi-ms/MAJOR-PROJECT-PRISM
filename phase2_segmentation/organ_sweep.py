"""
organ_sweep.py — Package 5: Path B, Independent Organ-Wide Statistical Sweep

Independently scans every segmented organ for statistically abnormal voxels
using that organ's own patient-adaptive baseline (organ_baseline.py) — with
zero dependency on what Phase 1 flagged. This is the slower, exhaustive half
of Phase 2's two-path strategy; Path A (seed_clusterer.py) reuses Phase 1's
real-time seeds, Path B re-derives evidence from scratch per organ.

Because Path B candidates are found entirely within one organ's mask, this
module is the natural place to populate organ_label, organ_local_zscore, and
organ_overlap_fraction at creation — Path A has no organ membership info at
the point a track is clustered.
"""

import logging
import numpy as np
from scipy import ndimage

from schemas.candidate import Candidate, ShapeFeatures, DensityHU

logger = logging.getLogger(__name__)

Z_SCORE_THRESHOLD = 3.0
MAD_SCORE_THRESHOLD = 3.5
IQR_FENCE_MULTIPLIER = 1.5
MAD_TO_SIGMA = 0.6745  # standard MAD -> z-equivalent conversion constant


def sweep_organs(
    volume: np.ndarray,
    seg_arr: np.ndarray,
    value_to_label: dict,
    organ_stats: dict,
    spacing: tuple,
    min_component_voxels: int = 10,
    min_persistent_slices: int = 3,
) -> list[Candidate]:
    """
    Independently sweep every segmented organ for statistical outlier voxels.

    Args:
        volume: 3D HU volume, shape (Z, Y, X).
        seg_arr: 3D integer label mask, same shape as volume. 0 = background.
        value_to_label: {int label value: str organ name}, e.g. {5: "liver"}.
        organ_stats: {str organ name: baseline dict from organ_baseline.py}.
        spacing: (row_mm, col_mm, z_mm) physical voxel spacing.
        min_component_voxels: Minimum connected-component size to keep.
        min_persistent_slices: Component spanning >= this many unique Z
            slices is marked persistence_ok=True (mirrors Package 1's
            finding_tracker persistence threshold).

    Returns:
        List of Candidate objects, tagged detected_by=["organ_sweep"].
    """
    row_mm, col_mm, z_mm = spacing
    voxel_volume_cc = (row_mm * col_mm * z_mm) / 1000.0

    candidates: list[Candidate] = []

    for val, label in value_to_label.items():
        if val == 0:
            continue

        baseline = organ_stats.get(label)
        if not baseline or baseline.get("voxel_count", 0) == 0:
            continue

        organ_mask = (seg_arr == val)
        if not np.any(organ_mask):
            continue

        outlier_mask = _find_outlier_voxels(volume, organ_mask, baseline)
        if not np.any(outlier_mask):
            continue

        labeled, num_components = ndimage.label(outlier_mask)
        if num_components == 0:
            continue

        slices = ndimage.find_objects(labeled)
        for comp_id, comp_slice in enumerate(slices, start=1):
            if comp_slice is None:
                continue

            sub_labeled = labeled[comp_slice]
            sub_mask = (sub_labeled == comp_id)
            voxel_count = int(np.sum(sub_mask))
            if voxel_count < min_component_voxels:
                continue

            sub_volume = volume[comp_slice]
            z_slice, y_slice, x_slice = comp_slice
            z_offset, y_offset, x_offset = z_slice.start, y_slice.start, x_slice.start

            candidates.append(_build_candidate_sub(
                sub_mask=sub_mask,
                sub_volume=sub_volume,
                offset=(z_offset, y_offset, x_offset),
                label=label,
                baseline=baseline,
                spacing=(row_mm, col_mm, z_mm),
                voxel_volume_cc=voxel_volume_cc,
                min_persistent_slices=min_persistent_slices,
            ))

    logger.info(
        "organ_sweep: %d organs scanned -> %d candidates",
        len(value_to_label), len(candidates),
    )
    return candidates


def _find_outlier_voxels(volume, organ_mask, baseline):
    """Majority-vote (>=2 of 4) outlier detection within one organ.
    
    OPTIMISED: operates only on the masked organ voxels (1-D array),
    then back-projects the result to a 3-D boolean mask.  This avoids
    creating four full-volume arrays (54 M elements each) for every organ.
    """
    trimmed_mean = baseline["trimmed_mean"]
    trimmed_std  = baseline["trimmed_std"]
    median       = baseline["median"]
    mad          = baseline["mad"]
    p5, p95      = baseline["p5"], baseline["p95"]
    q1, q3, iqr  = baseline["q1"], baseline["q3"], baseline["iqr"]

    # Extract only the voxels that belong to this organ — typically a few
    # thousand elements instead of 54 million.
    organ_voxels = volume[organ_mask]   # shape: (N,)
    votes = np.zeros(organ_voxels.shape, dtype=np.uint8)

    if trimmed_std > 1e-6:
        z = np.abs(organ_voxels - trimmed_mean) / trimmed_std
        votes += (z > Z_SCORE_THRESHOLD).astype(np.uint8)

    if mad > 1e-6:
        mad_score = MAD_TO_SIGMA * np.abs(organ_voxels - median) / mad
        votes += (mad_score > MAD_SCORE_THRESHOLD).astype(np.uint8)

    votes += ((organ_voxels < p5) | (organ_voxels > p95)).astype(np.uint8)

    if iqr > 1e-6:
        lower = q1 - IQR_FENCE_MULTIPLIER * iqr
        upper = q3 + IQR_FENCE_MULTIPLIER * iqr
        votes += ((organ_voxels < lower) | (organ_voxels > upper)).astype(np.uint8)

    # Back-project 1-D outlier flags into the 3-D volume space.
    outlier_flat = (votes >= 2)
    result = np.zeros(volume.shape, dtype=bool)
    result[organ_mask] = outlier_flat
    return result


def _build_candidate_sub(sub_mask, sub_volume, offset, label, baseline, spacing, voxel_volume_cc, min_persistent_slices):
    row_mm, col_mm, z_mm = spacing
    z_off, y_off, x_off = offset

    z_sub, y_sub, x_sub = np.nonzero(sub_mask)
    z_idx = z_sub + z_off
    y_idx = y_sub + y_off
    x_idx = x_sub + x_off

    z_min, z_max = int(z_idx.min()), int(z_idx.max())
    y_min, y_max = int(y_idx.min()), int(y_idx.max())
    x_min, x_max = int(x_idx.min()), int(x_idx.max())

    region_voxels = sub_volume[sub_mask]
    density = DensityHU(
        mean=round(float(np.mean(region_voxels)), 2),
        min=round(float(np.min(region_voxels)), 2),
        max=round(float(np.max(region_voxels)), 2),
        std=round(float(np.std(region_voxels)), 2),
    )

    voxel_count = int(np.sum(sub_mask))
    long_axis_mm = max(
        (x_max - x_min + 1) * col_mm,
        (y_max - y_min + 1) * row_mm,
        (z_max - z_min + 1) * z_mm,
    )
    short_axis_mm = min(
        (x_max - x_min + 1) * col_mm,
        (y_max - y_min + 1) * row_mm,
        (z_max - z_min + 1) * z_mm,
    )
    shape = ShapeFeatures(
        volume_cc=round(voxel_count * voxel_volume_cc, 3),
        long_axis_mm=round(long_axis_mm, 2),
        short_axis_mm=round(short_axis_mm, 2),
        sphericity=0.0,
        elongation=round(long_axis_mm / short_axis_mm, 3) if short_axis_mm > 0 else 0.0,
        margin_curvature_variance=0.0,
    )

    centroid_3d = [
        round(float(x_idx.mean()), 2),
        round(float(y_idx.mean()), 2),
        round(float(z_idx.mean()), 2),
    ]

    trimmed_mean = baseline["trimmed_mean"]
    trimmed_std = baseline["trimmed_std"]
    organ_local_zscore = (
        round(float(np.mean(np.abs(region_voxels - trimmed_mean) / trimmed_std)), 4)
        if trimmed_std > 1e-6 else 0.0
    )

    bbox_volume = (x_max - x_min + 1) * (y_max - y_min + 1) * (z_max - z_min + 1)
    organ_overlap_fraction = round(voxel_count / bbox_volume, 4) if bbox_volume > 0 else 0.0

    persistence_ok = len(np.unique(z_idx)) >= min_persistent_slices

    return Candidate(
        organ_label=label,
        detected_by=["organ_sweep"],
        bbox_3d=[x_min, y_min, z_min, x_max, y_max, z_max],
        centroid_3d=centroid_3d,
        slice_indices=sorted(set(int(z) for z in z_idx)),
        shape=shape,
        density_hu=density,
        organ_local_zscore=organ_local_zscore,
        organ_overlap_fraction=organ_overlap_fraction,
        persistence_ok=persistence_ok,
    )