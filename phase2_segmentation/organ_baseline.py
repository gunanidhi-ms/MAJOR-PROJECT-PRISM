"""
organ_baseline.py — Patient-Specific Organ Baseline Calculator

Computes trimmed (P5-P95) mean, standard deviation, and robust statistical 
metrics (median, MAD, IQR, etc.) computed dynamically from the patient's own 
segmented organ voxels.

This module contains NO population-level HU lookup tables or hardcoded values.
"""

import numpy as np

def compute_organ_baseline(
    organ_voxels_hu: np.ndarray,
    voxel_volume_cc: float = 0.0,
) -> dict:
    """
    Computes patient-specific statistical baseline from the given HU voxels of an organ.

    Args:
        organ_voxels_hu: 1D numpy array of HU intensities inside the organ mask.
        voxel_volume_cc: Physical volume of a single voxel in cubic centimeters.

    Returns:
        dict: Conforming to schemas/organ_statistics.json.
    """
    voxel_count = int(organ_voxels_hu.size)
    volume_cc = float(voxel_count * voxel_volume_cc)

    if voxel_count == 0:
        return {
            "trimmed_mean": 0.0,
            "trimmed_std": 0.0,
            "median": 0.0,
            "mad": 0.0,
            "p5": 0.0,
            "p95": 0.0,
            "q1": 0.0,
            "q3": 0.0,
            "iqr": 0.0,
            "voxel_count": 0,
            "volume_cc": 0.0,
        }

    # Extract percentile thresholds
    p5, q1, median_val, q3, p95 = np.percentile(
        organ_voxels_hu, [5, 25, 50, 75, 95]
    )

    # Trimmed dataset: values within the P5 and P95 boundaries (inclusive)
    trimmed_mask = (organ_voxels_hu >= p5) & (organ_voxels_hu <= p95)
    trimmed = organ_voxels_hu[trimmed_mask]

    if trimmed.size > 0:
        trimmed_mean = float(np.mean(trimmed))
        trimmed_std = float(np.std(trimmed))
    else:
        trimmed_mean = float(np.mean(organ_voxels_hu))
        trimmed_std = float(np.std(organ_voxels_hu))

    # Robust metrics on the full array
    mad = float(np.median(np.abs(organ_voxels_hu - median_val)))
    iqr = float(q3 - q1)

    return {
        "trimmed_mean": trimmed_mean,
        "trimmed_std": trimmed_std,
        "median": float(median_val),
        "mad": mad,
        "p5": float(p5),
        "p95": float(p95),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": iqr,
        "voxel_count": voxel_count,
        "volume_cc": volume_cc,
    }
