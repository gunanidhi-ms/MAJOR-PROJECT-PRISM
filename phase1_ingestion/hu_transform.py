"""
hu_transform.py — Vectorized Hounsfield Unit Transformation

Converts raw DICOM pixel values to Hounsfield Units (HU) using the
linear rescaling formula:

    HU = RescaleSlope × pixel_value + RescaleIntercept

This is a fully vectorized NumPy operation on the entire 2D slice array
(no per-pixel loops). Target performance: <10ms for a 512×512 array.

Typical LUNA16 values:
    RescaleSlope = 1.0
    RescaleIntercept = -1024.0

HU reference scale:
    -1000 HU = Air
     -800 HU = Lung parenchyma
     -500 HU = Fat
        0 HU = Water
      +40 HU = Soft tissue
     +400 HU = Bone (cancellous)
    +1000 HU = Bone (cortical)
"""

import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

# Clinical HU bounds — values outside this range are non-physiological
HU_MIN = -1024.0
HU_MAX = 3071.0


def apply_hu_transform(
    pixel_array: np.ndarray,
    slope: float = 1.0,
    intercept: float = -1024.0,
    clip: bool = False,
) -> np.ndarray:
    """
    Apply the Hounsfield Unit linear rescale to a raw pixel array.
    
    Fully vectorized — operates on the entire array in a single NumPy
    expression. Converts to float32 for downstream processing.
    
    Args:
        pixel_array: 2D NumPy array of raw stored pixel values (int16).
        slope: DICOM RescaleSlope (0028,1053). Default 1.0.
        intercept: DICOM RescaleIntercept (0028,1052). Default -1024.0.
        clip: If True, clip output to [HU_MIN, HU_MAX] range.
    
    Returns:
        2D float32 NumPy array of Hounsfield Unit values.
    
    Performance:
        <10ms for 512×512 array on modern hardware.
    """
    t_start = time.perf_counter()

    # Core transform — single vectorized operation
    hu_array = pixel_array.astype(np.float32) * float(slope) + float(intercept)

    if clip:
        np.clip(hu_array, HU_MIN, HU_MAX, out=hu_array)

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    logger.debug(
        "HU transform: %dx%d in %.2fms (slope=%.2f, intercept=%.2f)",
        pixel_array.shape[0],
        pixel_array.shape[1] if pixel_array.ndim > 1 else 1,
        elapsed_ms,
        slope,
        intercept,
    )

    return hu_array


def apply_window(
    hu_array: np.ndarray,
    window_center: float,
    window_width: float,
) -> np.ndarray:
    """
    Apply a windowing function to an HU array for visualization.
    
    Maps HU values within the window to [0, 255] uint8 range.
    Values below the window → 0 (black).
    Values above the window → 255 (white).
    
    Args:
        hu_array: 2D float32 array of HU values.
        window_center: Center of the display window (HU).
        window_width: Width of the display window (HU).
    
    Returns:
        2D uint8 array suitable for image display.
    
    Common CT windows:
        Lung:       center=-600, width=1500
        Mediastinum: center=40,  width=400
        Bone:       center=400,  width=1800
        Soft tissue: center=50,  width=350
    """
    lower = window_center - window_width / 2.0
    upper = window_center + window_width / 2.0

    windowed = np.clip(hu_array, lower, upper)
    # Normalize to 0-255
    windowed = ((windowed - lower) / (upper - lower) * 255.0).astype(np.uint8)

    return windowed


def hu_stats(hu_array: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """
    Compute HU statistics for a region of interest.
    
    Args:
        hu_array: 2D float32 array of HU values.
        mask: Optional boolean mask; if provided, stats are computed
              only within the masked region.
    
    Returns:
        Dict with keys: mean, std, min, max, median
    """
    if mask is not None:
        values = hu_array[mask]
    else:
        values = hu_array.ravel()

    if values.size == 0:
        return {
            "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0,
            "q10": 0.0, "q25": 0.0, "q50": 0.0, "q75": 0.0, "q90": 0.0, "q99": 0.0
        }

    quantiles = np.percentile(values, [10, 25, 50, 75, 90, 99])

    return {
        "mean": round(float(np.mean(values)), 2),
        "std": round(float(np.std(values)), 2),
        "min": round(float(np.min(values)), 2),
        "max": round(float(np.max(values)), 2),
        "median": round(float(quantiles[2]), 2), # median is q50
        "q10": round(float(quantiles[0]), 2),
        "q25": round(float(quantiles[1]), 2),
        "q50": round(float(quantiles[2]), 2),
        "q75": round(float(quantiles[3]), 2),
        "q90": round(float(quantiles[4]), 2),
        "q99": round(float(quantiles[5]), 2),
    }

def auto_crop(hu_array: np.ndarray, threshold: float = -1500.0, padding: int = 10) -> np.ndarray:
    """
    Automatically crop scanner padding from DICOM images.
    
    Many scanners embed the CT image in a larger padded array (e.g., 512x693
    where the body is only 512x386 and the rest is padding at -8192 HU).
    
    This function finds the bounding box of all pixels above the threshold
    (which should be just above the minimum valid HU of -1024) and crops
    the image to that region plus a small padding margin.
    
    Works universally for any body part — lungs, abdomen, brain, bone.
    
    Args:
        hu_array: 2D float32 array of Hounsfield Units.
        threshold: HU value below which pixels are considered padding.
                   Default -1500 catches padding at -2048, -3024, -4096
                   while preserving air (-1024 HU) and all body tissue.
        padding: Number of pixels of margin to keep around the body.
    
    Returns:
        Cropped 2D float32 array. Returns original if no cropping needed.
    """
    mask = hu_array > threshold
    if not np.any(mask):
        return hu_array

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Only crop if we'd actually remove a significant portion (>10% padding)
    body_area = (rmax - rmin + 1) * (cmax - cmin + 1)
    total_area = hu_array.shape[0] * hu_array.shape[1]
    if body_area / total_area > 0.95:
        return hu_array  # Not much padding, skip cropping

    rmin = max(0, int(rmin) - padding)
    rmax = min(hu_array.shape[0], int(rmax) + padding + 1)
    cmin = max(0, int(cmin) - padding)
    cmax = min(hu_array.shape[1], int(cmax) + padding + 1)

    return hu_array[rmin:rmax, cmin:cmax]


