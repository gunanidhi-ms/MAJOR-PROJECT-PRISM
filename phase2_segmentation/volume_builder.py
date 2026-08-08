"""
volume_builder.py — NIfTI Volume Assembly from HU Slice Stacks

Takes the (Z, Y, X) numpy volume and (row_mm, col_mm, z_mm) spacing tuple
produced by VolumeAccumulator.export_volume() and writes a correctly-headered
NIfTI file suitable for TotalSegmentator input.

CRITICAL SPACING NOTE:
    SimpleITK uses (X, Y, Z) spacing order internally.
    VolumeAccumulator returns spacing as (row_mm, col_mm, z_mm) = (Y, X, Z).
    This module handles the swap: SetSpacing((col_mm, row_mm, z_mm)).
    Getting this wrong produces a NIfTI that "looks right" but makes every
    downstream mm/cc measurement silently wrong.

Usage:
    from phase2_segmentation.volume_builder import assemble_nifti

    nifti_path = assemble_nifti(volume_array, spacing_meta=(0.7, 0.7, 1.5))
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


def validate_volume_array(volume_array: np.ndarray) -> None:
    """
    Validate the input volume array before NIfTI assembly.

    Args:
        volume_array: Expected shape (Z, Y, X), dtype float32 or float64.

    Raises:
        ValueError: If the array fails validation checks.
    """
    if volume_array is None:
        raise ValueError("volume_array is None")

    if volume_array.ndim != 3:
        raise ValueError(
            f"volume_array must be 3D (Z, Y, X), got {volume_array.ndim}D "
            f"with shape {volume_array.shape}"
        )

    z, y, x = volume_array.shape

    if z < 2:
        raise ValueError(
            f"volume_array has only {z} slices along Z — need at least 2 "
            f"for a meaningful volume"
        )

    if y < 16 or x < 16:
        raise ValueError(
            f"volume_array spatial dimensions too small: ({y}, {x}). "
            f"Expected at least (16, 16)"
        )

    if volume_array.size == 0:
        raise ValueError("volume_array is empty (zero elements)")

    # Check for all-identical values (likely corrupted)
    if volume_array.min() == volume_array.max():
        raise ValueError(
            f"volume_array has constant value {volume_array.min()} — "
            f"likely corrupted or uninitialized"
        )

    # Check for extreme HU values that suggest unprocessed/corrupted data
    vmin, vmax = float(volume_array.min()), float(volume_array.max())
    if vmin < -2000 or vmax > 5000:
        logger.warning(
            "volume_array has extreme HU values (min=%.1f, max=%.1f). "
            "Expected range: [-1024, 3071]. Proceeding but results may be "
            "unreliable.",
            vmin,
            vmax,
        )

    logger.debug(
        "Volume validated: shape=%s, dtype=%s, HU range=[%.1f, %.1f]",
        volume_array.shape,
        volume_array.dtype,
        vmin,
        vmax,
    )


def validate_spacing(spacing_meta: Tuple[float, float, float]) -> None:
    """
    Validate the spacing tuple before NIfTI assembly.

    Args:
        spacing_meta: (row_mm, col_mm, z_mm) from VolumeAccumulator.

    Raises:
        ValueError: If spacing values are invalid.
    """
    if spacing_meta is None:
        raise ValueError("spacing_meta is None")

    if len(spacing_meta) != 3:
        raise ValueError(
            f"spacing_meta must be a 3-tuple (row_mm, col_mm, z_mm), "
            f"got length {len(spacing_meta)}"
        )

    row_mm, col_mm, z_mm = spacing_meta

    for name, val in [("row_mm", row_mm), ("col_mm", col_mm), ("z_mm", z_mm)]:
        if not isinstance(val, (int, float)):
            raise ValueError(f"spacing_meta.{name} must be numeric, got {type(val)}")
        if val <= 0:
            raise ValueError(
                f"spacing_meta.{name} must be positive, got {val}. "
                f"Zero or negative spacing produces invalid NIfTI geometry."
            )
        if val > 50.0:
            logger.warning(
                "spacing_meta.%s = %.2f mm is unusually large "
                "(typical CT range: 0.3–5.0 mm). Proceeding anyway.",
                name,
                val,
            )

    logger.debug("Spacing validated: row=%.3f, col=%.3f, z=%.3f mm", row_mm, col_mm, z_mm)


def assemble_nifti(
    volume_array: np.ndarray,
    spacing_meta: Tuple[float, float, float],
    out_path: str = "temp_volume.nii.gz",
    orientation_patient: Optional[list] = None,
) -> str:
    """
    Assemble a correctly-headered NIfTI from the HU volume and spacing.

    Args:
        volume_array: 3D numpy array of shape (Z, Y, X) in Hounsfield Units.
                      This is the first element of VolumeAccumulator.export_volume().
        spacing_meta: Tuple of (row_mm, col_mm, z_mm).
                      This is the fourth element of VolumeAccumulator.export_volume().
                      IMPORTANT: This is (Y-spacing, X-spacing, Z-spacing).
        out_path: File path for the output NIfTI. Defaults to "temp_volume.nii.gz".
                  Parent directories will be created if they don't exist.
        orientation_patient: Optional list of 6 DICOM direction cosines
                            [row_x, row_y, row_z, col_x, col_y, col_z].
                            If None, identity orientation is used.

    Returns:
        Absolute path to the written NIfTI file.

    Raises:
        ValueError: If inputs fail validation.
        RuntimeError: If SimpleITK fails to write the file.
    """
    # ── Validate inputs ──
    validate_volume_array(volume_array)
    validate_spacing(spacing_meta)

    row_mm, col_mm, z_mm = spacing_meta

    # ── Import SimpleITK lazily to give clear error if not installed ──
    try:
        import SimpleITK as sitk
    except ImportError as e:
        raise RuntimeError(
            "SimpleITK is required for NIfTI assembly. "
            "Install it with: pip install SimpleITK"
        ) from e

    # ── Ensure output directory exists ──
    out_abs = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)

    # ── Convert numpy array to SimpleITK Image ──
    # SimpleITK.GetImageFromArray expects (Z, Y, X) ordering — which is
    # exactly what VolumeAccumulator.export_volume() produces.
    volume_f32 = volume_array.astype(np.float32, copy=False)
    image = sitk.GetImageFromArray(volume_f32)

    # ── Set spacing ──
    # CRITICAL: SimpleITK spacing is (X, Y, Z) = (col_mm, row_mm, z_mm)
    # VolumeAccumulator returns (row_mm, col_mm, z_mm) = (Y_spacing, X_spacing, Z_spacing)
    # We MUST swap row and col here.
    sitk_spacing = (float(col_mm), float(row_mm), float(z_mm))
    image.SetSpacing(sitk_spacing)

    logger.info(
        "NIfTI spacing set: SimpleITK (X,Y,Z) = (%.4f, %.4f, %.4f) mm  "
        "[from accumulator (row,col,z) = (%.4f, %.4f, %.4f) mm]",
        *sitk_spacing,
        row_mm,
        col_mm,
        z_mm,
    )

    # ── Set origin ──
    # Default to (0, 0, 0). In a full clinical pipeline this would come
    # from ImagePositionPatient of the first slice. For TotalSegmentator
    # input this is acceptable.
    image.SetOrigin((0.0, 0.0, 0.0))

    # ── Set direction ──
    # Convert DICOM ImageOrientationPatient (6 cosines) to a 3×3 direction
    # matrix, or use identity if not provided.
    if orientation_patient and len(orientation_patient) == 6:
        row_cosines = orientation_patient[:3]
        col_cosines = orientation_patient[3:6]
        # Compute the slice normal as cross product of row × col
        normal = [
            row_cosines[1] * col_cosines[2] - row_cosines[2] * col_cosines[1],
            row_cosines[2] * col_cosines[0] - row_cosines[0] * col_cosines[2],
            row_cosines[0] * col_cosines[1] - row_cosines[1] * col_cosines[0],
        ]
        # SimpleITK direction is a flattened 3×3 matrix (row-major)
        direction = (
            row_cosines[0], col_cosines[0], normal[0],
            row_cosines[1], col_cosines[1], normal[1],
            row_cosines[2], col_cosines[2], normal[2],
        )
        image.SetDirection(direction)
        logger.debug("Direction set from DICOM orientation cosines")
    else:
        # Identity direction — standard axial orientation
        image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        logger.debug("Direction set to identity (standard axial)")

    # ── Write NIfTI ──
    try:
        sitk.WriteImage(image, out_abs)
    except Exception as e:
        raise RuntimeError(
            f"SimpleITK failed to write NIfTI to {out_abs}: {e}"
        ) from e

    # ── Verify the file was actually written ──
    if not os.path.isfile(out_abs):
        raise RuntimeError(f"NIfTI file was not created at {out_abs}")

    file_size_mb = os.path.getsize(out_abs) / (1024 * 1024)
    logger.info(
        "NIfTI written: %s (%.1f MB), volume shape=%s",
        out_abs,
        file_size_mb,
        volume_array.shape,
    )

    return out_abs
