"""
generate_test_dicoms.py — Synthetic DICOM CT Slice Generator (LUNA16-style)

Generates realistic 512×512 CT slices with proper DICOM headers mimicking
LUNA16 dataset characteristics. Embeds synthetic "nodule" blobs at clinically
accurate Hounsfield Unit (HU) values for testing the triage pipeline.

LUNA16 CT characteristics:
  - Matrix: 512 × 512 pixels
  - Bit depth: 16-bit signed integers
  - RescaleSlope: 1.0, RescaleIntercept: -1024
  - Pixel spacing: ~0.6–0.8 mm
  - Slice thickness: 1.0–2.5 mm
  - HU range: -1024 (air) to ~3000 (bone/metal)

Tissue HU windows used:
  - Air/outside:        -1000 HU
  - Lung parenchyma:    -700 to -500 HU
  - Ground-glass opacity: -700 to -300 HU
  - Soft tissue:        -100 to +100 HU
  - Solid nodules:       -50 to +100 HU
"""

import os
import sys
import numpy as np
from datetime import datetime, date
from pathlib import Path

try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    from pydicom.sequence import Sequence
except ImportError:
    print("ERROR: pydicom is required. Install with: pip install pydicom")
    sys.exit(1)


# ── LUNA16-realistic DICOM parameters ──────────────────────────────────────────

ROWS = 512
COLS = 512
RESCALE_SLOPE = 1.0
RESCALE_INTERCEPT = -1024.0
BITS_ALLOCATED = 16
BITS_STORED = 16
HIGH_BIT = 15
PIXEL_REPRESENTATION = 1  # signed
SAMPLES_PER_PIXEL = 1
PHOTOMETRIC_INTERPRETATION = "MONOCHROME2"
PIXEL_SPACING = [0.703125, 0.703125]  # mm, typical LUNA16
SLICE_THICKNESS = 1.25  # mm
SERIES_DESCRIPTION = "PRISM Synthetic CT — LUNA16 Mimic"


def _make_lung_background(rng: np.random.Generator) -> np.ndarray:
    """
    Create a realistic CT background in stored pixel values.
    
    Returns a 512×512 int16 array in *stored* pixel space
    (actual HU = stored * slope + intercept).
    
    Structure:
      - Outer region: air (-1000 HU → stored = 24)
      - Elliptical body: soft tissue (~40 HU → stored = 1064)
      - Two lung fields: lung parenchyma (-800 HU → stored = 224)
    """
    # Start with air: HU = -1000 → stored = (-1000 - intercept) / slope
    air_stored = int((-1000.0 - RESCALE_INTERCEPT) / RESCALE_SLOPE)
    img = np.full((ROWS, COLS), air_stored, dtype=np.int16)

    yy, xx = np.mgrid[0:ROWS, 0:COLS]
    cy, cx = ROWS // 2, COLS // 2

    # ── Body ellipse (soft tissue ~40 HU) ──
    body_mask = ((xx - cx) / 200.0) ** 2 + ((yy - cy) / 160.0) ** 2 < 1.0
    tissue_hu = 40.0  # soft tissue
    tissue_stored = int((tissue_hu - RESCALE_INTERCEPT) / RESCALE_SLOPE)
    img[body_mask] = tissue_stored

    # ── Add some texture noise to body (±15 HU) ──
    noise = rng.normal(0, 15, (ROWS, COLS)).astype(np.int16)
    img[body_mask] += noise[body_mask]

    # ── Left lung field ──
    left_lung_mask = (
        ((xx - (cx - 70)) / 90.0) ** 2 + ((yy - (cy + 10)) / 120.0) ** 2 < 1.0
    )
    lung_hu = -800.0
    lung_stored = int((lung_hu - RESCALE_INTERCEPT) / RESCALE_SLOPE)
    lung_noise = rng.normal(0, 30, (ROWS, COLS)).astype(np.int16)
    img[left_lung_mask] = lung_stored + lung_noise[left_lung_mask]

    # ── Right lung field ──
    right_lung_mask = (
        ((xx - (cx + 70)) / 90.0) ** 2 + ((yy - (cy + 10)) / 120.0) ** 2 < 1.0
    )
    img[right_lung_mask] = lung_stored + lung_noise[right_lung_mask]

    return img, left_lung_mask, right_lung_mask


def _embed_nodule(
    img: np.ndarray,
    lung_mask: np.ndarray,
    hu_value: float,
    radius_px: int,
    rng: np.random.Generator,
) -> tuple:
    """
    Embed a circular nodule blob inside a lung field.
    
    Returns (center_y, center_x, radius) of placed nodule, or None if 
    placement failed.
    """
    # Find valid positions inside the lung field
    valid_ys, valid_xs = np.where(lung_mask)
    if len(valid_ys) == 0:
        return None

    # Pick a random center, ensure nodule fits inside lung
    for _ in range(50):  # max placement attempts
        idx = rng.integers(0, len(valid_ys))
        cy, cx = int(valid_ys[idx]), int(valid_xs[idx])

        # Check the nodule circle fits within lung mask
        yy, xx = np.mgrid[0:ROWS, 0:COLS]
        circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius_px ** 2

        if np.all(lung_mask[circle]):
            nodule_stored = int((hu_value - RESCALE_INTERCEPT) / RESCALE_SLOPE)
            # Add slight internal texture (±8 HU)
            nodule_noise = rng.normal(0, 8, (ROWS, COLS)).astype(np.int16)
            img[circle] = nodule_stored + nodule_noise[circle]
            return (cy, cx, radius_px)

    return None


def generate_dicom_slice(
    instance_number: int,
    study_uid: str,
    series_uid: str,
    patient_id: str = "PRISM_TEST_001",
    embed_anomaly: str | None = None,
    rng: np.random.Generator = None,
) -> tuple:
    """
    Generate a single synthetic DICOM CT slice.
    
    Args:
        instance_number: Slice position in the series (1-indexed)
        study_uid: DICOM Study Instance UID
        series_uid: DICOM Series Instance UID
        patient_id: Patient ID tag
        embed_anomaly: None, "ground_glass", or "solid_nodule"
        rng: NumPy random generator for reproducibility
    
    Returns:
        (FileDataset, nodule_info) — the DICOM dataset and optional nodule metadata
    """
    if rng is None:
        rng = np.random.default_rng()

    # ── Build pixel data ──
    img, left_lung, right_lung = _make_lung_background(rng)
    nodule_info = None

    if embed_anomaly == "ground_glass":
        hu = rng.uniform(-600, -400)  # ground-glass range
        radius = rng.integers(8, 20)
        lung = left_lung if rng.random() > 0.5 else right_lung
        nodule_info = _embed_nodule(img, lung, hu, radius, rng)
        if nodule_info:
            nodule_info = {
                "type": "ground_glass",
                "center_y": nodule_info[0],
                "center_x": nodule_info[1],
                "radius_px": nodule_info[2],
                "hu_target": float(hu),
            }

    elif embed_anomaly == "solid_nodule":
        hu = rng.uniform(-20, 80)  # solid nodule range
        radius = rng.integers(5, 15)
        lung = right_lung if rng.random() > 0.5 else left_lung
        nodule_info = _embed_nodule(img, lung, hu, radius, rng)
        if nodule_info:
            nodule_info = {
                "type": "solid_nodule",
                "center_y": nodule_info[0],
                "center_x": nodule_info[1],
                "radius_px": nodule_info[2],
                "hu_target": float(hu),
            }

    # ── Build DICOM FileDataset ──
    sop_uid = generate_uid()
    filename = f"slice_{instance_number:04d}.dcm"

    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\x00" * 128)

    # ── Patient Module ──
    ds.PatientName = "PRISM^Test^Patient"
    ds.PatientID = patient_id
    ds.PatientBirthDate = "19700101"
    ds.PatientSex = "O"

    # ── Study Module ──
    ds.StudyInstanceUID = study_uid
    ds.StudyDate = date.today().strftime("%Y%m%d")
    ds.StudyTime = datetime.now().strftime("%H%M%S.%f")
    ds.StudyDescription = "PRISM Synthetic CT Study"
    ds.StudyID = "PRISM001"

    # ── Series Module ──
    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = 1
    ds.SeriesDescription = SERIES_DESCRIPTION
    ds.Modality = "CT"

    # ── Image Module (LUNA16-matching) ──
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ds.SOPInstanceUID = sop_uid
    ds.InstanceNumber = instance_number
    ds.ImagePositionPatient = [0.0, 0.0, float(instance_number) * SLICE_THICKNESS]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.PixelSpacing = PIXEL_SPACING
    ds.SliceThickness = SLICE_THICKNESS
    ds.SliceLocation = float(instance_number) * SLICE_THICKNESS

    # ── Pixel Data encoding ──
    ds.Rows = ROWS
    ds.Columns = COLS
    ds.BitsAllocated = BITS_ALLOCATED
    ds.BitsStored = BITS_STORED
    ds.HighBit = HIGH_BIT
    ds.PixelRepresentation = PIXEL_REPRESENTATION
    ds.SamplesPerPixel = SAMPLES_PER_PIXEL
    ds.PhotometricInterpretation = PHOTOMETRIC_INTERPRETATION
    ds.RescaleSlope = str(RESCALE_SLOPE)
    ds.RescaleIntercept = str(RESCALE_INTERCEPT)
    ds.RescaleType = "HU"
    ds.WindowCenter = "-600"
    ds.WindowWidth = "1500"

    ds.PixelData = img.tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    return ds, nodule_info


def generate_test_series(
    output_dir: str,
    num_slices: int = 20,
    seed: int = 42,
) -> list:
    """
    Generate a full synthetic CT series mimicking LUNA16 data.
    
    Creates `num_slices` DICOM files with:
      - ~30% containing ground-glass opacities
      - ~20% containing solid nodules
      - ~50% clean (no anomalies)
    
    Args:
        output_dir: Directory to write .dcm files
        num_slices: Number of slices to generate
        seed: Random seed for reproducibility
    
    Returns:
        List of dicts with slice metadata and any embedded nodule info
    """
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)

    study_uid = generate_uid()
    series_uid = generate_uid()

    # Decide which slices get anomalies
    anomaly_schedule = []
    for i in range(num_slices):
        r = rng.random()
        if r < 0.3:
            anomaly_schedule.append("ground_glass")
        elif r < 0.5:
            anomaly_schedule.append("solid_nodule")
        else:
            anomaly_schedule.append(None)

    manifest = []
    for i in range(num_slices):
        instance_num = i + 1
        anomaly_type = anomaly_schedule[i]

        ds, nodule_info = generate_dicom_slice(
            instance_number=instance_num,
            study_uid=study_uid,
            series_uid=series_uid,
            embed_anomaly=anomaly_type,
            rng=rng,
        )

        filepath = os.path.join(output_dir, f"slice_{instance_num:04d}.dcm")
        ds.save_as(filepath)

        entry = {
            "file": filepath,
            "instance_number": instance_num,
            "anomaly_type": anomaly_type,
            "nodule_info": nodule_info,
        }
        manifest.append(entry)

        status = f"  ✓ Slice {instance_num:3d}/{num_slices}"
        if anomaly_type:
            status += f"  [{anomaly_type}]"
            if nodule_info:
                status += f"  HU={nodule_info['hu_target']:.0f}, r={nodule_info['radius_px']}px"
        print(status)

    print(f"\n✅ Generated {num_slices} DICOM slices in: {output_dir}")
    print(f"   Study UID:  {study_uid}")
    print(f"   Series UID: {series_uid}")
    print(f"   Anomalies:  {sum(1 for a in anomaly_schedule if a is not None)} / {num_slices}")

    return manifest


if __name__ == "__main__":
    output = os.path.join(os.path.dirname(__file__), "sample_dicoms")
    generate_test_series(output, num_slices=20, seed=42)
