"""
seed_clusterer.py -- Package 4: Path A, 3D Clustering of Phase 1 Seeds

Converts finding_tracker.py tracks (multi-slice-linked 2D Findings) into
3D Candidate objects using the shared schema from schemas/candidate.py.

Z-index note: slice_idx in each track tuple IS the volume's Z-axis index --
both come from the same ordered_findings list returned by
VolumeAccumulator.export_volume(), so no instance-number lookup is needed.
"""

import logging
import numpy as np

from schemas.candidate import Candidate, ShapeFeatures, DensityHU
from phase1_ingestion.finding_tracker import track_persistence_ok

logger = logging.getLogger(__name__)


def cluster_phase1_seeds(
    tracks: list,
    volume: np.ndarray,           # shape (Z, Y, X), from export_volume()
    spacing: tuple,                # (row_mm, col_mm, z_mm)
    min_slices: int = 3,
) -> list[Candidate]:
    """
    Convert Phase 1 finding tracks into 3D Candidate objects.

    Args:
        tracks: Output of finding_tracker.link_findings_across_slices() --
                list of tracks, each a list of (slice_idx, Finding) tuples.
        volume: 3D HU volume, shape (Z, Y, X).
        spacing: (row_mm, col_mm, z_mm) physical voxel spacing.
        min_slices: Minimum track length to be considered persistent.

    Returns:
        List of Candidate objects, one per persistent track.
    """
    row_mm, col_mm, z_mm = spacing
    voxel_volume_cc = (row_mm * col_mm * z_mm) / 1000.0  # mm^3 -> cc

    candidates: list[Candidate] = []

    for track in tracks:
        if not track_persistence_ok(track, min_slices=min_slices):
            continue

        bbox_3d = _compute_bbox_3d(track)
        x_min, y_min, z_min, x_max, y_max, z_max = bbox_3d

        z_min_c, z_max_c = max(0, z_min), min(volume.shape[0] - 1, z_max)
        y_min_c, y_max_c = max(0, y_min), min(volume.shape[1] - 1, y_max)
        x_min_c, x_max_c = max(0, x_min), min(volume.shape[2] - 1, x_max)

        region = volume[
            z_min_c:z_max_c + 1,
            y_min_c:y_max_c + 1,
            x_min_c:x_max_c + 1,
        ]

        if region.size == 0:
            logger.warning("seed_clusterer: empty region for track, skipping")
            continue

        density = DensityHU(
            mean=round(float(np.mean(region)), 2),
            min=round(float(np.min(region)), 2),
            max=round(float(np.max(region)), 2),
            std=round(float(np.std(region)), 2),
        )

        long_axis_mm = max(
            (x_max - x_min + 1) * col_mm,
            (y_max - y_min + 1) * row_mm,
        )
        short_axis_mm = min(
            (x_max - x_min + 1) * col_mm,
            (y_max - y_min + 1) * row_mm,
        )

        shape = ShapeFeatures(
            volume_cc=round(region.size * voxel_volume_cc, 3),
            long_axis_mm=round(long_axis_mm, 2),
            short_axis_mm=round(short_axis_mm, 2),
            sphericity=0.0,
            elongation=round(long_axis_mm / short_axis_mm, 3) if short_axis_mm > 0 else 0.0,
            margin_curvature_variance=0.0,
        )

        centroid_3d = _compute_centroid_3d(track)

        strongest = max(track, key=lambda t: getattr(t[1], "confidence", 0.0))
        strongest_finding = strongest[1]

        candidate = Candidate(
            organ_label="unclassified",
            detected_by=["phase1_seed"],
            bbox_3d=[x_min, y_min, z_min, x_max, y_max, z_max],
            centroid_3d=centroid_3d,
            slice_indices=[t[0] for t in track],
            shape=shape,
            density_hu=density,
            phase1_confidence=getattr(strongest_finding, "confidence", 0.0),
            phase1_severity=getattr(strongest_finding, "severity_score", 0.0),
            phase1_anomaly_type=getattr(strongest_finding, "anomaly_type", ""),
            persistence_ok=True,
        )
        candidates.append(candidate)

    logger.info(
        "seed_clusterer: %d tracks -> %d candidates (min_slices=%d)",
        len(tracks), len(candidates), min_slices,
    )
    return candidates


def _compute_bbox_3d(track) -> tuple:
    xs_min, ys_min, xs_max, ys_max, zs = [], [], [], [], []
    for slice_idx, finding in track:
        x, y, w, h = finding.bbox
        xs_min.append(x)
        ys_min.append(y)
        xs_max.append(x + w)
        ys_max.append(y + h)
        zs.append(slice_idx)
    return (
        min(xs_min), min(ys_min), min(zs),
        max(xs_max), max(ys_max), max(zs),
    )


def _compute_centroid_3d(track) -> list:
    xs = [f.centroid[0] for _, f in track]
    ys = [f.centroid[1] for _, f in track]
    zs = [s for s, _ in track]
    return [
        round(sum(xs) / len(xs), 2),
        round(sum(ys) / len(ys), 2),
        round(sum(zs) / len(zs), 2),
    ]
