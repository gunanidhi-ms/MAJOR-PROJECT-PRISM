"""
finding_tracker.py — Cross-Slice Finding Tracker

Links Phase 1 per-slice findings into multi-slice "tracks" based on
centroid proximity across consecutive slices. Filters tracks by
persistence (minimum slice span) to suppress single-slice noise.

This module needs ZERO DICOM data — it operates purely on lists of
Finding objects, so it can be fully tested with synthetic inputs.

Usage:
    from phase1_ingestion.finding_tracker import (
        link_findings_across_slices,
        track_persistence_ok,
    )

    tracks = link_findings_across_slices(ordered_findings)
    for track in tracks:
        if track_persistence_ok(track, min_slices=3):
            # This is a spatially persistent finding
            ...

Full implementation: FINAL.md §1.3
"""

import logging
import math
from typing import List, Tuple

logger = logging.getLogger(__name__)


def _centroid_distance(c1: List[float], c2: List[float]) -> float:
    """
    Euclidean distance between two 2D centroids.

    Args:
        c1: [x, y] centroid of finding 1.
        c2: [x, y] centroid of finding 2.

    Returns:
        Euclidean distance in pixels.
    """
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    return math.sqrt(dx * dx + dy * dy)


def link_findings_across_slices(
    ordered_findings: List[list],
    max_centroid_drift_px: float = 15.0,
) -> List[List[Tuple[int, object]]]:
    """
    Link findings across consecutive slices into multi-slice tracks.

    A finding on slice N is linked to a finding on slice N+1 if their
    2D centroids are within max_centroid_drift_px pixels. Each finding
    can belong to at most one track (greedy nearest-neighbor matching).

    Args:
        ordered_findings: List of lists — one inner list of Finding objects
                          per slice, in spatial (Z) order.  Each Finding
                          must have a .centroid attribute ([x, y]).
        max_centroid_drift_px: Maximum centroid displacement (pixels) between
                               consecutive slices for two findings to be
                               considered part of the same structure.

    Returns:
        List of tracks. Each track is a list of (slice_index, Finding) tuples,
        sorted by slice_index.
    """
    # Active tracks: each is a list of (slice_index, finding)
    active_tracks: List[List[Tuple[int, object]]] = []
    completed_tracks: List[List[Tuple[int, object]]] = []

    for slice_idx, findings in enumerate(ordered_findings):
        if not findings:
            # No findings on this slice — all active tracks become completed
            completed_tracks.extend(active_tracks)
            active_tracks = []
            continue

        # Try to match each finding on this slice to an active track
        used_findings = set()
        used_tracks = set()
        matches = []

        # Compute all pairwise distances between active track heads and
        # current slice findings, then greedily match closest pairs
        for t_idx, track in enumerate(active_tracks):
            last_slice_idx, last_finding = track[-1]

            # Only link to findings on the immediately previous or current slice
            # (allow gap of at most 1 slice for robustness)
            if slice_idx - last_slice_idx > 2:
                continue

            last_centroid = _get_centroid(last_finding)
            if last_centroid is None:
                continue

            for f_idx, finding in enumerate(findings):
                finding_centroid = _get_centroid(finding)
                if finding_centroid is None:
                    continue

                dist = _centroid_distance(last_centroid, finding_centroid)
                if dist <= max_centroid_drift_px:
                    matches.append((dist, t_idx, f_idx))

        # Sort by distance — greedily assign closest matches first
        matches.sort(key=lambda x: x[0])

        for dist, t_idx, f_idx in matches:
            if t_idx in used_tracks or f_idx in used_findings:
                continue
            active_tracks[t_idx].append((slice_idx, findings[f_idx]))
            used_tracks.add(t_idx)
            used_findings.add(f_idx)

        # Unmatched active tracks → completed
        for t_idx in range(len(active_tracks)):
            if t_idx not in used_tracks:
                completed_tracks.append(active_tracks[t_idx])

        # Keep only matched active tracks
        active_tracks = [
            active_tracks[t_idx]
            for t_idx in range(len(active_tracks))
            if t_idx in used_tracks
        ]

        # Unmatched findings on this slice → new active tracks
        for f_idx, finding in enumerate(findings):
            if f_idx not in used_findings:
                active_tracks.append([(slice_idx, finding)])

    # Finalize remaining active tracks
    completed_tracks.extend(active_tracks)

    logger.info(
        "FindingTracker: linked %d total findings into %d tracks",
        sum(len(fs) for fs in ordered_findings),
        len(completed_tracks),
    )

    return completed_tracks


def track_persistence_ok(
    track: List[Tuple[int, object]],
    min_slices: int = 3,
) -> bool:
    """
    Check whether a track spans enough slices to be considered persistent.

    A single-slice or two-slice blip is likely noise; a finding that
    persists across >= min_slices consecutive slices is far more likely
    to be a real 3D structure.

    Args:
        track: List of (slice_index, Finding) tuples from
               link_findings_across_slices().
        min_slices: Minimum number of slices the track must span.

    Returns:
        True if the track spans >= min_slices slices.
    """
    if not track:
        return False
    return len(track) >= min_slices


def get_track_summary(track: List[Tuple[int, object]]) -> dict:
    """
    Compute summary statistics for a finding track.

    Args:
        track: List of (slice_index, Finding) tuples.

    Returns:
        Dict with track metadata: slice range, span, mean centroid, etc.
    """
    if not track:
        return {}

    slice_indices = [t[0] for t in track]
    centroids = [_get_centroid(t[1]) for t in track if _get_centroid(t[1]) is not None]

    summary = {
        "slice_start": min(slice_indices),
        "slice_end": max(slice_indices),
        "slice_span": max(slice_indices) - min(slice_indices) + 1,
        "num_findings": len(track),
        "persistent": len(track) >= 3,
    }

    if centroids:
        mean_x = sum(c[0] for c in centroids) / len(centroids)
        mean_y = sum(c[1] for c in centroids) / len(centroids)
        summary["mean_centroid"] = [round(mean_x, 1), round(mean_y, 1)]

        # Compute total centroid drift
        total_drift = sum(
            _centroid_distance(centroids[i], centroids[i + 1])
            for i in range(len(centroids) - 1)
        )
        summary["total_drift_px"] = round(total_drift, 1)

    return summary


def _get_centroid(finding) -> List[float] | None:
    """
    Extract centroid from a Finding object (supports both attribute
    and dict access for flexibility).
    """
    if hasattr(finding, "centroid"):
        return finding.centroid
    if isinstance(finding, dict):
        return finding.get("centroid")
    return None
