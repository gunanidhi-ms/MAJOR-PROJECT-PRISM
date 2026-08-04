"""
candidate.py — Shared Candidate Dataclass for Phase 2

This is the single object shape that Packages 4, 5, 6, and 7 all pass around.
Freezing this contract before any implementation prevents mismatched object
shapes between packages — the single most common integration failure in a
sprint like this.

Usage:
    from schemas.candidate import Candidate, ShapeFeatures, DensityHU
"""

import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class ShapeFeatures:
    """3D shape descriptors computed in real physical units (mm/cc), never voxel counts."""
    volume_cc: float = 0.0
    long_axis_mm: float = 0.0
    short_axis_mm: float = 0.0
    sphericity: float = 0.0          # 0.0 (elongated) to 1.0 (perfect sphere)
    elongation: float = 0.0          # long_axis / short_axis ratio
    margin_curvature_variance: float = 0.0  # high = spiculated/irregular margin

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class DensityHU:
    """HU density statistics within the candidate region."""
    mean: float = 0.0
    min: float = 0.0
    max: float = 0.0
    std: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class Candidate:
    """
    Shared candidate object flowing through Packages 4 → 5 → 6 → 7.

    Lifecycle:
      - Created by Package 4 (cluster_3d) from Phase 1 seeds (Path A)
        or by Package 5 (organ_sweep) independently (Path B).
      - Merged by Package 6 (merge_candidates), filtered by clinical_filter,
        re-validated by validator_3d.
      - Scored by Package 7 (scorer_3d), serialized by phase2_api.

    Fields are grouped by when they're populated:

    [CREATION]  Set at construction by Package 4 or 5.
    [MERGE]     Set during Package 6 merge step.
    [FILTER]    Set during Package 6 clinical filtering.
    [SCORE]     Set during Package 7 scoring.
    """

    # ── [CREATION] Identity & source ──────────────────────────────────────────
    candidate_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    organ_label: str = "unclassified"       # TotalSegmentator label or "unclassified"
    detected_by: List[str] = field(default_factory=lambda: ["phase1_seed"])
    # One of: ["phase1_seed"], ["organ_sweep"], ["phase1_seed", "organ_sweep"]

    # ── [CREATION] 3D spatial location ────────────────────────────────────────
    bbox_3d: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0])
    # [x_min, y_min, z_min, x_max, y_max, z_max] in voxel coordinates
    centroid_3d: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # [x, y, z] in voxel coordinates
    slice_indices: List[int] = field(default_factory=list)
    # Which slices this candidate spans (z-axis indices)

    # ── [CREATION] Physical shape features ────────────────────────────────────
    shape: ShapeFeatures = field(default_factory=ShapeFeatures)

    # ── [CREATION] HU density statistics ──────────────────────────────────────
    density_hu: DensityHU = field(default_factory=DensityHU)

    # ── [CREATION] Phase 1 provenance (Path A only) ───────────────────────────
    phase1_confidence: float = 0.0
    phase1_severity: float = 0.0
    phase1_anomaly_type: str = ""

    # ── [MERGE] Corroboration flag ────────────────────────────────────────────
    corroborated: bool = False
    # True if found by both Path A (Phase 1 seeds) and Path B (organ sweep)

    # ── [FILTER] Organ-local statistics ───────────────────────────────────────
    organ_local_zscore: float = 0.0
    organ_overlap_fraction: float = 0.0  # How much of this candidate overlaps its organ mask

    # ── [FILTER] Persistence (from finding_tracker) ───────────────────────────
    persistence_ok: bool = False
    # True if the finding appears in >= min_slices consecutive slices

    # ── [FILTER] Clinical suppression ─────────────────────────────────────────
    suppressed: bool = False
    suppression_reason: str = ""
    # If suppressed, which rule rejected it (e.g., "lung_nodule_<6mm",
    # "simple_renal_cyst", "population_common")

    # ── [SCORE] Final scoring ─────────────────────────────────────────────────
    fused_confidence: float = 0.0
    gate: str = ""  # "AUTO_CONFIRMED" or "MANUAL_REVIEW"
    emergency_score: int = 0

    # ── [FILTER] Texture features (from TotalSegmentator --radiomics) ─────────
    texture_features: Dict[str, float] = field(default_factory=dict)
    # e.g., {"glcm_contrast": ..., "glcm_homogeneity": ..., "texture_cv": ...}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to the findings_output.json schema shape for Phase 3 handoff."""
        return {
            "candidate_id": self.candidate_id,
            "organ_label": self.organ_label,
            "detected_by": self.detected_by,
            "corroborated": self.corroborated,
            "shape": self.shape.to_dict(),
            "density_hu": self.density_hu.to_dict(),
            "organ_local_zscore": round(self.organ_local_zscore, 4),
            "fused_confidence": round(self.fused_confidence, 4),
            "gate": self.gate,
            "emergency_score": self.emergency_score,
        }

    def to_internal_dict(self) -> Dict[str, Any]:
        """Full serialization including internal fields — for delta-report logging."""
        d = self.to_dict()
        d.update({
            "bbox_3d": self.bbox_3d,
            "centroid_3d": self.centroid_3d,
            "slice_indices": self.slice_indices,
            "phase1_confidence": self.phase1_confidence,
            "phase1_severity": self.phase1_severity,
            "phase1_anomaly_type": self.phase1_anomaly_type,
            "organ_overlap_fraction": self.organ_overlap_fraction,
            "persistence_ok": self.persistence_ok,
            "suppressed": self.suppressed,
            "suppression_reason": self.suppression_reason,
            "texture_features": self.texture_features,
        })
        return d
