"""
triage_screen.py — Universal CT Slice Triage Screening

Phase 1 is a triage engine, not a diagnostic engine. It identifies statistically 
unusual density patterns while suppressing expected anatomy. It minimizes false positives 
on normal CT studies and remains independent of organ segmentation, disease classification, 
and deep learning. All detections represent candidate regions requiring further analysis 
in later phases rather than definitive pathological findings.
"""

import time
import logging
import math
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass, field, asdict

try:
    from scipy import ndimage
except ImportError:
    raise ImportError("scipy is required: pip install scipy")

logger = logging.getLogger(__name__)


@dataclass
class Finding:
    bbox: List[int]
    centroid: List[float]
    area: int
    mean_hu: float
    confidence: float
    severity_score: float
    
    # Internal fields for debugging and JSON compatibility if needed
    anomaly_type: str = "Statistical Outlier"

    @property
    def hu_value(self) -> float:
        return self.mean_hu

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": self.bbox,
            "centroid": self.centroid,
            "area": self.area,
            "mean_hu": self.mean_hu,
            "confidence": self.confidence,
            "severity_score": self.severity_score,
            "finding_type": self.anomaly_type # preserved for frontend backward compatibility
        }


@dataclass
class TriageResult:
    slice_number: int
    body_region: str
    scan_type: str
    statistics: Dict[str, float]
    findings: List[Finding]
    emergency_score: int
    action: str
    
    # Internal fields for logging and backward compatibility
    processing_time_ms: float = 0.0
    num_raw_components: int = 0
    num_filtered_components: int = 0
    failure_reason: str = ""

    @property
    def flagged(self) -> bool:
        return self.action != "CONTINUE"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slice_number": self.slice_number,
            "body_region": self.body_region,
            "scan_type": self.scan_type,
            "statistics": self.statistics,
            "findings": [f.to_dict() for f in self.findings],
            "emergency_score": self.emergency_score,
            "action": self.action,
            
            # Additional fields to keep ws_server / frontend from crashing 
            # if they expect older fields from the previous protocol.
            "slice_id": self.slice_number,
            "flagged": self.action != "CONTINUE",
            "warning_type": self.findings[0].anomaly_type.upper() if self.findings else "NONE",
            "processing_time_ms": self.processing_time_ms,
            "failure_reason": self.failure_reason,
            "stats": self.statistics,
            "slice_hu_mean": self.statistics.get("mean", 0.0),
            "slice_hu_std": self.statistics.get("std", 0.0),
            "all_findings": [f.to_dict() for f in self.findings],
            "score_details": [{"points": self.emergency_score, "description": "Dynamic severity mapping"}],
            "emergency_action": self.action,
        }


def _step_0_quality_assessment(hu_array: np.ndarray) -> Tuple[bool, str]:
    """Assess slice quality."""
    if hu_array.ndim != 2:
        return False, "Invalid slice shape"
    if np.min(hu_array) < -2000 or np.max(hu_array) > 5000:
        return False, "Corrupted HU values"
        
    body_pixels = np.sum(hu_array > -1000)
    if body_pixels < 1000:
        return False, "Insufficient body pixels (empty slice)"
        
    return True, ""


def _step_1_body_mask(hu_array: np.ndarray) -> np.ndarray:
    """Robust body mask generation."""
    mask = hu_array > -600.0
    if not np.any(mask):
        return mask
        
    structure = ndimage.generate_binary_structure(2, 1)
    mask = ndimage.binary_opening(mask, structure=structure, iterations=1)
    mask = ndimage.binary_closing(mask, structure=structure, iterations=2)
    filled = ndimage.binary_fill_holes(mask)

    labeled, n_features = ndimage.label(filled, structure=structure)
    if n_features > 1:
        sizes = ndimage.sum(filled, labeled, range(1, n_features + 1))
        if isinstance(sizes, (int, float)):
            sizes = [sizes]
        largest = np.argmax(sizes) + 1
        filled = (labeled == largest)
        
    # Grow 15 pixels using distance transform to include lung air and subcutaneous fat
    dist = ndimage.distance_transform_edt(~filled)
    filled = dist <= 15
    return filled


def _step_2A_scan_type(hu_array: np.ndarray, body_mask: np.ndarray) -> str:
    """Characterize scan type using soft tissue 99th percentile."""
    soft_tissue = hu_array[body_mask & (hu_array > -50) & (hu_array < 450)][::4]
    if soft_tissue.size == 0:
        return "non-contrast"
    p99 = float(np.percentile(soft_tissue, 99))
    if p99 >= 140.0:
        return "contrast"
    return "non-contrast"


def _step_2B_body_region(hu_array: np.ndarray, body_mask: np.ndarray, dicom_hint: str = "") -> str:
    """Identify anatomical body region."""
    hint = dicom_hint.upper().strip()
    if hint:
        if any(k in hint for k in ["HEAD", "BRAIN", "SKULL"]): return "head"
        if any(k in hint for k in ["LUNG", "CHEST", "THORAX"]): return "chest"
        if any(k in hint for k in ["ABDOMEN", "ABD", "PANCREAS", "LIVER"]): return "abdomen"
        if any(k in hint for k in ["PELVIS"]): return "pelvis"
        if any(k in hint for k in ["SPINE"]): return "spine"
        if any(k in hint for k in ["EXTREMITY", "KNEE", "HIP"]): return "extremity"

    body_pixels = hu_array[body_mask & (hu_array > -1024) & (hu_array < 3000)][::4]
    if body_pixels.size == 0:
        return "unknown"

    mean_hu = float(np.mean(body_pixels))
    std_hu = float(np.std(body_pixels))
    
    lung_fraction = np.sum((body_pixels > -1000) & (body_pixels < -300)) / body_pixels.size
    soft_tissue_fraction = np.sum((body_pixels > -100) & (body_pixels < 100)) / body_pixels.size
    bone_fraction = np.sum(body_pixels > 300) / body_pixels.size

    if lung_fraction > 0.25:
        return "chest"
    if std_hu < 150 and mean_hu > -50 and mean_hu < 60 and bone_fraction < 0.05:
        return "head"
    if bone_fraction > 0.15 and soft_tissue_fraction < 0.4:
        return "extremity"
    if soft_tissue_fraction > 0.25:
        return "abdomen"

    return "unknown"


def _step_3_anatomy_suppression(hu_array: np.ndarray, body_mask: np.ndarray, region: str) -> np.ndarray:
    """Expected Normal Anatomy Suppression."""
    search_mask = body_mask.copy()
    
    # Pre-calculate distance transform once
    dist = ndimage.distance_transform_edt(body_mask)
    
    body_pixels = hu_array[body_mask][::4]
    if body_pixels.size > 0:
        p99 = float(np.percentile(body_pixels, 99))
    else:
        p99 = 350.0
    bone_threshold = max(p99, 350.0)
    
    bone_candidates = (hu_array > bone_threshold) & body_mask
    
    if region == "head":
        search_mask = search_mask & (dist > 5)
        search_mask = search_mask & ~bone_candidates
    elif region == "chest":
        search_mask = search_mask & (dist > 3)
        search_mask = search_mask & ~bone_candidates
        
        # Chest should first detect largest bilateral air regions (lungs) and ignore them.
        lung_candidates = (hu_array < -700) & body_mask
        struct = ndimage.generate_binary_structure(2, 2)
        l_labels, l_features = ndimage.label(lung_candidates, structure=struct)
        if l_features > 0:
            l_sizes = np.bincount(l_labels.ravel())
            valid_lungs = np.where(l_sizes[1:] > 5000)[0] + 1
            lung_mask = np.isin(l_labels, valid_lungs)
            search_mask = search_mask & ~lung_mask
    elif region == "abdomen":
        subcutaneous_fat = (dist < 15) & (hu_array < -30)
        search_mask = search_mask & ~subcutaneous_fat
        search_mask = search_mask & ~bone_candidates
    else:
        search_mask = search_mask & (dist > 5)
        search_mask = search_mask & ~bone_candidates

    return search_mask


def _step_4_adaptive_baseline(hu_array: np.ndarray, search_mask: np.ndarray) -> Dict[str, float]:
    """Compute full suite of statistics on the adaptive search mask."""
    # Subsample pixels for speed (percentiles are slow on large arrays)
    pixels = hu_array[search_mask][::4]
    if pixels.size == 0:
        return {
            "mean": 0.0, "std": 1.0, "median": 0.0, "iqr": 0.0, "mad": 1.0,
            "p01": 0.0, "p05": 0.0, "p10": 0.0, "p25": 0.0, 
            "p75": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0
        }
    
    p01, p05, p10, p25, p50, p75, p90, p95, p99 = np.percentile(
        pixels, [1, 5, 10, 25, 50, 75, 90, 95, 99]
    )
    
    mean_val = np.mean(pixels)
    std_val = np.std(pixels)
    mad_val = np.median(np.abs(pixels - p50))
    iqr_val = p75 - p25

    return {
        "mean": float(mean_val),
        "std": float(std_val) if std_val > 0 else 1.0,
        "median": float(p50),
        "iqr": float(iqr_val),
        "mad": float(mad_val) if mad_val > 0 else 1.0,
        "p01": float(p01),
        "p05": float(p05),
        "p10": float(p10),
        "p25": float(p25),
        "p75": float(p75),
        "p90": float(p90),
        "p95": float(p95),
        "p99": float(p99)
    }


def _step_5_candidate_pixels_and_cca(hu_array: np.ndarray, search_mask: np.ndarray, stats: Dict[str, float]) -> Tuple[List[Dict[str, Any]], int, np.ndarray]:
    """Candidate Pixels and Connected Component Analysis."""
    mean_val = stats["mean"]
    std_val = stats["std"]
    
    z_scores = (hu_array - mean_val) / std_val
    
    # Loose criteria to capture candidate pixels for component analysis
    outliers = (np.abs(z_scores) > 2.5) | (hu_array > stats["p95"]) | (hu_array < stats["p05"])
    outliers = outliers & search_mask
    
    structure = ndimage.generate_binary_structure(2, 2)
    labeled, n_features = ndimage.label(outliers, structure=structure)
    
    raw_components = []
    if n_features == 0:
        return raw_components, 0, labeled
        
    sizes = np.bincount(labeled.ravel())
    slices = ndimage.find_objects(labeled)
    
    for i in range(1, n_features + 1):
        area = int(sizes[i])
        if area < 15: continue # ignore tiny noise
        sl = slices[i - 1]
        if sl is None: continue
        
        y_slice, x_slice = sl
        bbox = [int(x_slice.start), int(y_slice.start), int(x_slice.stop - x_slice.start), int(y_slice.stop - y_slice.start)]
        comp_mask = (labeled == i)
        pixels = hu_array[comp_mask]
        
        centroid = ndimage.center_of_mass(comp_mask)
        raw_components.append({
            "label": i, "area": area, "bbox": bbox, 
            "centroid": [round(centroid[0], 1), round(centroid[1], 1)],
            "mean_hu": float(np.mean(pixels)), "min_hu": float(np.min(pixels)),
            "max_hu": float(np.max(pixels)), "std_hu": float(np.std(pixels)),
            "mask": comp_mask
        })
        
    return raw_components, len(raw_components), labeled


def _step_6_weighted_evidence_and_local_stats(components: List[Dict[str, Any]], hu_array: np.ndarray, global_stats: Dict[str, float]) -> List[Dict[str, Any]]:
    """Weighted Evidence Evaluation on Components including Adaptive Local Neighborhood."""
    accepted = []
    for comp in components:
        comp_mean = comp["mean_hu"]
        
        # Global Evidence
        g_z = abs(comp_mean - global_stats["mean"]) / global_stats["std"]
        g_mad = abs(comp_mean - global_stats["median"]) / global_stats["mad"]
        
        # Adaptive Local Neighborhood
        x, y, w, h = comp["bbox"]
        pad = max(16, min(w, h))
        y_start = max(0, y - pad)
        y_end = min(hu_array.shape[0], y + h + pad)
        x_start = max(0, x - pad)
        x_end = min(hu_array.shape[1], x + w + pad)
        
        local_region = hu_array[y_start:y_end, x_start:x_end]
        if local_region.size > 0:
            l_mean = float(np.mean(local_region))
            l_std = float(np.std(local_region))
            if l_std == 0: l_std = 1.0
            l_z = abs(comp_mean - l_mean) / l_std
        else:
            l_mean, l_std, l_z = global_stats["mean"], global_stats["std"], g_z
            
        # Weighted Evidence Voting Model (Configurable weights conceptually)
        # Weights: Global Z (3), Local Z (3), MAD (2), Percentile (2), Contrast (2)
        evidence_score = 0
        if g_z > 3.0: evidence_score += 3
        if l_z > 3.0: evidence_score += 3
        if g_mad > 4.0: evidence_score += 2
        
        if comp_mean > global_stats["p99"] or comp_mean < global_stats["p01"]:
            evidence_score += 2
            
        local_contrast = abs(comp_mean - l_mean)
        if local_contrast > 100: evidence_score += 2
        
        # Threshold to proceed
        if evidence_score >= 6:
            comp["evidence_score"] = evidence_score
            comp["local_z"] = l_z
            accepted.append(comp)
            
    return accepted


def _step_7_confidence_and_anatomy(
    components: List[Dict[str, Any]], 
    stats: Dict[str, float],
    region: str,
    history_tracker: Any = None
) -> List[Finding]:
    """Filter components and assign confidence considering expected anatomy and temporal persistence."""
    findings = []
    for comp in components:
        area = comp["area"]
        w, h = comp["bbox"][2], comp["bbox"][3]
        mean_hu = comp["mean_hu"]
        
        # Base confidence
        conf = 0.8
        
        # Anatomy Suppression via Confidence Reduction
        aspect = max(w/h, h/w) if min(w, h) > 0 else 10.0
        solidity = area / float(w * h) if w * h > 0 else 0.0
        
        # Vessel penalty (tubular, 50 to 250 HU)
        if aspect > 4.0 and 50 < mean_hu < 250:
            conf -= 0.4
            
        # Normal internal air penalty (bowel/stomach/etc)
        if mean_hu < -400:
            if solidity > 0.5 and area < 2000:
                conf -= 0.3  # likely isolated normal bowel gas
            if region == "abdomen" and area > 1000 and mean_hu > -900:
                conf -= 0.2  # likely stomach/colon gas
        
        # Bone edge penalty
        if mean_hu > 300 and solidity < 0.2:
            conf -= 0.3
            
        # Temporal persistence confidence reduction
        if history_tracker:
            temp_f = Finding(bbox=comp["bbox"], centroid=comp["centroid"], area=area, mean_hu=mean_hu, confidence=conf, severity_score=0.0)
            pers = history_tracker.calculate_persistence(temp_f)
            if pers >= 2:
                # Reduce confidence progressively based on persistence length
                conf -= min(0.6, pers * 0.05)
                
        # Statistical boost for high evidence
        if comp.get("evidence_score", 0) >= 10:
            conf += 0.2
            
        conf = max(0.0, min(1.0, conf))
        if conf < 0.3:
            continue # ignore low confidence findings
            
        # Severity mapping
        severity = min(1.0, (comp.get("evidence_score", 6) / 12.0) * 0.5 + (area / 2000.0) * 0.5)
        
        anomaly_type = "Statistical Hyperdense" if mean_hu > stats["mean"] else "Statistical Hypodense"
        if mean_hu < -800: anomaly_type = "Extreme Air"
        
        f = Finding(
            bbox=comp["bbox"], centroid=comp["centroid"], area=area,
            mean_hu=round(mean_hu, 2), confidence=round(conf, 2),
            severity_score=round(severity, 2), anomaly_type=anomaly_type
        )
        findings.append(f)
        
    return findings


def _step_9_10_score_and_action(findings: List[Finding]) -> Tuple[int, str]:
    """Calculate emergency score and action."""
    score = 0.0
    for f in findings:
        pts = f.severity_score * f.confidence * (f.area / 100.0) * 10.0
        pts = min(pts, 60.0)
        score += pts
        
    score = int(min(max(score, 0), 100))
    
    if score >= 80: action = "IMMEDIATE ALERT"
    elif score >= 50: action = "URGENT REVIEW"
    elif score >= 20: action = "WATCH"
    else: action = "CONTINUE"
    
    return score, action


def screen_slice(
    hu_array: np.ndarray,
    slice_id: int = 0,
    body_part_hint: str = "",
    history_tracker: Any = None
) -> TriageResult:
    """Universal CT Slice Triage Screening"""
    t_start = time.perf_counter()

    # Step 0: Quality Assessment
    is_valid, fail_reason = _step_0_quality_assessment(hu_array)
    if not is_valid:
        elapsed = (time.perf_counter() - t_start) * 1000
        return TriageResult(
            slice_number=slice_id, body_region="unknown", scan_type="unknown",
            statistics={}, findings=[], emergency_score=0, action="CONTINUE",
            processing_time_ms=round(elapsed, 2), failure_reason=fail_reason
        )

    # Step 1: Body Mask
    body_mask = _step_1_body_mask(hu_array)
    
    # Step 2A & 2B: Scan Type & Region
    scan_type = _step_2A_scan_type(hu_array, body_mask)
    region = _step_2B_body_region(hu_array, body_mask, body_part_hint)
    
    # Step 3: Anatomy Suppression
    search_mask = _step_3_anatomy_suppression(hu_array, body_mask, region)
    
    # Step 4: Adaptive Baseline
    stats = _step_4_adaptive_baseline(hu_array, search_mask)
    
    # Step 5: Candidate Pixels & CCA
    raw_comps, num_raw, _ = _step_5_candidate_pixels_and_cca(hu_array, search_mask, stats)
    
    # Step 6: Component Statistics & Weighted Evidence
    accepted_comps = _step_6_weighted_evidence_and_local_stats(raw_comps, hu_array, stats)
    num_filtered = len(accepted_comps)
    
    # Step 7 & 8: Anatomy/Confidence filtering & History tracking
    findings = _step_7_confidence_and_anatomy(accepted_comps, stats, region, history_tracker)
    
    # Track findings
    if history_tracker:
        history_tracker.add_slice_findings(slice_id, findings)
    
    # Step 9 & 10: Score and Action
    score, action = _step_9_10_score_and_action(findings)
    
    elapsed = (time.perf_counter() - t_start) * 1000
    
    log_msg = (
        f"Slice {slice_id} | {region.title()} | {scan_type.title()} | "
        f"Mean: {stats['mean']:.1f} | Std: {stats['std']:.1f} | "
        f"Comps (Raw/Filt): {num_raw}/{num_filtered} | "
        f"Findings: {len(findings)} | Score: {score} | {action} | {elapsed:.1f}ms"
    )
    logger.info(log_msg)

    return TriageResult(
        slice_number=slice_id,
        body_region=region,
        scan_type=scan_type,
        statistics={k: round(v, 2) for k, v in stats.items()},
        findings=findings,
        emergency_score=score,
        action=action,
        processing_time_ms=round(elapsed, 2),
        num_raw_components=num_raw,
        num_filtered_components=num_filtered
    )
