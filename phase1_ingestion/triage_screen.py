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

from phase1_ingestion.triage_config import TRIAGE_CONFIG

logger = logging.getLogger(__name__)


# ─── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class Finding:
    bbox: List[int]
    centroid: List[float]
    area: int
    mean_hu: float
    confidence: float
    severity_score: float
    
    anomaly_type: str = "Statistical Outlier"
    min_hu: float = 0.0
    max_hu: float = 0.0

    @property
    def hu_value(self) -> float:
        return self.mean_hu

    @property
    def finding_type(self) -> str:
        return self.anomaly_type

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": self.bbox,
            "centroid": self.centroid,
            "area": self.area,
            "mean_hu": self.mean_hu,
            "hu_mean": self.mean_hu,
            "hu_max": self.max_hu,
            "hu_min": self.min_hu,
            "confidence": self.confidence,
            "severity_score": self.severity_score,
            "finding_type": self.anomaly_type,
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
    
    # Borderline findings (confidence 0.10–0.30):
    # - NOT shown on the frontend (ws_server.py serialises findings only)
    # - ARE passed into VolumeAccumulator so they reach finding_tracker →
    #   seed_clusterer → Path A for corroboration against Path B candidates.
    borderline_findings: List[Finding] = field(default_factory=list)

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
            # Backward-compatible fields
            "slice_id": self.slice_number,
            "flagged": self.action != "CONTINUE",
            "warning_type": self.findings[0].anomaly_type.upper() if self.findings else "NONE",
            "processing_time_ms": self.processing_time_ms,
            "failure_reason": self.failure_reason,
            "stats": self.statistics,
            "slice_hu_mean": self.statistics.get("mean", 0.0),
            "slice_hu_std": self.statistics.get("std", 0.0),
            "displayed_findings": [f.to_dict() for f in self.findings],
            "all_findings": [f.to_dict() for f in self.findings],  # deprecated alias
            "score_details": [{"points": self.emergency_score, "description": "Dynamic severity mapping"}],
            "emergency_action": self.action,
        }


# ─── Step 0: Slice Quality Assessment ────────────────────────────────────────

def _step_0_quality_assessment(hu_array: np.ndarray) -> Tuple[bool, str]:
    """Reject corrupted, empty, or non-2D slices before any processing."""
    if hu_array.ndim != 2:
        return False, "Invalid slice shape"
    if hu_array.shape[0] < 64 or hu_array.shape[1] < 64:
        return False, "Slice too small"
    arr_min, arr_max = float(np.min(hu_array)), float(np.max(hu_array))
    if arr_min < -2000 or arr_max > 5000:
        return False, "Corrupted HU values"
    body_pixels = np.sum(hu_array > -500)
    if body_pixels < 1000:
        return False, "Insufficient body pixels (empty slice)"
    return True, ""


# ─── Step 1: Body Mask Generation ────────────────────────────────────────────

def _step_1_body_mask(hu_array: np.ndarray) -> np.ndarray:

    mask = hu_array > -600.0
    if not np.any(mask):
        return mask
        
    structure = ndimage.generate_binary_structure(2, 1)
    mask = ndimage.binary_opening(mask, structure=structure, iterations=1)
    mask = ndimage.binary_closing(mask, structure=structure, iterations=2)
    filled = ndimage.binary_fill_holes(mask)

    labeled, n_features = ndimage.label(filled, structure=structure)
    if n_features > 1:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0  # ignore background
        largest = np.argmax(sizes)
        filled = (labeled == largest)
        
    # Grow outward to include subcutaneous fat and skin border.
    # Background air components that leak in here are caught by the
    # image-boundary bbox rejection in Step 7, not here.
    dist = ndimage.distance_transform_edt(~filled)
    filled = dist <= 15
    return filled


# ─── Step 2A: Scan Type Characterization ─────────────────────────────────────

def _step_2A_scan_type(hu_array: np.ndarray, body_mask: np.ndarray) -> str:
    """Characterize scan type using soft tissue 99th percentile."""
    soft_tissue = hu_array[body_mask & (hu_array > -50) & (hu_array < 450)]
    if soft_tissue.size < 100:
        return "non-contrast"
    p99 = float(np.percentile(soft_tissue[::4], 99))
    if p99 >= 140.0:
        return "contrast"
    return "non-contrast"


# ─── Step 2B: Body Region Identification ─────────────────────────────────────

def _step_2B_body_region(hu_array: np.ndarray, body_mask: np.ndarray, dicom_hint: str = "") -> str:
    """
    Identify anatomical body region.

    Priority 1 — DICOM hint (BodyPartExamined / SeriesDescription):
      If the hint resolves to a known region, return immediately and skip the
      histogram heuristic entirely.  DICOM metadata is structurally reliable;
      the histogram fallback is not.

    Priority 2 — HU histogram heuristics:
      Used only when no DICOM hint is available or the hint is unrecognised.
      These thresholds are deliberately kept but logged so the region_agreement
      confusion matrix can drive empirical recalibration over time.
    """
    hint = dicom_hint.upper().strip()
    if hint:
        resolved = _match_dicom_hint(hint)
        if resolved != "unknown":
            logger.debug(
                "_step_2B: DICOM hint '%s' resolved to '%s' — histogram skipped",
                dicom_hint, resolved,
            )
            return resolved
        logger.debug(
            "_step_2B: DICOM hint present ('%s') but unresolved — falling back to histogram",
            dicom_hint,
        )
    else:
        logger.debug("_step_2B: no DICOM hint — histogram heuristic will run")

    # ── Histogram fallback ────────────────────────────────────────────────────
    body_pixels = hu_array[body_mask & (hu_array > -1024) & (hu_array < 3000)]
    if body_pixels.size < 100:
        return "unknown"
    sample = body_pixels[::4]

    mean_hu = float(np.mean(sample))
    std_hu = float(np.std(sample))
    
    lung_fraction = float(np.sum((sample > -1000) & (sample < -300))) / sample.size
    soft_tissue_fraction = float(np.sum((sample > -100) & (sample < 100))) / sample.size
    bone_fraction = float(np.sum(sample > 300)) / sample.size
    air_fraction = float(np.sum(sample < -500)) / sample.size
    
    body_area = float(np.sum(body_mask))
    image_area = float(body_mask.shape[0] * body_mask.shape[1])
    body_fill = body_area / image_area

    if lung_fraction > 0.25:
        return "chest"
    if std_hu < 150 and mean_hu > -50 and mean_hu < 60 and bone_fraction < 0.05:
        return "head"
    if body_fill < 0.35 and air_fraction > 0.01 and air_fraction < 0.15 and bone_fraction < 0.10:
        return "neck"
    if bone_fraction > 0.15 and soft_tissue_fraction < 0.4:
        return "extremity"
    if soft_tissue_fraction > 0.25:
        return "abdomen"

    return "unknown"


def _match_dicom_hint(hint: str) -> str:
    """
    Map an upper-cased DICOM hint string to a known body region.
    Returns 'unknown' if no match found.
    """
    if any(k in hint for k in ["HEAD", "BRAIN", "SKULL", "CRANIAL", "CEREBR"]):
        return "head"
    if any(k in hint for k in ["NECK", "CERVICAL", "C-SPINE", "CERVIC"]):
        return "neck"
    if any(k in hint for k in ["LUNG", "CHEST", "THORAX", "THORAC", "CARDIAC", "HEART"]):
        return "chest"
    if any(k in hint for k in ["ABDOMEN", "ABD", "PANCREAS", "LIVER", "HEPAT", "RENAL", "KIDNEY"]):
        return "abdomen"
    if any(k in hint for k in ["PELVIS", "PELVIC", "BLADDER", "PROSTATE"]):
        return "pelvis"
    if any(k in hint for k in ["SPINE", "LUMBAR", "THORACIC SPINE", "L-SPINE", "T-SPINE"]):
        return "spine"
    if any(k in hint for k in ["EXTREMITY", "KNEE", "HIP", "SHOULDER", "FEMUR", "HUMER"]):
        return "extremity"
    return "unknown"


# ─── Step 3: Expected Normal Anatomy Suppression ─────────────────────────────

def _compactness(comp_mask: np.ndarray) -> float:
    """
    Compute the solidity (compactness) of a binary 2D component.

    Returns area / bounding_box_area.  Compact blobs (bowel gas, discrete
    air pockets) score close to 1.0; sheet-like or irregular free air scores
    much lower.
    """
    rows = np.any(comp_mask, axis=1)
    cols = np.any(comp_mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return 0.0
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    bbox_area = (rmax - rmin + 1) * (cmax - cmin + 1)
    if bbox_area == 0:
        return 0.0
    return float(np.sum(comp_mask)) / float(bbox_area)


def _suppress_air_components(
    hu_array: np.ndarray,
    body_mask: np.ndarray,
    dist: np.ndarray,
    region: str,
    config: dict,
) -> np.ndarray:
    """
    Build a suppression mask for expected air structures within the body mask.

    One parameterized function replaces the original five near-duplicate
    per-region air-handling blocks.  Region determines which config thresholds
    apply, not which code path runs.

    Returns:
        Boolean mask (same shape as body_mask) — True where air should be
        suppressed from the search mask.

    Region behaviour:
        chest  — suppress large bilateral lung fields AND small airways
                 (trachea/bronchi).  Medium air (2000-5000 px) is KEPT as
                 potential pneumothorax.
        abdomen/pelvis — suppress compact, moderate-density internal gas
                 (bowel, rectum, bladder).  Large or low-density air is KEPT
                 as potential free air (pneumoperitoneum).
        neck   — suppress ALL internal air (trachea, pharynx, larynx are
                 all expected anatomy here).
        other  — suppress all air structures (generic / unknown regions).
    """
    air_mask = (hu_array < -200) & body_mask

    # Abdomen/pelvis: restrict to internal air only (skip peripheral fat/air)
    if region in ("abdomen", "pelvis"):
        min_dist = config["abdomen_subq_fat_band_px"]
        air_mask = air_mask & (dist > min_dist)

    struct = ndimage.generate_binary_structure(2, 2)
    labeled, n = ndimage.label(air_mask, structure=struct)
    if n == 0:
        return np.zeros(body_mask.shape, dtype=bool)

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # ignore background label

    suppress = np.zeros(body_mask.shape, dtype=bool)

    for label_id in range(1, n + 1):
        sz = int(sizes[label_id])
        if sz == 0:
            continue
        comp = (labeled == label_id)
        comp_mean = float(np.mean(hu_array[comp]))

        if region == "chest":
            # Large air field with typical lung parenchyma density → suppress as normal lung
            if sz > config["large_air_component_px"] and comp_mean > config["lung_density_hu"]:
                suppress |= comp
            # Small airway (trachea, bronchi) → always suppress
            elif sz < config["small_airway_component_px"]:
                suppress |= comp
            # Large air at near-vacuum density → leaked background, NOT PTX
            elif sz > config["large_air_component_px"] and comp_mean <= config["lung_density_hu"]:
                suppress |= comp
            # Medium air (2000-5000 px): potential pneumothorax — KEEP

        elif region in ("abdomen", "pelvis"):
            if sz < 50:
                # Tiny noise pixel clusters → suppress
                suppress |= comp
                continue
            solidity = _compactness(comp)
            # Normal bowel/stomach/bladder gas: compact, moderate-sized, moderate density
            # Pathological free air: very low density, often large, irregular shape
            is_normal_gas = (
                sz < config["abdomen_gas_max_px"]
                and solidity > config["abdomen_gas_solidity_min"]
                and comp_mean > config["abdomen_gas_hu_floor"]
            )
            # Also suppress large diffuse gas collections that are likely
            # colonic haustra air or stomach gas, even if slightly below the
            # HU floor (-700): normal colon gas regularly reaches -800 to
            # -950 HU.  Only keep very-low-density components (< -950 HU)
            # for potential pneumoperitoneum evaluation.
            is_deep_colonic_gas = (
                sz < config["abdomen_gas_max_px"] * 3
                and solidity > config["abdomen_gas_solidity_min"]
                and comp_mean > -950.0
                and comp_mean <= config["abdomen_gas_hu_floor"]
            )
            if is_normal_gas or is_deep_colonic_gas:
                suppress |= comp

        else:
            # Neck (trachea/pharynx/larynx all expected), spine, extremity, unknown:
            # suppress all internal air structures as normal anatomy
            suppress |= comp

    return suppress


def _step_3_anatomy_suppression(
    hu_array: np.ndarray,
    body_mask: np.ndarray,
    region: str,
    config: dict = None,
) -> np.ndarray:
    """
    Build the Search Mask by removing expected normal anatomy.
    
    This is the single most important step for false-positive reduction.
    
    Strategy per region:
      - All regions: suppress cortical bone adaptively, erode outer boundary.
      - Chest: suppress large bilateral lung cavities AND trachea/bronchi.
      - Abdomen: suppress subcutaneous fat border AND internal gas pockets
                 that are geometrically consistent with normal bowel/stomach.
      - Head: erode skull boundary deeply, suppress bone.

    All thresholds come from TRIAGE_CONFIG (no inline magic numbers).
    Air suppression is handled by _suppress_air_components().
    """
    if config is None:
        config = TRIAGE_CONFIG

    search_mask = body_mask.copy()
    
    # Distance transform (reused across all regions and in _suppress_air_components)
    dist = ndimage.distance_transform_edt(body_mask)
    
    # ── Adaptive bone threshold ──
    # P99 captures dense cortical bone at the top of the distribution.
    # The config floor ensures contrast-enhanced vessels (~200 HU) are NOT
    # suppressed as bone, while all bone (> floor) is caught.
    body_sample = hu_array[body_mask][::8]
    if body_sample.size > 0:
        p99 = float(np.percentile(body_sample, 99))
    else:
        p99 = config["bone_threshold_floor_hu"]
    bone_threshold = max(p99, config["bone_threshold_floor_hu"])
    
    # Bone suppression: threshold + dilation to catch partial-volume edges
    bone_candidates = (hu_array >= bone_threshold) & body_mask
    bone_dilated = ndimage.binary_dilation(bone_candidates, iterations=3)
    
    # ── Per-region boundary erosion and bone suppression ──
    erosion_px = config["boundary_erosion_px"]
    if region == "head":
        search_mask = search_mask & (dist > erosion_px.get("head", 8))
        search_mask = search_mask & ~bone_dilated
        # Suppress CSF ventricles (expected anatomy, HU ~0-15) to avoid
        # hypodense outlier false positives
        csf_mask = (hu_array < 15) & (hu_array > -10) & search_mask
        search_mask = search_mask & ~csf_mask

    elif region == "chest":
        search_mask = search_mask & (dist > erosion_px.get("chest", 4))
        search_mask = search_mask & ~bone_dilated
        # Suppress normal chest air (lungs, trachea, bronchi) via unified helper
        air_suppress = _suppress_air_components(hu_array, body_mask, dist, "chest", config)
        search_mask = search_mask & ~air_suppress

    elif region == "abdomen":
        # Subcutaneous fat band: near boundary AND fat HU range
        subq_band = config["abdomen_subq_fat_band_px"]
        subcutaneous_fat = (dist < subq_band) & (hu_array < -30) & (hu_array > -200)
        search_mask = search_mask & ~subcutaneous_fat
        search_mask = search_mask & ~bone_dilated
        # Suppress normal internal gas pockets (bowel, stomach)
        air_suppress = _suppress_air_components(hu_array, body_mask, dist, "abdomen", config)
        search_mask = search_mask & ~air_suppress

    elif region == "neck":
        search_mask = search_mask & (dist > erosion_px.get("default", 5))
        search_mask = search_mask & ~bone_dilated
        # Suppress ALL internal air (trachea, pharynx, larynx are all expected)
        air_suppress = _suppress_air_components(hu_array, body_mask, dist, "neck", config)
        search_mask = search_mask & ~air_suppress

    elif region == "pelvis":
        search_mask = search_mask & (dist > erosion_px.get("default", 5))
        search_mask = search_mask & ~bone_dilated
        # Suppress compact internal gas (rectum, bladder)
        air_suppress = _suppress_air_components(hu_array, body_mask, dist, "pelvis", config)
        search_mask = search_mask & ~air_suppress

    else:
        # Spine, Extremity, Unknown — suppress bone + boundary
        search_mask = search_mask & (dist > erosion_px.get("default", 5))
        search_mask = search_mask & ~bone_dilated
        # Suppress all air in generic/unknown regions
        air_suppress = _suppress_air_components(hu_array, body_mask, dist, "other", config)
        search_mask = search_mask & ~air_suppress

    # ── Universal post-suppression cleanup ──
    # After bone removal, gaps may contain background air (-1024 HU) that would
    # contaminate the statistical baseline.  Remove these residual air pixels
    # ONLY in regions where air is NOT the detection target.
    if region not in ("chest", "abdomen"):
        search_mask = search_mask & (hu_array > -500)

    return search_mask


# ─── Step 4: Adaptive Statistical Baseline ───────────────────────────────────

def _step_4_adaptive_baseline(hu_array: np.ndarray, search_mask: np.ndarray) -> Dict[str, float]:
    """
    Compute robust statistics ONLY on the search mask pixels.
    
    Subsamples for speed. Returns a comprehensive statistical profile
    that subsequent steps use to define "normalcy" for this specific slice.
    """
    pixels = hu_array[search_mask]
    if pixels.size == 0:
        return {
            "mean": 0.0, "std": 1.0, "median": 0.0, "iqr": 0.0, "mad": 1.0,
            "p01": 0.0, "p05": 0.0, "p10": 0.0, "p25": 0.0, 
            "p75": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0,
            "p005": 0.0, "p995": 0.0
        }
    
    # Subsample for speed (percentile computation is O(n log n))
    sample = pixels[::4] if pixels.size > 10000 else pixels
    
    pcts = np.percentile(sample, [0.5, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5])
    p005, p01, p05, p10, p25, p50, p75, p90, p95, p99, p995 = pcts
    
    # ── Trimmed statistics ──
    # Compute mean and std ONLY on the P5-P95 range.
    # This prevents outliers (the very things we want to detect) from
    # inflating the standard deviation and pulling the mean, which would
    # make the outliers appear less extreme than they actually are.
    trimmed = sample[(sample >= p05) & (sample <= p95)]
    if trimmed.size > 0:
        mean_val = float(np.mean(trimmed))
        std_val = float(np.std(trimmed))
    else:
        mean_val = float(np.mean(sample))
        std_val = float(np.std(sample))
    
    mad_val = float(np.median(np.abs(sample - p50)))
    iqr_val = float(p75 - p25)

    return {
        "mean": mean_val,
        "std": max(std_val, 1.0),
        "median": float(p50),
        "iqr": max(iqr_val, 1.0),
        "mad": max(mad_val, 1.0),
        "p005": float(p005),
        "p01": float(p01),
        "p05": float(p05),
        "p10": float(p10),
        "p25": float(p25),
        "p75": float(p75),
        "p90": float(p90),
        "p95": float(p95),
        "p99": float(p99),
        "p995": float(p995),
    }


# ─── Step 5: Candidate Pixel Detection & CCA ────────────────────────────────

def _step_5_candidate_detection(hu_array: np.ndarray, search_mask: np.ndarray, stats: Dict[str, float]) -> Tuple[List[Dict[str, Any]], int]:
    """
    Identify candidate outlier pixels using AND-gated multi-metric criteria,
    then group them into connected components.
    
    Key improvement: A pixel must satisfy MULTIPLE independent statistical
    criteria simultaneously (AND logic), not just one (OR logic).
    
    This dramatically reduces false positives from normal tissue variation.
    """
    mean_val = stats["mean"]
    std_val = stats["std"]
    mad_val = stats["mad"]
    median_val = stats["median"]
    p005 = stats["p005"]
    p995 = stats["p995"]
    
    z_scores = np.abs((hu_array - mean_val) / std_val)
    mad_scores = np.abs((hu_array - median_val) / mad_val)
    
    iqr = stats["iqr"]
    p25 = stats["p25"]
    p75 = stats["p75"]
    
    # ── Multi-metric majority vote ──
    # A pixel must satisfy at least 2 out of 4 independent criteria.
    #
    # The statistics are computed on the TRIMMED distribution (P5-P95),
    # so they are immune to contamination by the outliers themselves.
    # This means Z-scores accurately reflect how extreme a pixel is
    # relative to the normal tissue, not relative to a corrupted mean.
    #
    # Criteria:
    #   1. Z-score > 4.0 (using trimmed mean/std)
    #   2. MAD-score > 5.0 (robust to skew)
    #   3. Beyond P0.5/P99.5 bounds
    #   4. IQR fence: beyond P75 + 2.5*IQR or below P25 - 2.5*IQR
    
    extreme_z = z_scores > 4.0
    extreme_mad = mad_scores > 5.0
    extreme_pct = (hu_array > p995) | (hu_array < p005)
    extreme_iqr = (hu_array > p75 + 2.5 * iqr) | (hu_array < p25 - 2.5 * iqr)
    
    # Majority vote: require 2 out of 4
    vote_count = (
        extreme_z.astype(np.int8) + 
        extreme_mad.astype(np.int8) + 
        extreme_pct.astype(np.int8) + 
        extreme_iqr.astype(np.int8)
    )
    outliers = (vote_count >= 2) & search_mask
    
    if not np.any(outliers):
        return [], 0
    
    structure = ndimage.generate_binary_structure(2, 2)
    labeled, n_features = ndimage.label(outliers, structure=structure)
    
    if n_features == 0:
        return [], 0
    
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # ignore background
    slices = ndimage.find_objects(labeled)
    
    raw_components = []
    for i in range(1, n_features + 1):
        area = int(sizes[i])
        if area < 30:  # minimum detectable cluster size
            continue
        sl = slices[i - 1]
        if sl is None:
            continue
        
        y_slice, x_slice = sl
        bbox = [
            int(x_slice.start), int(y_slice.start),
            int(x_slice.stop - x_slice.start), int(y_slice.stop - y_slice.start)
        ]
        comp_mask = (labeled == i)
        comp_pixels = hu_array[comp_mask]
        
        centroid = ndimage.center_of_mass(comp_mask)
        raw_components.append({
            "label": i, "area": area, "bbox": bbox,
            "centroid": [round(float(centroid[0]), 1), round(float(centroid[1]), 1)],
            "mean_hu": float(np.mean(comp_pixels)),
            "min_hu": float(np.min(comp_pixels)),
            "max_hu": float(np.max(comp_pixels)),
            "std_hu": float(np.std(comp_pixels)),
            "mask": comp_mask,
        })
        
    return raw_components, n_features


# ─── Step 6: Local Context Validation ────────────────────────────────────────

def _step_6_local_context_validation(
    components: List[Dict[str, Any]], 
    hu_array: np.ndarray, 
    search_mask: np.ndarray,
    global_stats: Dict[str, float]
) -> List[Dict[str, Any]]:
    """
    Validate each component against its LOCAL neighborhood within the search mask.
    
    Key improvement: The local neighborhood is sampled ONLY from the search mask,
    not from raw HU values. This prevents bone/air/fat from inflating the local
    standard deviation and making everything look "normal" relative to its neighborhood.
    """
    validated = []
    for comp in components:
        comp_mean = comp["mean_hu"]
        
        # Global statistical evidence (using trimmed stats)
        g_z = abs(comp_mean - global_stats["mean"]) / global_stats["std"]
        g_mad = abs(comp_mean - global_stats["median"]) / global_stats["mad"]
        
        # Local neighborhood: extract from SEARCH MASK, EXCLUDING the component itself
        # This prevents the anomaly from contaminating its own reference statistics.
        x, y, w, h = comp["bbox"]
        pad = max(25, max(w, h) * 2)
        y_start = max(0, y - pad)
        y_end = min(hu_array.shape[0], y + h + pad)
        x_start = max(0, x - pad)
        x_end = min(hu_array.shape[1], x + w + pad)
        
        local_search = search_mask[y_start:y_end, x_start:x_end].copy()
        local_comp = comp["mask"][y_start:y_end, x_start:x_end]
        # Remove the component itself from the local reference
        local_search = local_search & ~local_comp
        
        local_hu = hu_array[y_start:y_end, x_start:x_end]
        local_pixels = local_hu[local_search]
        
        if local_pixels.size > 50:
            l_mean = float(np.mean(local_pixels))
            l_std = float(np.std(local_pixels))
            l_std = max(l_std, 1.0)
            l_z = abs(comp_mean - l_mean) / l_std
        else:
            l_mean = global_stats["mean"]
            l_z = g_z  # fallback to global
        
        # Local contrast: absolute HU difference from surrounding tissue
        local_contrast = abs(comp_mean - l_mean)
        
        # ── Evidence voting ──
        has_global_z = g_z > 3.5
        has_local_z = l_z > 2.5
        has_mad = g_mad > 4.0
        has_pct = comp_mean > global_stats["p99"] or comp_mean < global_stats["p01"]
        has_contrast = local_contrast > 60
        
        evidence_count = sum([has_global_z, has_local_z, has_mad, has_pct, has_contrast])
        
        # Require at least 2 out of 5 independent evidences.
        # This is deliberately lenient at this stage because
        # Step 7 (geometric filtering + confidence gating) provides
        # a second, independent layer of false-positive rejection.
        if evidence_count >= 2:
            comp["evidence_count"] = evidence_count
            comp["local_z"] = round(l_z, 2)
            comp["global_z"] = round(g_z, 2)
            comp["local_contrast"] = round(local_contrast, 1)
            validated.append(comp)
            
    return validated


# ─── Step 7: Geometric Filtering & Confidence ───────────────────────────────

def _step_7_geometric_filtering_and_confidence(
    components: List[Dict[str, Any]], 
    stats: Dict[str, float],
    search_mask: np.ndarray,
    body_mask: np.ndarray,
    region: str,
    config: dict = None,
) -> Tuple[List[Finding], List[Finding]]:
    """
    Apply geometric sanity checks and assign confidence scores.

    Returns a tuple of (findings, borderline_findings):

    findings:
        Confidence >= config["confidence_gate_display"] (default 0.30).
        These are shown live on the frontend via ws_server.py.

    borderline_findings:
        Confidence in [config["confidence_gate_borderline"], config["confidence_gate_display"])
        i.e. [0.10, 0.30) by default.
        NOT shown on the frontend, but passed into VolumeAccumulator alongside
        normal findings so they reach finding_tracker → seed_clusterer → Path A.
        This gives Path A more raw material to attempt corroboration against
        Path B's independently-found candidates — the key reason for this tier.

    Anything below confidence_gate_borderline (< 0.10) is dropped as noise.

    Key improvements:
      - Border-touching components are rejected.
      - Elongated structures (aspect > 6) are penalized heavily (likely vessels/artifacts).
      - Low-solidity components (scattered noise) are rejected.
      - Confidence starts at 0.5 and must be EARNED through evidence, not given.
    """
    if config is None:
        config = TRIAGE_CONFIG

    display_gate = config["confidence_gate_display"]
    borderline_gate = config["confidence_gate_borderline"]

    findings: List[Finding] = []
    borderline_findings: List[Finding] = []
    
    for comp in components:
        area = comp["area"]
        x, y, w, h = comp["bbox"]
        mean_hu = comp["mean_hu"]
        
        # ── Geometric rejection ──
        if w == 0 or h == 0:
            continue
            
        aspect = max(w / h, h / w)
        bbox_area = w * h
        solidity = area / float(bbox_area)
        
        # Too thin/elongated → likely vessel cross-section or beam artifact
        if aspect > 6.0:
            continue
            
        # Too scattered → noise
        if solidity < 0.15:
            continue
        
        # Image-boundary rejection: components whose bounding box touches
        # the image edge are virtually always CT table, scanner gantry, or
        # background air leaking through the body mask — never true findings.
        img_h, img_w = search_mask.shape
        img_area = img_h * img_w
        if x <= 1 or y <= 1 or (x + w) >= img_w - 1 or (y + h) >= img_h - 1:
            continue

        # ── CT gantry / table arc rejection ───────────────────────────────────
        # The CT scanner gantry appears as a large arc at the top of the FOV
        # with extreme negative HU (~−900 to −1000). Three complementary rules:

        # 1. Oversized bbox: a real intra-organ finding is NEVER larger than
        #    12% of the image area. The gantry arc bbox is typically 30–80%.
        bbox_fraction = bbox_area / float(img_area)
        if bbox_fraction > 0.12:
            continue

        # 2. Body containment: at least 40% of the component's pixels must
        #    fall inside the body mask. Gantry/table pixels are outside it.
        pixels_in_body = np.sum(comp["mask"] & body_mask)
        if area > 0 and (pixels_in_body / area) < 0.40:
            continue

        # 3. Top-edge gantry arc: bboxes whose top edge is in the top 5% of
        #    the image are almost certainly scanner hardware, not tissue.
        if y < img_h * 0.05:
            continue

        # Border touching: if the component mask touches the search mask boundary
        comp_dilated = ndimage.binary_dilation(comp["mask"], iterations=1)
        border_contact = np.sum(comp_dilated & ~search_mask)
        if border_contact > area * 0.3:
            continue

        # ── Outer-body dilation band check ────────────────────────────────────
        # The body mask in _step_1 is dilated 15px outward into subcutaneous
        # fat.  Findings whose CENTROID falls in this outer band are almost
        # certainly in fat or skin — not inside any organ.
        # Strategy: use the distance transform of the UNDILATED body mask (if
        # available via body_mask_eroded) or proxy via dist_from_body_center.
        # We use the simpler proxy here: if the component centroid is very
        # close to the body mask boundary (dist <= outer_band_px), cap confidence.
        cy, cx = comp["centroid"]
        outer_band_px = config.get("outer_body_band_px", 18) if config else 18
        centroid_in_band = False
        try:
            cy_int = int(round(cy))
            cx_int = int(round(cx))
            if 0 <= cy_int < body_mask.shape[0] and 0 <= cx_int < body_mask.shape[1]:
                # Compute distance from centroid to body mask edge using a local crop
                local_pad = outer_band_px + 5
                y0 = max(0, cy_int - local_pad)
                y1_loc = min(body_mask.shape[0], cy_int + local_pad + 1)
                x0 = max(0, cx_int - local_pad)
                x1_loc = min(body_mask.shape[1], cx_int + local_pad + 1)
                local_mask = body_mask[y0:y1_loc, x0:x1_loc]
                local_dist = ndimage.distance_transform_edt(local_mask)
                local_cy = cy_int - y0
                local_cx = cx_int - x0
                if 0 <= local_cy < local_dist.shape[0] and 0 <= local_cx < local_dist.shape[1]:
                    dist_to_edge = local_dist[local_cy, local_cx]
                    centroid_in_band = dist_to_edge <= outer_band_px
        except Exception:
            pass  # distance check is best-effort — never fail hard

        # ── Confidence calculation (start conservative, earn confidence) ──
        conf = 0.4  # conservative base

        # Findings in the outer fat/skin band get confidence capped at the
        # borderline gate so they never reach the display tier at Phase 1.
        # The organ postfilter in Phase 2 will give a definitive verdict.
        if centroid_in_band:
            outer_cap = (config.get("confidence_gate_borderline", 0.10) + 0.05) if config else 0.15
            conf = min(conf, outer_cap)

        evidence_count = comp.get("evidence_count", 3)
        global_z = comp.get("global_z", 3.5)
        local_contrast = comp.get("local_contrast", 80)
        
        # Statistical strength
        if global_z > 6.0:
            conf += 0.25
        elif global_z > 4.5:
            conf += 0.15
        elif global_z > 3.5:
            conf += 0.05
            
        # Evidence consensus
        if evidence_count >= 5:
            conf += 0.15
        elif evidence_count >= 4:
            conf += 0.10
            
        # Size: larger findings are more clinically significant
        if area > 1000:
            conf += 0.15
        elif area > 300:
            conf += 0.10
        elif area > 100:
            conf += 0.05
            
        # Compactness: compact masses are more concerning than scattered ones
        if solidity > 0.6:
            conf += 0.05
        
        # Local contrast: high contrast against surroundings is very suspicious
        if local_contrast > 200:
            conf += 0.10
        elif local_contrast > 100:
            conf += 0.05
        
        # ── Confidence penalties (universal, not disease-specific) ──
        
        # Elongated structures get penalized (likely vessels)
        if aspect > 4.0:
            conf -= 0.15
            
        # Very small findings are less certain
        if area < 80:
            conf -= 0.10
        
        conf = max(0.0, min(1.0, conf))
        
        # ── Severity score ──
        # Based on statistical extremity and physical size
        z_contribution = min(1.0, global_z / 10.0) * 0.6
        size_contribution = min(1.0, area / 3000.0) * 0.4
        severity = z_contribution + size_contribution
        severity = max(0.0, min(1.0, severity))
        
        # ── Anomaly type labeling ──
        if mean_hu < -800:
            anomaly_type = "Extreme Air"
        elif mean_hu > stats["mean"]:
            anomaly_type = "Statistical Hyperdense"
        else:
            anomaly_type = "Statistical Hypodense"
        
        f = Finding(
            bbox=comp["bbox"],
            centroid=comp["centroid"],
            area=area,
            mean_hu=round(mean_hu, 2),
            min_hu=round(comp.get("min_hu", 0.0), 2),
            max_hu=round(comp.get("max_hu", 0.0), 2),
            confidence=round(conf, 2),
            severity_score=round(severity, 2),
            anomaly_type=anomaly_type,
        )

        # ── Two-tier output ──
        # Display tier: shown live on frontend
        if conf >= display_gate:
            findings.append(f)
        # Borderline tier: feeds Path A only, never shown on frontend
        elif conf >= borderline_gate:
            borderline_findings.append(f)
        # Below borderline_gate: drop as noise (< 0.10 confidence)

    return findings, borderline_findings


# ─── Step 9/10: Emergency Score & Action ─────────────────────────────────────

def _step_9_10_score_and_action(findings: List[Finding]) -> Tuple[int, str]:
    """
    Calculate emergency score from findings.
    
    Formula per finding:
        points = severity * confidence^1.5 * sqrt(area / 500) * 40
        (capped at 50 per finding)
    
    The confidence^1.5 exponent provides a non-linear penalty for uncertain findings
    while being less punitive than ^2.0 for moderate-confidence true anomalies.
    
    Examples:
        conf=0.4 -> 0.25x (very low, noise-like)
        conf=0.65 -> 0.52x (moderate, e.g. brain hemorrhage)
        conf=0.9 -> 0.85x (high, clear anomaly)
    
    The sqrt(area/500) term means a 500px finding contributes 1.0x, but a 50px
    finding only contributes 0.32x. This prevents tiny gas pockets from scoring.
    """
    score = 0.0
    for f in findings:
        area_factor = math.sqrt(max(f.area, 1) / 500.0)
        pts = f.severity_score * (f.confidence ** 1.5) * area_factor * 40.0
        pts = min(pts, 50.0)
        score += pts
        
    score = int(min(max(score, 0), 100))
    
    if score >= 80:
        action = "IMMEDIATE ALERT"
    elif score >= 45:
        action = "URGENT REVIEW"
    elif score >= 20:
        action = "WATCH"
    else:
        action = "CONTINUE"
    
    return score, action


# ─── Main Entry Point ────────────────────────────────────────────────────────

def screen_slice(
    hu_array: np.ndarray,
    slice_id: int = 0,
    body_part_hint: str = "",
    history_tracker: Any = None,
) -> TriageResult:
    """
    Universal CT Slice Triage Screening.
    
    Processes a single 2D HU array through the complete 10-step pipeline
    and returns a TriageResult with findings, borderline_findings, emergency
    score, and action.

    findings:           Confidence >= 0.30 — shown live on frontend.
    borderline_findings: Confidence [0.10, 0.30) — feeds Path A corroboration
                        only; NOT pushed to frontend WebSocket.
    """
    t_start = time.perf_counter()

    # Step 0: Quality Assessment
    is_valid, fail_reason = _step_0_quality_assessment(hu_array)
    if not is_valid:
        elapsed = (time.perf_counter() - t_start) * 1000
        return TriageResult(
            slice_number=slice_id, body_region="unknown", scan_type="unknown",
            statistics={}, findings=[], borderline_findings=[], emergency_score=0,
            action="CONTINUE", processing_time_ms=round(elapsed, 2),
            failure_reason=fail_reason,
        )

    # Step 1: Body Mask
    body_mask = _step_1_body_mask(hu_array)
    
    # Step 2A & 2B: Scan Type & Region
    scan_type = _step_2A_scan_type(hu_array, body_mask)
    region = _step_2B_body_region(hu_array, body_mask, body_part_hint)
    
    # Step 3: Anatomy Suppression (uses TRIAGE_CONFIG)
    search_mask = _step_3_anatomy_suppression(hu_array, body_mask, region)
    
    # Step 4: Adaptive Baseline
    stats = _step_4_adaptive_baseline(hu_array, search_mask)
    
    # Step 5: Candidate Detection & CCA
    raw_comps, num_raw = _step_5_candidate_detection(hu_array, search_mask, stats)
    
    # Step 6: Local Context Validation
    validated_comps = _step_6_local_context_validation(raw_comps, hu_array, search_mask, stats)
    num_validated = len(validated_comps)
    
    # Step 7/8: Geometric Filtering & Confidence — two-tier output
    findings, borderline_findings = _step_7_geometric_filtering_and_confidence(
        validated_comps, stats, search_mask, body_mask, region
    )
    
    # Track findings (if history tracker is provided)
    if history_tracker:
        history_tracker.add_slice_findings(slice_id, findings)
    
    # Step 9/10: Score and Action (based on display-tier findings only)
    score, action = _step_9_10_score_and_action(findings)
    
    elapsed = (time.perf_counter() - t_start) * 1000
    
    log_msg = (
        f"Slice {slice_id} | {region.title()} | {scan_type.title()} | "
        f"Mean: {stats['mean']:.1f} | Std: {stats['std']:.1f} | "
        f"Comps (Raw/Valid): {num_raw}/{num_validated} | "
        f"Findings: {len(findings)} (display) + {len(borderline_findings)} (borderline) | "
        f"Score: {score} | {action} | {elapsed:.1f}ms"
    )
    logger.info(log_msg)

    return TriageResult(
        slice_number=slice_id,
        body_region=region,
        scan_type=scan_type,
        statistics={k: round(v, 2) for k, v in stats.items()},
        findings=findings,
        borderline_findings=borderline_findings,
        emergency_score=score,
        action=action,
        processing_time_ms=round(elapsed, 2),
        num_raw_components=num_raw,
        num_filtered_components=num_validated,
    )
