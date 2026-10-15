"""
anatomy_context.py — Spatial context for report enrichment (Phase 2 → Phase 3)

Derives, purely from the TotalSegmentator label map and candidate geometry:

  * orientation  — which voxel-axis direction is patient left / posterior / superior
                   (inferred from paired L/R organs and spine position; never assumed
                   blindly, and flagged as 'assumed' when it cannot be verified)
  * organ geometry — bbox, centroid, and whether the organ touches the edge of the
                   scanned volume (partial coverage → volume is not trustworthy)
  * organ contacts — structures touching each organ of interest (true surface
                   contact via mask dilation, not bounding-box proximity)
  * per-candidate context
        - sub_location : zone inside the host organ (central/peripheral,
                         anterior/posterior, upper/lower …)
        - description  : anatomical position for candidates with no organ label
                         (side, vertebral level, paravertebral, nearest structures)

Every statement is a geometric fact about the segmentation — no diagnosis.
All outputs are JSON-serialisable (they cross a process boundary to Phase 3).

Axis convention (matches VolumeAccumulator.export_volume): arrays are (Z, Y, X);
candidate centroid_3d / bbox_3d are in (x, y, z) voxel order.
"""

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Labels never worth reporting as an adjacent structure (noise / not informative).
_CONTACT_EXCLUDE = (
    "gluteus", "autochthon", "iliopsoas", "rib_", "costal", "skull", "scapula",
    "clavicula", "humerus", "femur", "hip_", "sacrum", "vertebrae", "sternum",
    "brain", "face",
)

_CONTACT_MARGIN_MM = 3.0
_MIN_CONTACT_VOXELS = 25
_MAX_NEIGHBOURS = 5
_PARAVERTEBRAL_MM = 45.0
_MIDLINE_MM = 10.0


def _side_of(label: str) -> Optional[str]:
    if label.endswith("_left"):
        return "left"
    if label.endswith("_right"):
        return "right"
    return None


def _is_vertebra(label: str) -> bool:
    return label.startswith("vertebrae_")


def _vertebra_level(label: str) -> str:
    return label.split("_", 1)[1] if "_" in label else label


def _humanise(label: str) -> str:
    return label.replace("_", " ")


# ──────────────────────────────────────────────────────────────────────────────
# Geometry
# ──────────────────────────────────────────────────────────────────────────────

def _organ_geometry(seg_arr: np.ndarray, cmap: Dict[int, str]) -> Dict[str, Dict[str, Any]]:
    """bbox / centroid / FOV-edge flags for every labelled structure."""
    from scipy import ndimage as ndi

    arr = seg_arr if np.issubdtype(seg_arr.dtype, np.integer) else seg_arr.astype(np.int32)
    slices = ndi.find_objects(arr)
    Z, Y, X = arr.shape
    geo: Dict[str, Dict[str, Any]] = {}

    for idx, sl in enumerate(slices):
        if sl is None:
            continue
        val = idx + 1
        label = cmap.get(val)
        if not label:
            continue
        sz, sy, sx = sl
        crop = arr[sl] == val
        zz, yy, xx = np.nonzero(crop)
        if zz.size == 0:
            continue
        cx = float(xx.mean() + sx.start)
        cy = float(yy.mean() + sy.start)
        cz = float(zz.mean() + sz.start)
        geo[label] = {
            "bbox": [int(sx.start), int(sy.start), int(sz.start),
                     int(sx.stop - 1), int(sy.stop - 1), int(sz.stop - 1)],
            "centroid": [cx, cy, cz],
            "touches_fov_edge": bool(
                sz.start == 0 or sz.stop == Z or sy.start == 0 or sy.stop == Y
                or sx.start == 0 or sx.stop == X
            ),
            "touches_z_edge": bool(sz.start == 0 or sz.stop == Z),
        }
    return geo


def _infer_orientation(geo: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Decide which index direction is patient-left / posterior / superior from
    anatomy itself. Falls back to the DICOM-LPS default and says so.
    """
    # ── Left/right: compare centroids of *_left vs *_right organs ──
    lefts, rights = [], []
    for label, g in geo.items():
        s = _side_of(label)
        if s == "left" and not _is_vertebra(label):
            lefts.append(g["centroid"][0])
        elif s == "right" and not _is_vertebra(label):
            rights.append(g["centroid"][0])
    if lefts and rights:
        left_is_high_x = bool(np.mean(lefts) > np.mean(rights))
        lr_verified = True
    else:
        left_is_high_x, lr_verified = True, False

    # ── Anterior/posterior: spine sits posterior to the body centroid ──
    verts = [g["centroid"] for l, g in geo.items() if _is_vertebra(l)]
    body = [g["centroid"] for l, g in geo.items() if not _is_vertebra(l)]
    if verts and body:
        posterior_is_high_y = bool(np.mean([v[1] for v in verts]) >= np.mean([b[1] for b in body]))
        ap_verified = True
    else:
        posterior_is_high_y, ap_verified = True, False

    # ── Superior/inferior: cervical/thoracic vertebrae vs lumbar/sacral ──
    upper = [g["centroid"][2] for l, g in geo.items() if l.startswith(("vertebrae_C", "vertebrae_T"))]
    lower = [g["centroid"][2] for l, g in geo.items() if l.startswith(("vertebrae_L", "vertebrae_S"))]
    if upper and lower:
        superior_is_high_z = bool(np.mean(upper) > np.mean(lower))
        si_verified = True
    else:
        chest = [g["centroid"][2] for l, g in geo.items() if l.startswith(("lung_", "heart", "trachea"))]
        pelvis = [g["centroid"][2] for l, g in geo.items() if l in ("urinary_bladder", "prostate", "uterus")
                  or l.startswith(("kidney_", "colon"))]
        if chest and pelvis:
            superior_is_high_z = bool(np.mean(chest) > np.mean(pelvis))
            si_verified = True
        else:
            superior_is_high_z, si_verified = True, False

    return {
        "left_is_high_x": left_is_high_x,
        "posterior_is_high_y": posterior_is_high_y,
        "superior_is_high_z": superior_is_high_z,
        "verified": {"left_right": lr_verified, "anterior_posterior": ap_verified,
                     "superior_inferior": si_verified},
    }


def _midline_x(geo: Dict[str, Dict[str, Any]], shape_x: int) -> float:
    verts = [g["centroid"][0] for l, g in geo.items() if _is_vertebra(l)]
    if verts:
        return float(np.mean(verts))
    return shape_x / 2.0


def _vertebra_levels(geo: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for label, g in geo.items():
        if _is_vertebra(label):
            out.append({
                "level": _vertebra_level(label),
                "z_min": g["bbox"][2], "z_max": g["bbox"][5],
                "z_center": g["centroid"][2],
            })
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Organ contacts (true surface contact)
# ──────────────────────────────────────────────────────────────────────────────

def _organ_contacts(
    seg_arr: np.ndarray,
    cmap: Dict[int, str],
    geo: Dict[str, Dict[str, Any]],
    spacing: Tuple[float, float, float],
    organs: Iterable[str],
) -> Dict[str, List[Dict[str, Any]]]:
    from scipy import ndimage as ndi

    name_to_val = {v: k for k, v in cmap.items()}
    Z, Y, X = seg_arr.shape
    row_mm, col_mm, z_mm = spacing
    iters = max(1, int(round(_CONTACT_MARGIN_MM / max(min(row_mm, col_mm, z_mm), 0.1))))
    out: Dict[str, List[Dict[str, Any]]] = {}

    for organ in organs:
        g = geo.get(organ)
        val = name_to_val.get(organ)
        if g is None or val is None:
            continue
        x0, y0, z0, x1, y1, z1 = g["bbox"]
        pad = iters + 1
        sl = (
            slice(max(0, z0 - pad), min(Z, z1 + pad + 1)),
            slice(max(0, y0 - pad), min(Y, y1 + pad + 1)),
            slice(max(0, x0 - pad), min(X, x1 + pad + 1)),
        )
        crop = seg_arr[sl]
        mask = crop == val
        ring = ndi.binary_dilation(mask, iterations=iters) & ~mask
        vals, counts = np.unique(crop[ring], return_counts=True)
        neighbours = []
        for v, c in sorted(zip(vals.tolist(), counts.tolist()), key=lambda t: -t[1]):
            if v == 0 or c < _MIN_CONTACT_VOXELS:
                continue
            label = cmap.get(v)
            if not label or label == organ:
                continue
            if any(tok in label for tok in _CONTACT_EXCLUDE):
                continue
            neighbours.append({"label": label, "contact_voxels": int(c)})
            if len(neighbours) >= _MAX_NEIGHBOURS:
                break
        out[organ] = neighbours
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Per-candidate localisation
# ──────────────────────────────────────────────────────────────────────────────

def _frac(value: float, lo: float, hi: float) -> Optional[float]:
    if hi - lo < 1e-6:
        return None
    return float(min(1.0, max(0.0, (value - lo) / (hi - lo))))


def sub_location_phrase(
    organ_label: str,
    centroid: List[float],
    organ_geo: Dict[str, Any],
    orientation: Dict[str, Any],
    midline_x: float,
    spacing: Tuple[float, float, float],
) -> str:
    """Zone of ``centroid`` inside its host organ, in patient-anatomical terms."""
    row_mm, col_mm, z_mm = spacing
    x0, y0, z0, x1, y1, z1 = organ_geo["bbox"]
    cx, cy, cz = centroid
    parts: List[str] = []
    is_lung = organ_label.startswith("lung_")
    side = _side_of(organ_label)

    # Medial–lateral / left–right (skip when organ is thin in that axis: < 15 mm)
    fx = _frac(cx, x0, x1)
    if fx is not None and (x1 - x0 + 1) * col_mm >= 15:
        if side:
            # Which end of the bbox faces the midline?
            medial_is_high = (midline_x > (x0 + x1) / 2.0)
            med_frac = fx if not medial_is_high else 1.0 - fx   # 0 = at medial edge
            if is_lung:
                if med_frac < 0.33:
                    parts.append("central")
                elif med_frac > 0.67:
                    parts.append("peripheral")
            else:
                if med_frac < 0.33:
                    parts.append("medial")
                elif med_frac > 0.67:
                    parts.append("lateral")
        else:
            left_frac = fx if orientation["left_is_high_x"] else 1.0 - fx
            if left_frac < 0.33:
                parts.append("right-sided")
            elif left_frac > 0.67:
                parts.append("left-sided")

    # Anterior–posterior
    fy = _frac(cy, y0, y1)
    if fy is not None and (y1 - y0 + 1) * row_mm >= 15:
        post_frac = fy if orientation["posterior_is_high_y"] else 1.0 - fy
        if post_frac < 0.33:
            parts.append("anterior")
        elif post_frac > 0.67:
            parts.append("posterior")

    # Superior–inferior
    fz = _frac(cz, z0, z1)
    if fz is not None and (z1 - z0 + 1) * z_mm >= 15:
        sup_frac = fz if orientation["superior_is_high_z"] else 1.0 - fz
        if sup_frac > 0.67:
            parts.append("upper" if is_lung else "superior")
        elif sup_frac < 0.33:
            parts.append("lower" if is_lung else "inferior")

    if not parts:
        return ""
    return ", ".join(parts) + " portion"


def _locate_unclassified(
    cand,
    geo: Dict[str, Dict[str, Any]],
    orientation: Dict[str, Any],
    midline_x: float,
    levels: List[Dict[str, Any]],
    spine_xy: Optional[Tuple[float, float]],
    body_y: Optional[float],
    spacing: Tuple[float, float, float],
    seg_arr: np.ndarray,
    cmap: Dict[int, str],
) -> Dict[str, Any]:
    """Anatomical description for a candidate with no organ label."""
    row_mm, col_mm, z_mm = spacing
    cx, cy, cz = cand.centroid_3d
    info: Dict[str, Any] = {}

    # Side / midline
    dx_mm = (cx - midline_x) * col_mm
    if abs(dx_mm) < _MIDLINE_MM:
        side = "midline"
    else:
        is_left = (dx_mm > 0) == orientation["left_is_high_x"]
        side = "left" if is_left else "right"
    info["side"] = side

    # AP half
    if body_y is not None:
        post = (cy > body_y) if orientation["posterior_is_high_y"] else (cy < body_y)
        info["ap"] = "posterior" if post else "anterior"

    # Vertebral level
    level_txt = ""
    if levels:
        inside = [l for l in levels if l["z_min"] <= cz <= l["z_max"]]
        if inside:
            nearest = min(inside, key=lambda l: abs(l["z_center"] - cz))
            level_txt = f"at the {nearest['level']} level"
        else:
            nearest = min(levels, key=lambda l: abs(l["z_center"] - cz))
            level_txt = f"at approximately the {nearest['level']} level"
    info["level"] = level_txt

    # Paravertebral?
    paravertebral = False
    if spine_xy is not None:
        d = np.hypot((cx - spine_xy[0]) * col_mm, (cy - spine_xy[1]) * row_mm)
        paravertebral = bool(d <= _PARAVERTEBRAL_MM)
    info["paravertebral"] = paravertebral

    # Nearest segmented structures within the candidate's bounding region
    nearby: List[str] = []
    try:
        Z, Y, X = seg_arr.shape
        x0, y0, z0, x1, y1, z1 = [int(v) for v in cand.bbox_3d]
        mx = max(1, int(round(5.0 / max(col_mm, 0.1))))
        my = max(1, int(round(5.0 / max(row_mm, 0.1))))
        mz = max(1, int(round(5.0 / max(z_mm, 0.1))))
        crop = seg_arr[
            max(0, z0 - mz): min(Z, z1 + mz + 1),
            max(0, y0 - my): min(Y, y1 + my + 1),
            max(0, x0 - mx): min(X, x1 + mx + 1),
        ]
        vals, counts = np.unique(crop, return_counts=True)
        for v, c in sorted(zip(vals.tolist(), counts.tolist()), key=lambda t: -t[1]):
            if v == 0 or c < 10:
                continue
            label = cmap.get(v)
            if not label or any(tok in label for tok in _CONTACT_EXCLUDE):
                continue
            nearby.append(label)
            if len(nearby) >= 3:
                break
    except Exception as e:
        logger.debug("anatomy_context: nearby-structure scan failed: %s", e)
    info["nearby"] = nearby

    # Compose
    if paravertebral:
        desc = "midline paravertebral region" if side == "midline" else f"{side} paravertebral region"
    else:
        ap = info.get("ap", "")
        if side == "midline":
            desc = f"midline {ap} location".replace("  ", " ")
        else:
            desc = f"{side}-sided {ap} location".replace("  ", " ")
    if level_txt:
        desc += f", {level_txt}"
    if nearby:
        desc += f"; closest segmented structures: {', '.join(_humanise(n) for n in nearby)}"
    info["description"] = desc.strip()
    return info


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def build_anatomy_context(
    seg_arr: np.ndarray,
    cmap: Dict[int, str],
    spacing: Tuple[float, float, float],
    candidates: List[Any],
    organs_of_interest: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    Build the JSON-serialisable anatomy context handed to the findings adapter.

    Args:
        seg_arr: (Z, Y, X) integer label volume from TotalSegmentator.
        cmap: {label value → organ name}.
        spacing: (row_mm, col_mm, z_mm).
        candidates: surviving Candidate objects.
        organs_of_interest: organs to compute surface contacts for. Defaults to
            organs that host a surviving candidate.
    """
    geo = _organ_geometry(seg_arr, cmap)
    orientation = _infer_orientation(geo)
    Z, Y, X = seg_arr.shape
    midline = _midline_x(geo, X)
    levels = _vertebra_levels(geo)

    verts = [g["centroid"] for l, g in geo.items() if _is_vertebra(l)]
    spine_xy = (float(np.mean([v[0] for v in verts])), float(np.mean([v[1] for v in verts]))) if verts else None
    body = [g["centroid"][1] for l, g in geo.items() if not _is_vertebra(l)]
    body_y = float(np.mean(body)) if body else None

    survivors = [c for c in candidates if not getattr(c, "suppressed", False)]
    if organs_of_interest is None:
        organs_of_interest = {c.organ_label for c in survivors if c.organ_label in geo}
    contacts = _organ_contacts(seg_arr, cmap, geo, spacing, organs_of_interest)

    cand_ctx: Dict[str, Dict[str, Any]] = {}
    for c in survivors:
        try:
            if c.organ_label in geo:
                phrase = sub_location_phrase(
                    c.organ_label, list(c.centroid_3d), geo[c.organ_label],
                    orientation, midline, spacing,
                )
                cand_ctx[c.candidate_id] = {"sub_location": phrase}
            else:
                cand_ctx[c.candidate_id] = _locate_unclassified(
                    c, geo, orientation, midline, levels, spine_xy, body_y,
                    spacing, seg_arr, cmap,
                )
        except Exception as e:
            logger.warning("anatomy_context: failed for candidate %s: %s", c.candidate_id, e)

    organs_out = {}
    for label, g in geo.items():
        entry = {
            "touches_fov_edge": g["touches_fov_edge"],
            "touches_z_edge": g["touches_z_edge"],
        }
        if label in contacts:
            entry["contacts"] = [n["label"] for n in contacts[label]]
        organs_out[label] = entry

    logger.info(
        "anatomy_context: %d organs, %d candidates localised, orientation verified=%s",
        len(organs_out), len(cand_ctx), orientation["verified"],
    )
    return {
        "orientation": orientation,
        "midline_x": midline,
        "organs": organs_out,
        "candidates": cand_ctx,
    }
