"""
mesh_builder.py — 3D Organ Surface Mesh Generator

Converts TotalSegmentator's 3D segmentation volume (temp_seg.nii) into
browser-renderable surface meshes using the Marching Cubes algorithm
(skimage.measure.marching_cubes).

Output format (compact JSON for Three.js BufferGeometry):
    {
        "organs": [
            {
                "label":    "liver",
                "vertices": [[x,y,z], ...],   # normalised to [-0.5, 0.5]
                "faces":    [[a,b,c], ...],    # triangle face indices
                "colours":  [[r,g,b], ...],    # per-vertex RGB [0..1]
                                               # grey = normal, red = anomaly
                "organ_colour": [r,g,b],       # anatomical base colour
                "is_anomalous": bool,          # does any anomaly overlap?
                "volume_cc": float,
            },
            ...
        ],
        "anomalies": [                         # standalone anomaly boxes (for
            {                                  # candidates outside seg organs)
                "bbox_norm": [x,y,z,w,h,d],   # normalised centre + size
                "label":     str,
                "hu_mean":   float,
            }
        ],
        "region":   str,
        "n_organs": int,
        "n_anomaly_candidates": int,
    }

Performance design:
    - Volume is downsampled BEFORE marching cubes (skip factor 2–3) to
      produce meshes of ~5k–20k triangles per organ, safe for WebGL.
    - Per-vertex anomaly colouring: a vertex is red if its 3D position
      falls within any candidate's bbox_3d (voxel space, pre-downsample).
    - Mesh is NOT sent as floats with full precision; vertices are rounded
      to 4 decimal places to reduce JSON payload.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# ── Anatomical colour palette (matches frontend VolumetricViewer) ──────────────
# Values are [R, G, B] normalised 0..1
ORGAN_COLOURS: Dict[str, List[float]] = {
    "liver":              [0.55, 0.27, 0.07],
    "spleen":             [0.61, 0.14, 0.21],
    "kidney_right":       [0.80, 0.40, 0.00],
    "kidney_left":        [0.80, 0.40, 0.00],
    "gallbladder":        [0.40, 0.67, 0.27],
    "pancreas":           [0.80, 0.60, 0.33],
    "stomach":            [0.53, 0.53, 0.20],
    "urinary_bladder":    [0.33, 0.60, 0.80],
    "heart":              [0.80, 0.13, 0.27],
    "aorta":              [1.00, 0.40, 0.27],
    "lung_left":          [0.40, 0.60, 0.80],
    "lung_right":         [0.40, 0.60, 0.80],
    "trachea":            [0.60, 0.80, 0.90],
    "esophagus":          [0.70, 0.50, 0.40],
    "small_bowel":        [0.55, 0.45, 0.35],
    "duodenum":           [0.60, 0.48, 0.30],
    "colon":              [0.45, 0.35, 0.25],
    "femur_left":         [0.90, 0.87, 0.78],
    "femur_right":        [0.90, 0.87, 0.78],
    "hip_left":           [0.85, 0.82, 0.73],
    "hip_right":          [0.85, 0.82, 0.73],
    "sacrum":             [0.88, 0.85, 0.75],
    "spine_L1":           [0.85, 0.82, 0.73],
    "spine_L2":           [0.85, 0.82, 0.73],
    "spine_L3":           [0.85, 0.82, 0.73],
    "spine_L4":           [0.85, 0.82, 0.73],
    "spine_L5":           [0.85, 0.82, 0.73],
    "inferior_vena_cava": [0.20, 0.40, 0.80],
    "portal_vein_and_splenic_vein": [0.60, 0.20, 0.60],
}
_DEFAULT_ORGAN_COLOUR = [0.38, 0.55, 0.68]   # steel blue for unlisted organs
_ANOMALY_COLOUR       = [1.0,  0.13, 0.27]   # bright red
_NORMAL_GREY          = [0.55, 0.60, 0.65]   # cool grey for vertex colouring


def build_organ_meshes(
    seg_arr: np.ndarray,
    cmap: Dict[int, str],
    candidates: List[Dict],
    spacing: Tuple[float, float, float],
    downsample: int = 2,
    max_organs: int = 30,
) -> Dict[str, Any]:
    """
    Generate 3D surface meshes for all segmented organs and colour
    anomalous zones red.

    Args:
        seg_arr:    3D integer label array (Z, Y, X), 0 = background.
        cmap:       {int_label: organ_name_str} from TotalSegmentator.
        candidates: List of candidate dicts from candidates.json.
                    Each must have "bbox_3d": [x0,y0,z0,x1,y1,z1] in
                    voxel index space (X, Y, Z ordering from the Candidate
                    schema).
        spacing:    (row_mm, col_mm, z_mm) — physical voxel spacing.
        downsample: Stride when downsampling seg_arr before marching cubes.
                    2 = 8x volume reduction, good quality + fast.
        max_organs: Maximum number of organs to include in output (sorted
                    by volume descending, keep largest).

    Returns:
        Dict matching the output schema described in the module docstring.
    """
    from skimage.measure import marching_cubes

    # ── 0. Downsample seg_arr for mesh extraction ──────────────────────────────
    ds = int(downsample)
    seg_ds = seg_arr[::ds, ::ds, ::ds]   # shape (Z', Y', X')
    Z_ds, Y_ds, X_ds = seg_ds.shape

    # spacing is (z_mm, y_mm, x_mm) matching the (Z, Y, X) array axis order
    z_mm, y_mm, x_mm = spacing
    spacing_ds = (z_mm * ds, y_mm * ds, x_mm * ds)   # (Z, Y, X) in mm after downsample

    # ── 1. Compute global bounding box in mm for normalisation ─────────────────
    # We normalise ALL organ vertices into [-0.5, 0.5]³ centred at origin
    # so Three.js can render them without unit conversion.
    vol_mm = np.array([
        Z_ds * spacing_ds[0],
        Y_ds * spacing_ds[1],
        X_ds * spacing_ds[2],
    ])
    max_dim = vol_mm.max()  # longest physical dimension → unit cube edge

    def vox_to_norm(verts_zyx: np.ndarray) -> np.ndarray:
        """Convert (N,3) voxel vertices (z,y,x) → normalised (N,3) (x,y,z)."""
        # Scale to mm then divide by max_dim → [0, 1] then centre → [-0.5, 0.5]
        mm = verts_zyx * np.array(spacing_ds)           # (N,3) in mm (z,y,x)
        norm = mm / max_dim - 0.5                        # (N,3) normalised (z,y,x)
        # Reorder axes to X, Y, Z for Three.js right-hand coordinate system
        return norm[:, [2, 1, 0]]                        # (N,3): x, y, z

    # ── 2. Build anomaly voxel set (in DOWNSAMPLED voxel space) ───────────────
    # Candidates have bbox_3d = [x0, y0, z0, x1, y1, z1] (voxel indices,
    # Candidate schema uses X=col, Y=row, Z=slice ordering).
    # We convert to downsampled seg_arr indexing: seg_ds[z, y, x]
    anomaly_vox_boxes = []   # list of (z0,y0,x0, z1,y1,x1) in ds space
    unsuppressed = [c for c in candidates if not c.get("suppressed", False)]

    for cand in unsuppressed:
        b = cand.get("bbox_3d", [])
        if len(b) < 6:
            continue
        # bbox_3d schema: [x_min, y_min, z_min, x_max, y_max, z_max]
        # (X=column, Y=row, Z=slice per Candidate.bbox_3d definition)
        x0, y0, z0, x1, y1, z1 = [int(v) for v in b]
        # Convert to downsampled indices
        anomaly_vox_boxes.append((
            z0 // ds, y0 // ds, x0 // ds,
            z1 // ds, y1 // ds, x1 // ds,
        ))

    def vertex_is_anomalous(verts_zyx: np.ndarray) -> np.ndarray:
        """
        Return boolean mask (N,) — True if vertex is inside any anomaly bbox
        in downsampled voxel space.
        """
        anomalous = np.zeros(len(verts_zyx), dtype=bool)
        vz = verts_zyx[:, 0]
        vy = verts_zyx[:, 1]
        vx = verts_zyx[:, 2]
        for (az0, ay0, ax0, az1, ay1, ax1) in anomaly_vox_boxes:
            mask = (
                (vz >= az0) & (vz <= az1) &
                (vy >= ay0) & (vy <= ay1) &
                (vx >= ax0) & (vx <= ax1)
            )
            anomalous |= mask
        return anomalous

    # ── 3. Sort organs by voxel count (largest first) ─────────────────────────
    unique_vals, counts = np.unique(seg_ds, return_counts=True)
    organ_order = [
        (val, cnt) for val, cnt in zip(unique_vals, counts)
        if val != 0 and cmap.get(int(val), "")
    ]
    organ_order.sort(key=lambda x: x[1], reverse=True)
    organ_order = organ_order[:max_organs]

    # ── 4. Run marching cubes per organ ───────────────────────────────────────
    output_organs = []
    for val, vox_count in organ_order:
        label = cmap.get(int(val), f"organ_{val}")
        organ_mask = (seg_ds == val)

        # Skip organs too small to produce a reasonable mesh
        if vox_count < 20:
            continue

        try:
            verts, faces, normals, _ = marching_cubes(
                organ_mask.astype(np.float32),
                level=0.5,
                spacing=spacing_ds,
                allow_degenerate=False,
            )
        except Exception as e:
            logger.debug("marching_cubes failed for %s: %s", label, e)
            continue

        if len(verts) == 0 or len(faces) == 0:
            continue

        # Convert mm-space verts (z,y,x in spacing_ds units) to voxel space
        # for anomaly containment check, then normalise to [-0.5, 0.5]
        verts_vox_zyx = verts / np.array(spacing_ds)   # (N,3) in ds-voxel space

        # Per-vertex colour: grey = normal, red = anomaly
        is_anom = vertex_is_anomalous(verts_vox_zyx)
        organ_col = ORGAN_COLOURS.get(label, _DEFAULT_ORGAN_COLOUR)
        colours = np.where(
            is_anom[:, None],
            np.array(_ANOMALY_COLOUR),
            np.array(_NORMAL_GREY),
        ).astype(np.float32)   # (N, 3)

        # Normalise vertex positions
        verts_norm = vox_to_norm(verts_vox_zyx)   # (N,3) x,y,z in [-0.5, 0.5]

        # Decimate mesh if too large (keep at most 15k triangles per organ)
        verts_norm, faces, colours = _decimate(verts_norm, faces, colours, max_faces=12000)

        volume_cc = float(vox_count * (x_mm * ds) * (y_mm * ds) * (z_mm * ds) / 1000.0)
        is_anomalous = bool(np.any(is_anom))

        output_organs.append({
            "label":         label,
            "vertices":      verts_norm.round(4).tolist(),
            "faces":         faces.tolist(),
            "colours":       colours.round(3).tolist(),
            "organ_colour":  organ_col,
            "is_anomalous":  is_anomalous,
            "volume_cc":     round(volume_cc, 2),
        })
        logger.debug(
            "mesh_builder: %s — %d verts, %d faces, anomalous=%s",
            label, len(verts_norm), len(faces), is_anomalous,
        )

    # ── 5. Build standalone anomaly boxes for candidates not inside any organ ──
    # (These are the Phase 1 findings that survived but don't match an organ.
    #  We still show them as small bounding boxes, but labelled as unconfirmed.)
    anomaly_boxes = []
    for cand in unsuppressed[:50]:  # cap at 50 standalone boxes
        b = cand.get("bbox_3d", [])
        if len(b) < 6:
            continue
        x0, y0, z0, x1, y1, z1 = b
        cx_n = _norm_coord((x0+x1)/2, 0, seg_arr.shape[2], max_dim, x_mm)
        cy_n = _norm_coord((y0+y1)/2, 0, seg_arr.shape[1], max_dim, y_mm)
        cz_n = _norm_coord((z0+z1)/2, 0, seg_arr.shape[0], max_dim, z_mm)
        w_n  = ((x1-x0)*x_mm) / max_dim
        h_n  = ((y1-y0)*y_mm) / max_dim
        d_n  = ((z1-z0)*z_mm)   / max_dim
        anomaly_boxes.append({
            "bbox_norm":  [round(cx_n,4), round(cy_n,4), round(cz_n,4),
                           round(w_n,4),  round(h_n,4),  round(d_n,4)],
            "label":      cand.get("organ_label", ""),
            "hu_mean":    cand.get("density_hu", {}).get("mean", 0),
            "confidence": cand.get("confidence_score", 0),
            "validated_label": cand.get("validated_label", ""),
        })

    logger.info(
        "mesh_builder: %d organs meshed, %d anomaly boxes, %d unsuppressed candidates",
        len(output_organs), len(anomaly_boxes), len(unsuppressed),
    )

    return {
        "organs":                  output_organs,
        "anomalies":               anomaly_boxes,
        "region":                  "",
        "n_organs":                len(output_organs),
        "n_anomaly_candidates":    len(unsuppressed),
        "max_dim_mm":              round(float(max_dim), 2),
    }


# ── Private helpers ─────────────────────────────────────────────────────────────

def _norm_coord(v, v_min, v_max, max_dim, mm_per_vox):
    """Normalise a single voxel coordinate to [-0.5, 0.5] space."""
    return (v * mm_per_vox) / max_dim - 0.5


def _decimate(
    verts: np.ndarray,
    faces: np.ndarray,
    colours: np.ndarray,
    max_faces: int = 12000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast uniform decimation: keep every N-th face to stay under max_faces.
    Rebuilds a compact vertex buffer for the kept faces.
    Preserves per-vertex colours by majority-vote on kept faces.
    """
    if len(faces) <= max_faces:
        return verts, faces, colours

    # Stride decimation: keep every stride-th face
    stride = int(np.ceil(len(faces) / max_faces))
    kept_faces = faces[::stride]

    # Compact: only keep vertices that are referenced by kept_faces
    used_verts = np.unique(kept_faces.ravel())
    old_to_new = np.full(len(verts), -1, dtype=np.int32)
    old_to_new[used_verts] = np.arange(len(used_verts), dtype=np.int32)

    new_verts   = verts[used_verts]
    new_colours = colours[used_verts]
    new_faces   = old_to_new[kept_faces]

    return new_verts, new_faces, new_colours
