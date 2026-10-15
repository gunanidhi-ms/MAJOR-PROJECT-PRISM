"""
triage_config.py — Phase 1 Triage Screen Configuration

Single source of truth for all numeric thresholds used in triage_screen.py.
Every branch in _step_3_anatomy_suppression reads from TRIAGE_CONFIG instead
of inline literals, making each threshold visible, named, and citable.

To recalibrate any threshold: edit this file, not triage_screen.py.
The region_agreement.csv confusion matrix (logs/) provides empirical data
for guiding recalibration of the histogram heuristics in _step_2B_body_region.
"""

TRIAGE_CONFIG: dict = {

    # ── Bone suppression ──────────────────────────────────────────────────────
    # Floor HU for the adaptive bone threshold.  P99 of the body sample is
    # computed per-slice, then max()'d with this value.  The floor of 350 HU
    # ensures contrast-enhanced vessels (~200 HU) are NOT suppressed as bone
    # while all cortical bone (always > 350 HU) is caught.
    "bone_threshold_floor_hu": 350.0,

    # ── Boundary erosion (pixels from body mask inner edge) ──────────────────
    # Controls how deeply we erode from the skin/edge before searching for
    # anomalies.  Head needs deeper erosion (thick skull ~8-12 mm).
    "boundary_erosion_px": {"head": 8, "chest": 4, "default": 5},

    # ── Chest air suppression ─────────────────────────────────────────────────
    # Components larger than this are assessed for lung/PTX suppression.
    "large_air_component_px": 5000,
    # Trachea / bronchi are always < this size and are always suppressed.
    "small_airway_component_px": 2000,
    # Density separator between lung parenchyma and background/vacuum air.
    # Real PTX: -900 to -960 HU.  Lung parenchyma: -700 to -900 HU.
    # Background leaked into body mask: -1024 HU.
    # Components > large_air_component_px AND density > lung_density_hu →
    #   suppress as normal lung field.
    # Components > large_air_component_px AND density <= lung_density_hu →
    #   suppress as leaked background air (NOT pneumothorax candidate).
    "lung_density_hu": -980.0,

    # ── Abdomen ───────────────────────────────────────────────────────────────
    # Pixel band from body edge for subcutaneous fat suppression.
    "abdomen_subq_fat_band_px": 15,
    # Normal bowel/stomach gas upper size.  Above this → potential free air.
    "abdomen_gas_max_px": 5000,
    # Compactness (solidity) floor.  Compact = normal bowel gas pocket.
    # Irregular, sheet-like = free air (pneumoperitoneum).
    "abdomen_gas_solidity_min": 0.25,
    # HU floor for normal bowel/stomach gas suppression.
    # Compact gas above this → suppress as normal bowel/stomach (primary rule).
    # A secondary rule also suppresses compact gas down to -950 HU, covering
    # normal colonic haustra gas (routinely -800 to -950 HU in abdominal CT).
    # Only components below -950 HU with irregular shape are kept as
    # potential pneumoperitoneum candidates.
    "abdomen_gas_hu_floor": -700.0,

    # ── Pelvis ────────────────────────────────────────────────────────────────
    # Small compact gas (rectum, bladder) below this pixel count → suppress.
    "pelvis_small_gas_px": 2000,

    # ── Step 7: Confidence tier gates ─────────────────────────────────────────
    # Findings at or above this confidence are shown live on the frontend.
    "confidence_gate_display": 0.30,
    # Findings in [borderline_gate, display_gate) feed Path A (seed_clusterer)
    # for corroboration, but are NOT pushed to the frontend WebSocket.
    # This gives Path A more raw material without flooding the real-time UI.
    "confidence_gate_borderline": 0.10,

    # ── Outer-body dilation band ──────────────────────────────────────────────
    # _step_1_body_mask dilates outward 15px into subcutaneous fat.
    # Findings whose centroid falls within this many pixels of the body mask
    # boundary are almost certainly in fat or skin, not inside any organ.
    # Their confidence is capped at (confidence_gate_borderline + 0.05) so
    # they never reach the display tier at Phase 1.
    # The organ postfilter in Phase 2 gives the definitive verdict.
    "outer_body_band_px": 18,

}
