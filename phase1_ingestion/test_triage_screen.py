"""
test_triage_screen.py -- Comprehensive Fake CT Slice Test Suite

Generates synthetic 512x512 HU arrays that approximate real CT anatomy
and validates the triage engine against expected outcomes.

Usage:
    python -m phase1_ingestion.test_triage_screen
"""

import numpy as np
from phase1_ingestion.triage_screen import screen_slice


# --- Helpers -----------------------------------------------------------------

def make_ellipse(shape, center, radii, value, arr):
    """Draw a filled ellipse on arr."""
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    mask = ((yy - center[0]) / radii[0]) ** 2 + ((xx - center[1]) / radii[1]) ** 2 <= 1
    arr[mask] = value
    return mask


def make_circle(shape, center, radius, value, arr):
    """Draw a filled circle on arr."""
    return make_ellipse(shape, center, (radius, radius), value, arr)


def make_rect(arr, y_start, y_end, x_start, x_end, value):
    """Draw a filled rectangle on arr."""
    arr[y_start:y_end, x_start:x_end] = value


def make_ring(shape, center, outer_r, inner_r, value, arr):
    """Draw an annular ring (ring of bone, etc)."""
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    outer = ((yy - center[0]) / outer_r[0]) ** 2 + ((xx - center[1]) / outer_r[1]) ** 2 <= 1
    inner = ((yy - center[0]) / inner_r[0]) ** 2 + ((xx - center[1]) / inner_r[1]) ** 2 <= 1
    ring = outer & ~inner
    arr[ring] = value
    return ring


def add_noise(arr, mask, mean, std):
    """Add Gaussian noise to a region defined by mask."""
    region = mask if isinstance(mask, np.ndarray) else (arr == mask)
    arr[region] = np.random.normal(mean, std, np.sum(region)).astype(np.float32)


# --- Fake CT Generators -----------------------------------------------------

def gen_normal_head():
    """Normal head CT: skull ring + uniform brain parenchyma + CSF ventricles.
    
    Anatomy:
      - Air background at -1024
      - Skull: thin ring of bone (900 HU) between radii 190-200 and 170-180
      - Brain parenchyma: N(35, 8) HU
      - CSF ventricles: N(8, 3) HU
    """
    arr = np.full((512, 512), -1024.0, dtype=np.float32)
    # Skull ring (bone)
    make_ring((512, 512), (256, 256), (200, 180), (185, 165), 900.0, arr)
    # Brain parenchyma (fills inside skull)
    brain_mask = make_ellipse((512, 512), (256, 256), (185, 165), 35.0, arr)
    add_noise(arr, brain_mask, 35, 8)
    # CSF ventricles
    vent_mask = make_ellipse((512, 512), (250, 256), (18, 30), 8.0, arr)
    add_noise(arr, vent_mask, 8, 3)
    return arr


def gen_head_hemorrhage():
    """Head CT with acute intracerebral hemorrhage (~72 HU, ~25px radius).
    
    The hemorrhage is well within the brain, at a density clearly above
    normal parenchyma (35 HU) but below bone (900 HU).
    """
    arr = gen_normal_head()
    bleed_mask = make_circle((512, 512), (220, 310), 25, 72.0, arr)
    add_noise(arr, bleed_mask, 72, 4)
    return arr


def gen_head_massive_hemorrhage():
    """Head CT with massive hemorrhage (~78 HU, ~50x45 px ellipse = ~7000px area).
    
    This represents a large parenchymal hematoma that should trigger URGENT.
    """
    arr = gen_normal_head()
    bleed_mask = make_ellipse((512, 512), (230, 300), (50, 45), 78.0, arr)
    add_noise(arr, bleed_mask, 78, 5)
    return arr


def gen_normal_chest():
    """Normal chest CT: body wall + bilateral lungs + mediastinum + spine.
    
    Anatomy:
      - Body wall: N(30, 20) HU soft tissue
      - Bilateral lungs: N(-850, 25) HU, large 
      - Heart: N(50, 10) HU
      - Spine: 600 HU (will be suppressed as bone)
      - Sternum: 500 HU (will be suppressed as bone)
    """
    arr = np.full((512, 512), -1024.0, dtype=np.float32)
    # Body wall
    body_mask = make_ellipse((512, 512), (270, 256), (200, 220), 30.0, arr)
    add_noise(arr, body_mask, 30, 20)
    # Left lung
    ll_mask = make_ellipse((512, 512), (240, 160), (140, 80), -850.0, arr)
    add_noise(arr, ll_mask, -850, 25)
    # Right lung
    rl_mask = make_ellipse((512, 512), (240, 350), (140, 80), -850.0, arr)
    add_noise(arr, rl_mask, -850, 25)
    # Spine
    make_ellipse((512, 512), (380, 256), (20, 18), 600.0, arr)
    # Sternum
    make_rect(arr, 80, 110, 250, 262, 500.0)
    # Heart
    heart_mask = make_ellipse((512, 512), (280, 240), (50, 55), 50.0, arr)
    add_noise(arr, heart_mask, 50, 10)
    return arr


def gen_chest_with_pneumothorax():
    """Chest CT with right-sided pneumothorax (air in pleural space).
    
    Key: The PTX crescent must be physically SEPARATED from the lung by
    a thick band of visceral pleura (soft tissue). We achieve this by:
      1. Making the right lung smaller (shrunk by 15px in each radius)
      2. Filling the gap between the old and new lung boundary with soft tissue
      3. Placing PTX air OUTSIDE this soft tissue barrier
    """
    arr = np.full((512, 512), -1024.0, dtype=np.float32)
    yy, xx = np.ogrid[:512, :512]
    
    # Body wall
    body_mask = make_ellipse((512, 512), (270, 256), (200, 220), 30.0, arr)
    add_noise(arr, body_mask, 30, 20)
    
    # Left lung (normal, full size)
    ll_mask = make_ellipse((512, 512), (240, 160), (140, 80), -850.0, arr)
    add_noise(arr, ll_mask, -850, 25)
    
    # Right lung (SMALLER -- collapsed due to PTX)
    rl_mask = make_ellipse((512, 512), (240, 350), (120, 60), -850.0, arr)
    add_noise(arr, rl_mask, -850, 25)
    
    # Visceral pleura: thick band of soft tissue around the collapsed right lung
    # This MUST be thick enough (>= 8px) to break air connectivity at the -500 HU threshold
    pleura_outer = ((yy - 240) / 128.0) ** 2 + ((xx - 350) / 68.0) ** 2 <= 1
    pleura_inner = ((yy - 240) / 120.0) ** 2 + ((xx - 350) / 60.0) ** 2 <= 1
    pleura_band = pleura_outer & ~pleura_inner
    arr[pleura_band] = np.random.normal(30, 5, np.sum(pleura_band)).astype(np.float32)
    
    # PTX crescent: air OUTSIDE the pleural barrier, INSIDE the body wall
    # Only on the right (anterior-lateral) side, NOT touching the pleura
    ptx_outer = ((yy - 240) / 160.0) ** 2 + ((xx - 350) / 95.0) ** 2 <= 1
    ptx_inner = ((yy - 240) / 135.0) ** 2 + ((xx - 350) / 75.0) ** 2 <= 1
    ptx_region = ptx_outer & ~ptx_inner & (xx > 360) & (yy > 120) & (yy < 320)
    # Exclude any overlap with the pleural band
    ptx_region = ptx_region & ~pleura_band
    # CRITICAL: Ensure PTX does not bleed outside the body wall into background air
    body_wall_inner = ((yy - 270) / 190.0) ** 2 + ((xx - 256) / 210.0) ** 2 <= 1
    ptx_region = ptx_region & body_wall_inner
    arr[ptx_region] = np.random.normal(-960, 8, np.sum(ptx_region)).astype(np.float32)
    
    # Spine and sternum
    make_ellipse((512, 512), (380, 256), (20, 18), 600.0, arr)
    make_rect(arr, 80, 110, 250, 262, 500.0)
    # Heart
    heart_mask = make_ellipse((512, 512), (280, 240), (50, 55), 50.0, arr)
    add_noise(arr, heart_mask, 50, 10)
    return arr


def gen_normal_abdomen():
    """Normal abdomen CT: body wall + fat ring + internal organs + bowel gas.
    
    Anatomy:
      - Subcutaneous fat ring: N(-80, 15) HU
      - Soft tissue body: N(45, 15) HU
      - Liver: 60 HU
      - Spleen: 55 HU
      - Spine: 650 HU (bone, will be suppressed)
      - Small bowel gas pockets: -300 to -400 HU (normal, compact, should be suppressed)
    """
    arr = np.full((512, 512), -1024.0, dtype=np.float32)
    # Body
    body_mask = make_ellipse((512, 512), (256, 256), (210, 200), 45.0, arr)
    add_noise(arr, body_mask, 45, 15)
    # Fat ring
    fat_ring = make_ring((512, 512), (256, 256), (210, 200), (195, 185), -80.0, arr)
    add_noise(arr, fat_ring, -80, 15)
    # Liver
    make_ellipse((512, 512), (230, 340), (60, 70), 60.0, arr)
    # Spleen
    make_ellipse((512, 512), (230, 160), (35, 40), 55.0, arr)
    # Spine (bone)
    make_ellipse((512, 512), (380, 256), (18, 16), 650.0, arr)
    # Normal bowel gas (small, compact, moderate density: -300 to -400 HU)
    make_circle((512, 512), (280, 200), 8, -350.0, arr)
    make_circle((512, 512), (310, 300), 6, -400.0, arr)
    make_circle((512, 512), (260, 280), 5, -300.0, arr)
    return arr


def gen_abdomen_with_free_air():
    """Abdomen CT with pathological free air (pneumoperitoneum).
    
    Free air rises to non-dependent areas and is characteristically:
      - Very low density (< -800 HU, near-vacuum)
      - Large crescent under the anterior abdominal wall
      - NOT compact like bowel gas
    
    This should NOT be suppressed by the bowel gas filter.
    """
    arr = gen_normal_abdomen()
    # Large crescent of free air deep inside the body (well past dist>15 boundary)
    # at y=100-130 (well within the body ellipse which extends to ~y=46)
    # This ensures dist > 15 so it enters the air analysis path
    make_rect(arr, 100, 135, 170, 340, -920.0)
    # Additional irregular free air collection
    make_ellipse((512, 512), (105, 256), (20, 70), -950.0, arr)
    return arr


def gen_normal_pelvis():
    """Normal pelvis CT: pelvic bones + bladder + rectum.
    
    Key: Pelvic bones are large but should be completely suppressed
    by the bone threshold + dilation. Rectal gas is expected.
    """
    arr = np.full((512, 512), -1024.0, dtype=np.float32)
    # Body
    body_mask = make_ellipse((512, 512), (256, 256), (200, 220), 40.0, arr)
    add_noise(arr, body_mask, 40, 12)
    # Pelvic bones (bilateral iliac crests)
    make_ellipse((512, 512), (256, 130), (80, 30), 700.0, arr)
    make_ellipse((512, 512), (256, 380), (80, 30), 700.0, arr)
    # Sacrum (same density as iliac crests)
    make_ellipse((512, 512), (360, 256), (25, 30), 700.0, arr)
    # Bladder (fluid-filled)
    blad_mask = make_ellipse((512, 512), (230, 256), (40, 45), 15.0, arr)
    add_noise(arr, blad_mask, 15, 5)
    # Rectal gas (normal, compact, small)
    make_circle((512, 512), (310, 256), 10, -300.0, arr)
    return arr


def gen_neck():
    """Normal neck CT: spine + trachea + pharynx + soft tissue.
    
    Key characteristics that distinguish neck from abdomen:
      - Small body cross-section (low body_fill ratio)
      - Central airway (trachea/pharynx)
      - Small cervical spine
    """
    arr = np.full((512, 512), -1024.0, dtype=np.float32)
    # Small oval body (much smaller than abdomen/chest)
    body_mask = make_ellipse((512, 512), (256, 256), (100, 85), 40.0, arr)
    add_noise(arr, body_mask, 40, 10)
    # Cervical spine (small)
    make_ellipse((512, 512), (305, 256), (12, 11), 700.0, arr)
    # Trachea (central air)
    make_circle((512, 512), (240, 256), 10, -950.0, arr)
    # Pharynx / airway (slightly larger)
    make_ellipse((512, 512), (225, 256), (15, 12), -900.0, arr)
    return arr


def gen_extremity():
    """Normal extremity (femur cross-section): cortical bone + marrow + muscle.
    
    Small limb cross-section with bone ring and marrow cavity.
    """
    arr = np.full((512, 512), -1024.0, dtype=np.float32)
    # Muscle
    muscle_mask = make_circle((512, 512), (256, 256), 80, 50.0, arr)
    add_noise(arr, muscle_mask, 50, 12)
    # Cortical bone ring
    make_ring((512, 512), (256, 256), (20, 20), (12, 12), 900.0, arr)
    # Medullary cavity (bone marrow)
    marrow_mask = make_circle((512, 512), (256, 256), 12, 30.0, arr)
    add_noise(arr, marrow_mask, 30, 8)
    return arr


def gen_contrast_enhanced_abdomen():
    """Contrast-enhanced abdomen: enhanced vessels and kidneys (150-250 HU)."""
    arr = gen_normal_abdomen()
    # Aorta (enhanced)
    make_circle((512, 512), (300, 256), 12, 200.0, arr)
    # Kidney cortices (enhanced)
    make_ellipse((512, 512), (270, 350), (25, 20), 180.0, arr)
    make_ellipse((512, 512), (270, 160), (25, 20), 170.0, arr)
    return arr


def gen_spine_only():
    """Spine-focused CT: vertebral body + cortical shell + spinal canal."""
    arr = np.full((512, 512), -1024.0, dtype=np.float32)
    # Paraspinal muscles
    body_mask = make_ellipse((512, 512), (256, 256), (150, 120), 45.0, arr)
    add_noise(arr, body_mask, 45, 12)
    # Vertebral body (all bone at uniform density so threshold catches it)
    make_ellipse((512, 512), (300, 256), (30, 25), 900.0, arr)
    # Spinal canal (CSF)
    canal_mask = make_circle((512, 512), (310, 256), 8, 10.0, arr)
    add_noise(arr, canal_mask, 10, 3)
    # Spinous process (same bone density)
    make_rect(arr, 325, 370, 252, 260, 900.0)
    return arr


def gen_completely_uniform():
    """Perfectly uniform slice -- should never flag anything."""
    arr = np.full((512, 512), -1024.0, dtype=np.float32)
    body_mask = make_ellipse((512, 512), (256, 256), (200, 200), 40.0, arr)
    add_noise(arr, body_mask, 40, 5)
    return arr


def gen_high_noise_slice():
    """High-noise (low-dose) reconstruction -- noise should NOT flag."""
    arr = np.full((512, 512), -1024.0, dtype=np.float32)
    body_mask = make_ellipse((512, 512), (256, 256), (200, 200), 40.0, arr)
    add_noise(arr, body_mask, 40, 40)
    return arr


def gen_chest_with_trachea():
    """Upper chest with prominent trachea + main bronchi.
    
    All airways should be suppressed as expected chest anatomy.
    """
    arr = np.full((512, 512), -1024.0, dtype=np.float32)
    # Body
    body_mask = make_ellipse((512, 512), (270, 256), (190, 210), 30.0, arr)
    add_noise(arr, body_mask, 30, 18)
    # Large bilateral lungs (> 25% body area for chest auto-detection)
    ll_mask = make_ellipse((512, 512), (240, 160), (130, 75), -850.0, arr)
    add_noise(arr, ll_mask, -850, 25)
    rl_mask = make_ellipse((512, 512), (240, 350), (130, 75), -850.0, arr)
    add_noise(arr, rl_mask, -850, 25)
    # Trachea
    make_circle((512, 512), (260, 256), 12, -950.0, arr)
    # Main bronchi (smaller, branching left and right)
    make_circle((512, 512), (265, 235), 7, -920.0, arr)
    make_circle((512, 512), (265, 277), 7, -920.0, arr)
    # Spine
    make_ellipse((512, 512), (380, 256), (18, 16), 600.0, arr)
    return arr


def gen_abdomen_multiple_bowel_gas():
    """Abdomen with many (10+) small bowel gas pockets -- all should be suppressed."""
    arr = gen_normal_abdomen()
    gas_positions = [
        (200, 220, 6), (220, 300, 7), (250, 190, 5), (280, 330, 8),
        (300, 200, 4), (310, 270, 6), (320, 310, 5), (260, 240, 7),
        (340, 280, 4), (250, 350, 5), (290, 180, 6),
    ]
    for cy, cx, r in gas_positions:
        make_circle((512, 512), (cy, cx), r, np.random.uniform(-400, -250), arr)
    return arr


# --- Test Cases ---------------------------------------------------------------

class TestCase:
    def __init__(self, name, generator, hint, expected_action, expected_findings_range,
                 description=""):
        self.name = name
        self.generator = generator
        self.hint = hint
        self.expected_action = expected_action
        self.expected_findings_range = expected_findings_range  # (min, max) 
        self.description = description


TEST_CASES = [
    # --- Clean / Normal anatomy (should NOT flag) ---
    TestCase("Normal Brain", gen_normal_head, "HEAD",
             "CONTINUE", (0, 0),
             "Uniform brain with skull ring and CSF. Nothing abnormal."),
    
    TestCase("Normal Chest (with hint)", gen_normal_chest, "CHEST",
             "CONTINUE", (0, 0),
             "Normal bilateral lungs, heart, spine. All anatomy expected."),
    
    TestCase("Normal Chest (auto-detect)", gen_normal_chest, "",
             "CONTINUE", (0, 0),
             "Same chest but without DICOM hint. Lung fraction > 25% auto-detects."),
    
    TestCase("Normal Abdomen", gen_normal_abdomen, "ABDOMEN",
             "CONTINUE", (0, 3),
             "Normal organs with 3 small bowel gas pockets. Gas should be suppressed."),
    
    TestCase("Normal Pelvis", gen_normal_pelvis, "PELVIS",
             "CONTINUE", (0, 0),
             "Pelvic bones, bladder, rectal gas. All expected anatomy."),
    
    TestCase("Normal Neck", gen_neck, "",
             "CONTINUE", (0, 0),
             "Trachea, spine, pharynx. Airway is expected."),
    
    TestCase("Normal Extremity", gen_extremity, "EXTREMITY",
             "CONTINUE", (0, 0),
             "Femur cross-section. Bone + muscle + marrow."),
    
    TestCase("Normal Spine", gen_spine_only, "SPINE",
             "CONTINUE", (0, 1),
             "Vertebral body with cortical shell and spinal canal."),
    
    TestCase("Clean Uniform", gen_completely_uniform, "",
             "CONTINUE", (0, 0),
             "Perfectly uniform tissue."),
    
    TestCase("High Noise Slice", gen_high_noise_slice, "",
             "CONTINUE", (0, 2),
             "Very noisy reconstruction. Noise should NOT cause false positives."),
    
    TestCase("Upper Chest + Trachea", gen_chest_with_trachea, "CHEST",
             "CONTINUE", (0, 0),
             "Prominent trachea + bronchi. All should be suppressed in chest."),
    
    TestCase("Contrast Abdomen", gen_contrast_enhanced_abdomen, "ABDOMEN",
             "CONTINUE", (0, 2),
             "Contrast-enhanced scan. Enhanced vessels are expected."),
    
    TestCase("Abdomen Many Gas Pockets", gen_abdomen_multiple_bowel_gas, "ABDOMEN",
             "CONTINUE", (0, 3),
             "11+ small bowel gas pockets. All should be suppressed."),
    
    # --- Pathological (SHOULD detect) ---
    TestCase("Brain Hemorrhage", gen_head_hemorrhage, "HEAD",
             "WATCH", (1, 3),
             "Acute ICH (~72 HU, ~2000 px). Should be detected as hyperdense."),
    
    TestCase("Massive Brain Bleed", gen_head_massive_hemorrhage, "HEAD",
             "URGENT REVIEW", (1, 3),
             "Large ICH (~78 HU, ~7000 px). Should trigger urgent."),
    
    TestCase("Pneumoperitoneum", gen_abdomen_with_free_air, "ABDOMEN",
             "URGENT REVIEW", (1, 5),
             "Free intraperitoneal air (<-800 HU). Large free air triggers urgent."),
    
    TestCase("Pneumothorax", gen_chest_with_pneumothorax, "CHEST",
             "WATCH", (1, 5),
             "Right-sided PTX crescent, separated from lung by pleural line."),
]


# --- Test Runner --------------------------------------------------------------

def run_all_tests():
    np.random.seed(42)
    
    passed = 0
    failed = 0
    results = []
    
    print("=" * 90)
    print("  PRISM Triage Engine -- Synthetic CT Test Suite")
    print("=" * 90)
    print()
    
    for tc in TEST_CASES:
        np.random.seed(42)  # reproducible per test
        hu_array = tc.generator()
        result = screen_slice(hu_array, slice_id=0, body_part_hint=tc.hint)
        
        # Check action
        action_ok = result.action == tc.expected_action
        # Check findings count
        n_findings = len(result.findings)
        findings_ok = tc.expected_findings_range[0] <= n_findings <= tc.expected_findings_range[1]
        
        ok = action_ok and findings_ok
        status = "PASS" if ok else "FAIL"
        
        if ok:
            passed += 1
        else:
            failed += 1
        
        icon = "[OK]" if ok else "[XX]"
        print(f"  {icon} [{status}] {tc.name}")
        print(f"       Region: {result.body_region} | Scan: {result.scan_type}")
        print(f"       Findings: {n_findings} (expected {tc.expected_findings_range[0]}-{tc.expected_findings_range[1]})"
              f" | Score: {result.emergency_score} | Action: {result.action} (expected {tc.expected_action})")
        
        if result.findings:
            for f in result.findings:
                sev = f.severity_score
                conf = f.confidence
                print(f"         -> {f.anomaly_type}: area={f.area}, HU={f.mean_hu}, "
                      f"conf={conf:.2f}, sev={sev:.2f}")
        
        if not ok:
            reasons = []
            if not action_ok:
                reasons.append(f"ACTION: got {result.action}, expected {tc.expected_action}")
            if not findings_ok:
                reasons.append(f"FINDINGS: got {n_findings}, expected {tc.expected_findings_range}")
            print(f"       ** FAILURE: {'; '.join(reasons)}")
        
        print()
        results.append((tc.name, ok, result))
    
    print("=" * 90)
    print(f"  Results: {passed} passed, {failed} failed, {len(TEST_CASES)} total")
    print("=" * 90)
    
    return results, passed, failed


if __name__ == "__main__":
    results, passed, failed = run_all_tests()
    exit(0 if failed == 0 else 1)
