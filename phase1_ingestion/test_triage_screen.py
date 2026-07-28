"""
test_triage_screen.py — Comprehensive Fake CT Slice Test Suite

Generates synthetic 512x512 HU arrays that approximate real CT anatomy
and validates the triage engine against expected outcomes.

Usage:
    python -m phase1_ingestion.test_triage_screen
"""

import numpy as np
from phase1_ingestion.triage_screen import screen_slice


# ─── Helpers ──────────────────────────────────────────────────────────────────

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


# ─── Fake CT Generators ──────────────────────────────────────────────────────

def gen_normal_head():
    """Normal head CT: skull ring + uniform brain parenchyma + CSF ventricles."""
    arr = np.full((512, 512), -1024.0, dtype=np.float32)  # air outside
    # Skull ring: outer ellipse bone, inner ellipse brain
    make_ellipse((512, 512), (256, 256), (200, 180), 800.0, arr)  # bone
    make_ellipse((512, 512), (256, 256), (180, 160), 35.0, arr)   # brain parenchyma
    # Add gaussian noise to brain
    brain_mask = arr == 35.0
    arr[brain_mask] = np.random.normal(35, 8, np.sum(brain_mask)).astype(np.float32)
    # CSF ventricles (small, central)
    make_ellipse((512, 512), (250, 256), (20, 35), 8.0, arr)
    return arr


def gen_head_hemorrhage():
    """Head CT with acute intracerebral hemorrhage (60-80 HU, ~40x40 px)."""
    arr = gen_normal_head()
    # Hemorrhage in right parietal lobe
    make_circle((512, 512), (220, 310), 25, 72.0, arr)
    return arr


def gen_head_massive_hemorrhage():
    """Head CT with massive hemorrhage (100x100 px, midline shift equivalent)."""
    arr = gen_normal_head()
    make_ellipse((512, 512), (230, 300), (50, 45), 78.0, arr)
    return arr


def gen_normal_chest():
    """Normal chest CT: body wall + bilateral lungs + mediastinum + spine."""
    arr = np.full((512, 512), -1024.0, dtype=np.float32)  # air outside
    # Body wall (elliptical soft tissue)
    make_ellipse((512, 512), (270, 256), (200, 220), 30.0, arr)
    body_mask = arr == 30.0
    arr[body_mask] = np.random.normal(30, 20, np.sum(body_mask)).astype(np.float32)
    
    # Left lung
    make_ellipse((512, 512), (240, 160), (140, 80), -850.0, arr)
    lung_l_mask = arr == -850.0
    arr[lung_l_mask] = np.random.normal(-850, 25, np.sum(lung_l_mask)).astype(np.float32)
    
    # Right lung
    make_ellipse((512, 512), (240, 350), (140, 80), -850.0, arr)
    lung_r_mask = arr == -850.0
    arr[lung_r_mask] = np.random.normal(-850, 25, np.sum(lung_r_mask)).astype(np.float32)
    
    # Spine (posterior midline bone)
    make_ellipse((512, 512), (380, 256), (20, 18), 600.0, arr)
    # Sternum (anterior midline bone)
    make_rect(arr, 80, 110, 250, 262, 500.0)
    # Heart (dense soft tissue, slightly higher HU than body)
    make_ellipse((512, 512), (280, 240), (50, 55), 50.0, arr)
    heart_mask = (arr == 50.0) & (np.abs(np.indices((512,512))[0] - 280) < 50)
    return arr


def gen_chest_with_pneumothorax():
    """Chest CT with right-sided pneumothorax (air in pleural space)."""
    arr = gen_normal_chest()
    # Pneumothorax: thin crescent of air between lung and chest wall (right side)
    # In a real PTX, there is air between the visceral and parietal pleura
    # This should NOT be connected to the lung itself
    yy, xx = np.ogrid[:512, :512]
    outer_mask = ((yy - 240) / 145) ** 2 + ((xx - 350) / 85) ** 2 <= 1
    inner_mask = ((yy - 240) / 130) ** 2 + ((xx - 350) / 70) ** 2 <= 1
    ptx_mask = outer_mask & ~inner_mask & (xx > 350)  # right side crescent
    arr[ptx_mask] = np.random.normal(-950, 10, np.sum(ptx_mask)).astype(np.float32)
    return arr


def gen_normal_abdomen():
    """Normal abdomen CT: body wall + internal organs + bowel gas pockets."""
    arr = np.full((512, 512), -1024.0, dtype=np.float32)
    # Body wall
    make_ellipse((512, 512), (256, 256), (210, 200), 45.0, arr)
    body_mask = arr == 45.0
    arr[body_mask] = np.random.normal(45, 15, np.sum(body_mask)).astype(np.float32)
    
    # Subcutaneous fat ring
    yy, xx = np.ogrid[:512, :512]
    outer = ((yy - 256) / 210) ** 2 + ((xx - 256) / 200) ** 2 <= 1
    inner = ((yy - 256) / 195) ** 2 + ((xx - 256) / 185) ** 2 <= 1
    fat_ring = outer & ~inner
    arr[fat_ring] = np.random.normal(-80, 15, np.sum(fat_ring)).astype(np.float32)
    
    # Liver (right upper quadrant, slightly higher HU)
    make_ellipse((512, 512), (230, 340), (60, 70), 60.0, arr)
    # Spleen (left upper quadrant)
    make_ellipse((512, 512), (230, 160), (35, 40), 55.0, arr)
    # Spine
    make_ellipse((512, 512), (380, 256), (18, 16), 650.0, arr)
    
    # Normal bowel gas pockets (small, scattered)
    make_circle((512, 512), (280, 200), 8, -350.0, arr)
    make_circle((512, 512), (310, 300), 6, -400.0, arr)
    make_circle((512, 512), (260, 280), 5, -300.0, arr)
    return arr


def gen_abdomen_with_free_air():
    """Abdomen CT with pathological free air (pneumoperitoneum).
    Free air rises to non-dependent areas and forms large, irregular crescents."""
    arr = gen_normal_abdomen()
    # Large crescent of free air under anterior abdominal wall
    make_rect(arr, 55, 75, 180, 330, -900.0)
    # Additional irregular collection
    make_ellipse((512, 512), (80, 256), (15, 60), -920.0, arr)
    return arr


def gen_normal_pelvis():
    """Normal pelvis CT: pelvic bones + bladder + rectum + soft tissue."""
    arr = np.full((512, 512), -1024.0, dtype=np.float32)
    # Body
    make_ellipse((512, 512), (256, 256), (200, 220), 40.0, arr)
    body_mask = arr == 40.0
    arr[body_mask] = np.random.normal(40, 12, np.sum(body_mask)).astype(np.float32)
    
    # Pelvic bones (bilateral iliac crests)
    make_ellipse((512, 512), (256, 130), (80, 30), 700.0, arr)
    make_ellipse((512, 512), (256, 380), (80, 30), 700.0, arr)
    # Sacrum
    make_ellipse((512, 512), (360, 256), (25, 30), 650.0, arr)
    
    # Bladder (fluid-filled, low HU)
    make_ellipse((512, 512), (230, 256), (40, 45), 15.0, arr)
    # Rectal gas (normal)
    make_circle((512, 512), (310, 256), 12, -300.0, arr)
    return arr


def gen_neck():
    """Normal neck CT: spine + trachea + soft tissue."""
    arr = np.full((512, 512), -1024.0, dtype=np.float32)
    # Small oval body
    make_ellipse((512, 512), (256, 256), (120, 100), 40.0, arr)
    body_mask = arr == 40.0
    arr[body_mask] = np.random.normal(40, 10, np.sum(body_mask)).astype(np.float32)
    
    # Cervical spine
    make_ellipse((512, 512), (320, 256), (15, 14), 700.0, arr)
    # Trachea (central air)
    make_circle((512, 512), (240, 256), 12, -950.0, arr)
    # Pharynx / airway
    make_ellipse((512, 512), (220, 256), (20, 15), -900.0, arr)
    return arr


def gen_extremity():
    """Normal extremity (femur cross-section): bone + muscle + fat."""
    arr = np.full((512, 512), -1024.0, dtype=np.float32)
    # Small circular limb
    make_circle((512, 512), (256, 256), 80, 50.0, arr)
    body_mask = arr == 50.0
    arr[body_mask] = np.random.normal(50, 12, np.sum(body_mask)).astype(np.float32)
    # Cortical bone ring
    make_circle((512, 512), (256, 256), 20, 800.0, arr)
    # Medullary cavity
    make_circle((512, 512), (256, 256), 12, 30.0, arr)
    return arr


def gen_contrast_enhanced_abdomen():
    """Contrast-enhanced abdomen: vessels and organs enhance to 150-250 HU."""
    arr = gen_normal_abdomen()
    # Aorta (enhanced)
    make_circle((512, 512), (300, 256), 12, 200.0, arr)
    # Kidney cortices (enhanced)
    make_ellipse((512, 512), (270, 350), (25, 20), 180.0, arr)
    make_ellipse((512, 512), (270, 160), (25, 20), 170.0, arr)
    return arr


def gen_spine_only():
    """Spine-focused CT: vertebral body + spinal canal + paraspinal muscles."""
    arr = np.full((512, 512), -1024.0, dtype=np.float32)
    # Body outline (smaller FOV)
    make_ellipse((512, 512), (256, 256), (150, 120), 45.0, arr)
    body_mask = arr == 45.0
    arr[body_mask] = np.random.normal(45, 12, np.sum(body_mask)).astype(np.float32)
    # Vertebral body
    make_ellipse((512, 512), (300, 256), (30, 25), 250.0, arr)
    # Cortical shell
    yy, xx = np.ogrid[:512, :512]
    outer = ((yy - 300) / 32) ** 2 + ((xx - 256) / 27) ** 2 <= 1
    inner = ((yy - 300) / 28) ** 2 + ((xx - 256) / 23) ** 2 <= 1
    cortical = outer & ~inner
    arr[cortical] = 800.0
    # Spinal canal
    make_circle((512, 512), (310, 256), 8, 10.0, arr)
    # Spinous process
    make_rect(arr, 325, 370, 252, 260, 700.0)
    return arr


def gen_completely_uniform():
    """Perfectly uniform slice — should never flag anything."""
    arr = np.full((512, 512), -1024.0, dtype=np.float32)
    make_ellipse((512, 512), (256, 256), (200, 200), 40.0, arr)
    body_mask = arr == 40.0
    arr[body_mask] = np.random.normal(40, 5, np.sum(body_mask)).astype(np.float32)
    return arr


def gen_high_noise_slice():
    """High-noise (low-dose) reconstruction — noise should NOT flag."""
    arr = np.full((512, 512), -1024.0, dtype=np.float32)
    make_ellipse((512, 512), (256, 256), (200, 200), 40.0, arr)
    body_mask = arr == 40.0
    # Much higher noise (std=40 instead of typical 10-15)
    arr[body_mask] = np.random.normal(40, 40, np.sum(body_mask)).astype(np.float32)
    return arr


def gen_chest_with_trachea():
    """Upper chest with prominent trachea — trachea should be suppressed."""
    arr = np.full((512, 512), -1024.0, dtype=np.float32)
    # Body
    make_ellipse((512, 512), (270, 256), (190, 210), 30.0, arr)
    body_mask = arr == 30.0
    arr[body_mask] = np.random.normal(30, 18, np.sum(body_mask)).astype(np.float32)
    # Large bilateral lungs
    make_ellipse((512, 512), (240, 160), (130, 75), -850.0, arr)
    lung_mask = arr == -850.0
    arr[lung_mask] = np.random.normal(-850, 25, np.sum(lung_mask)).astype(np.float32)
    make_ellipse((512, 512), (240, 350), (130, 75), -850.0, arr)
    lung_mask2 = arr == -850.0
    arr[lung_mask2] = np.random.normal(-850, 25, np.sum(lung_mask2)).astype(np.float32)
    # Prominent trachea
    make_circle((512, 512), (260, 256), 15, -950.0, arr)
    # Main bronchi
    make_circle((512, 512), (265, 230), 8, -920.0, arr)
    make_circle((512, 512), (265, 282), 8, -920.0, arr)
    # Spine
    make_ellipse((512, 512), (380, 256), (18, 16), 600.0, arr)
    return arr


def gen_abdomen_multiple_bowel_gas():
    """Abdomen with many (10+) small bowel gas pockets — all should be suppressed."""
    arr = gen_normal_abdomen()
    # Scatter many small gas pockets
    gas_positions = [
        (200, 220, 6), (220, 300, 7), (250, 190, 5), (280, 330, 8),
        (300, 200, 4), (310, 270, 6), (320, 310, 5), (260, 240, 7),
        (340, 280, 4), (250, 350, 5), (290, 180, 6),
    ]
    for cy, cx, r in gas_positions:
        make_circle((512, 512), (cy, cx), r, np.random.uniform(-400, -250), arr)
    return arr


# ─── Test Runner ──────────────────────────────────────────────────────────────

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
    # ─── Clean / Normal ───
    TestCase("Normal Brain", gen_normal_head, "HEAD",
             "CONTINUE", (0, 0),
             "Uniform brain with skull ring and CSF. Nothing abnormal."),
    
    TestCase("Normal Chest (with hint)", gen_normal_chest, "CHEST",
             "CONTINUE", (0, 0),
             "Normal bilateral lungs, heart, spine. All anatomy expected."),
    
    TestCase("Normal Chest (auto-detect)", gen_normal_chest, "",
             "CONTINUE", (0, 0),
             "Same as above but without DICOM hint. Lung fraction > 25% should auto-detect."),
    
    TestCase("Normal Abdomen", gen_normal_abdomen, "ABDOMEN",
             "CONTINUE", (0, 3),
             "Normal organs with 3 small bowel gas pockets. Gas should be suppressed."),
    
    TestCase("Normal Pelvis", gen_normal_pelvis, "PELVIS",
             "CONTINUE", (0, 0),
             "Pelvic bones, bladder, rectal gas. All expected anatomy."),
    
    TestCase("Normal Neck", gen_neck, "",
             "CONTINUE", (0, 0),
             "Trachea, spine, pharynx. Airway is expected here."),
    
    TestCase("Normal Extremity", gen_extremity, "EXTREMITY",
             "CONTINUE", (0, 0),
             "Femur cross-section. Bone + muscle + marrow."),
    
    TestCase("Normal Spine", gen_spine_only, "SPINE",
             "CONTINUE", (0, 0),
             "Vertebral body with cortical shell and spinal canal."),
    
    TestCase("Clean Uniform", gen_completely_uniform, "",
             "CONTINUE", (0, 0),
             "Perfectly uniform tissue — nothing to detect."),
    
    TestCase("High Noise Slice", gen_high_noise_slice, "",
             "CONTINUE", (0, 2),
             "Very noisy reconstruction. Noise should NOT cause false positives."),
    
    TestCase("Upper Chest + Trachea", gen_chest_with_trachea, "CHEST",
             "CONTINUE", (0, 0),
             "Prominent trachea + bronchi. All should be suppressed in chest."),
    
    TestCase("Contrast Abdomen", gen_contrast_enhanced_abdomen, "ABDOMEN",
             "CONTINUE", (0, 2),
             "Contrast-enhanced scan. Enhanced vessels are expected, not pathological."),
    
    TestCase("Abdomen Many Gas Pockets", gen_abdomen_multiple_bowel_gas, "ABDOMEN",
             "CONTINUE", (0, 3),
             "11+ small bowel gas pockets. All should be suppressed."),
    
    # ─── Pathological (should detect) ───
    TestCase("Brain Hemorrhage", gen_head_hemorrhage, "HEAD",
             "WATCH", (1, 3),
             "Acute ICH (~72 HU, ~2000 px). Should be detected as hyperdense."),
    
    TestCase("Massive Brain Bleed", gen_head_massive_hemorrhage, "HEAD",
             "URGENT REVIEW", (1, 3),
             "Large ICH (~78 HU, ~7000 px). Should trigger urgent."),
    
    TestCase("Pneumoperitoneum", gen_abdomen_with_free_air, "ABDOMEN",
             "WATCH", (1, 5),
             "Free intraperitoneal air. Large crescent should be detected."),
    
    # The pneumothorax test is interesting because the false PTX finding
    # crescent is NOT connected to the lung cavity 
    TestCase("Pneumothorax", gen_chest_with_pneumothorax, "CHEST",
             "WATCH", (1, 5),
             "Right-sided PTX crescent. Should survive lung suppression."),
]


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
        
        # Print result
        icon = "[OK]" if ok else "[XX]"
        print(f"  {icon} [{status}] {tc.name}")
        print(f"       Region: {result.body_region} | Scan: {result.scan_type}")
        print(f"       Findings: {n_findings} (expected {tc.expected_findings_range[0]}-{tc.expected_findings_range[1]})"
              f" | Score: {result.emergency_score} | Action: {result.action} (expected {tc.expected_action})")
        
        if result.findings:
            for f in result.findings:
                print(f"         -> {f.anomaly_type}: area={f.area}, HU={f.mean_hu:.0f}, "
                      f"conf={f.confidence}, sev={f.severity_score}")
        
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
