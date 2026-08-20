# hu_reference_table.py
# These bands are literature-derived triage heuristics — validate against your
# annotated data before treating them as final. Unmapped structures or ambiguous
# readings should default to ORANGE, never GREEN.

HU_REFERENCE_TABLE = {
    # ---- NEURO ----
    "brain_gray_matter":  {"region": "neuro", "normal": (37, 45),
                            "mild": [(30, 37), (45, 50)], "severe": [(None, 30), (50, None)],
                            "signal": "density"},
    "brain_white_matter": {"region": "neuro", "normal": (20, 30),
                            "mild": [(15, 20), (30, 35)], "severe": [(None, 15), (35, None)],
                            "signal": "density"},
    "csf_ventricle":      {"region": "neuro", "normal": (0, 15),
                            "signal": "shape", "note": "flag on volume/asymmetry only"},

    # ---- CARDIOTHORACIC ----
    "lung_parenchyma":    {"region": "cardiothoracic", "normal": (-950, -700),
                            "mild": [(-700, -300)], "severe": [(-300, 100)], "signal": "density"},
    "mediastinal_tissue": {"region": "cardiothoracic", "normal": (35, 55),
                            "mild": [(20, 35), (55, 70)], "severe": [(None, 20), (70, None)],
                            "signal": "density"},
    "pleural_space":      {"region": "cardiothoracic", "normal": (None, None),
                            "mild": [(0, 20)], "severe": [(35, 70)], "signal": "density+volume"},

    # ---- ABDOMEN/PELVIS ----
    "liver":              {"region": "abdomen", "normal": (40, 70),
                            "mild": [(25, 40)], "severe": [(None, 25), (100, None)], "signal": "density"},
    "spleen":             {"region": "abdomen", "normal": (40, 60),
                            "mild": [(25, 40), (60, 75)], "severe": [(None, 25), (75, None)], "signal": "density"},
    "kidney_cortex":      {"region": "abdomen_kub", "normal": (30, 45),
                            "mild": [(20, 30)], "severe": [(150, None)], "signal": "density+location"},
    "peritoneal_fluid":   {"region": "abdomen", "normal": (None, None),
                            "mild": [(0, 20)], "severe": [(30, 100)], "signal": "density+context"},

    # ---- KUB ----
    "renal_collecting_system": {"region": "kub", "normal": (0, 20),
                                 "severe": [(200, None)], "signal": "density"},
    "bladder_lumen":      {"region": "kub", "normal": (0, 20),
                            "signal": "shape", "note": "dilation flagged structurally, not by HU"},

    # ---- ORTHO/SPINE ----
    "cortical_bone":      {"region": "ortho_spine", "normal": (1000, 1900),
                            "mild": [(700, 1000)], "severe": [(None, 700)], "signal": "density"},
    "cancellous_bone":    {"region": "ortho_spine", "normal": (300, 400),
                            "mild": [(150, 300)], "severe": [(None, 150), (600, None)], "signal": "density"},
    "bone_marrow":        {"region": "ortho", "normal": (-100, -30),
                            "mild": [(-30, 0)], "severe": [(30, None)], "signal": "density"},
    "intervertebral_disc": {"region": "spine", "normal": (40, 90),
                             "mild": [(20, 40)], "signal": "density+shape"},
    "epidural_space":     {"region": "spine", "normal": (0, 15),
                            "severe": [(50, 90)], "signal": "density"},
}

# Adjusted Reference Table (Portal Venous Phase Baseline) for CECT
HU_REFERENCE_TABLE_CECT = {
    "liver":              {"region": "abdomen", "normal": (100, 150),
                            "mild": [(70, 100), (150, 180)], "severe": [(None, 70), (200, None)], "signal": "density"},
    "spleen":             {"region": "abdomen", "normal": (110, 140),
                            "mild": [(80, 110)], "severe": [(None, 80), (160, None)], "signal": "density"},
    "kidney_cortex":      {"region": "abdomen_kub", "normal": (130, 180),
                            "mild": [(90, 130)], "severe": [(None, 90)], "signal": "density+location"},
    "pancreas":           {"region": "abdomen", "normal": (80, 120),
                            "mild": [(50, 80)], "severe": [(None, 50)], "signal": "density"},
    "aorta":              {"region": "vascular", "normal": (150, 300),
                            "mild": [(100, 150)], "severe": [(None, 100)], "signal": "density"},
    "bowel_wall":         {"region": "abdomen", "normal": (60, 90),
                            "mild": [(40, 60)], "severe": [(None, 40)], "signal": "density"}
}

REGION_ORGAN_MAP = {
    "neuro":         ["brain", "brainstem", "cerebellum", "ventricle", "brain_gray_matter", "brain_white_matter", "csf_ventricle"],
    "cardiothoracic": ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
                        "lung_lower_lobe_right", "heart", "aorta", "pulmonary_artery", "lung_parenchyma", "mediastinal_tissue", "pleural_space"],
    "abdomen_pelvis": ["liver", "spleen", "pancreas", "stomach", "kidney_left", "kidney_right",
                        "urinary_bladder", "small_bowel", "colon", "kidney_cortex", "peritoneal_fluid", "bowel_wall"],
    "kub":            ["kidney_left", "kidney_right", "urinary_bladder", "renal_collecting_system", "bladder_lumen"],
    "ortho":          ["femur_left", "femur_right", "humerus_left", "humerus_right",
                        "patella", "tibia", "fibula", "cortical_bone", "cancellous_bone", "bone_marrow"],
    "spine":          ["vertebrae_C1", "vertebrae_L5", "spinal_cord", "intervertebral_disc", "epidural_space"],
}

def detect_region(present_labels):
    scores = {region: len(set(present_labels) & set(organs))
              for region, organs in REGION_ORGAN_MAP.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"
