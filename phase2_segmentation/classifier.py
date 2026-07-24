# classifier.py
import numpy as np

def classify_band(mean_hu, ref):
    lo, hi = ref.get("normal", (None, None))
    if lo is not None and hi is not None and lo <= mean_hu <= hi:
        return "GREEN"
    for blo, bhi in ref.get("severe", []):
        if (blo is None or mean_hu >= blo) and (bhi is None or mean_hu <= bhi):
            return "RED"
    for blo, bhi in ref.get("mild", []):
        if (blo is None or mean_hu >= blo) and (bhi is None or mean_hu <= bhi):
            return "ORANGE"
    return "ORANGE"  # unclassified deviation -> caution, never silently pass as GREEN

def classify_structure(organ_label, roi_pixels=None, mean_hu=None, table=None):
    if table is None:
        return {"organ": organ_label, "band": "ORANGE", "note": "No reference table provided"}
    
    ref = table.get(organ_label)
    if ref is None:
        return {"organ": organ_label, "band": "ORANGE",
                "note": "unmapped structure - route to manual review"}
    
    if mean_hu is None and roi_pixels is not None:
        mean_hu = float(np.mean(roi_pixels))
    elif mean_hu is None:
        return {"organ": organ_label, "band": "ORANGE", "note": "No HU data provided"}
        
    return {
        "organ": organ_label,
        "mean_hu": round(mean_hu, 1),
        "band": classify_band(mean_hu, ref),
        "primary_signal": ref["signal"],
    }
