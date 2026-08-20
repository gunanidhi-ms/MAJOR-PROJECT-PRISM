# contrast_detection.py
import numpy as np

def detect_contrast_from_voxels(aorta_voxels=None, liver_voxels=None, aorta_mean_hu=None, liver_mean_hu=None):
    """
    Detect contrast directly from voxel values (either from raw voxels or pre-calculated means).
    """
    if aorta_mean_hu is None and aorta_voxels is not None:
        aorta_mean_hu = np.mean(aorta_voxels)
    if liver_mean_hu is None and liver_voxels is not None:
        liver_mean_hu = np.mean(liver_voxels)
        
    if aorta_mean_hu is None or liver_mean_hu is None:
        return "UNKNOWN"
        
    # Non-Contrast CT: Unenhanced blood (30-50 HU) & Unenhanced Liver (40-70 HU)
    if aorta_mean_hu < 70 and liver_mean_hu < 75:
        return "NON_CONTRAST"
    
    # Contrast CT (Arterial Phase): Aorta is extremely bright (>200 HU)
    elif aorta_mean_hu >= 200 and liver_mean_hu < 90:
        return "CONTRAST_ARTERIAL"
    
    # Contrast CT (Portal Venous Phase): Liver Parenchyma enhances (>90-100 HU)
    elif liver_mean_hu >= 90 or (120 <= aorta_mean_hu <= 250):
        return "CONTRAST_PORTAL_VENOUS"
        
    elif aorta_mean_hu < 120 and liver_mean_hu < 80:
        return "DELAYED_OR_LOW_ENHANCEMENT"
    
    return "CONTRAST_PORTAL_VENOUS" # Default fallback for general CECT
