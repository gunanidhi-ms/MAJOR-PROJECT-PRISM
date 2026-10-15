import os
import time
import argparse
import logging
from collections import defaultdict
from glob import glob

try:
    from pynetdicom import AE
    from pynetdicom.sop_class import CTImageStorage
    import pydicom
except ImportError:
    raise ImportError("pynetdicom and pydicom are required.")

logger = logging.getLogger(__name__)


def build_patient_registry(dicom_dir: str) -> dict:

    print(f"[{time.strftime('%H:%M:%S')}] Scanning directory: {dicom_dir}")
    
    dcm_files = []
    for root, _, files in os.walk(dicom_dir):
        for f in files:
            # TCIA files often have no extension, or use .dcm / .ima
            if f.lower().endswith('.dcm') or f.lower().endswith('.ima') or '.' not in f:
                dcm_files.append(os.path.join(root, f))
                
    if not dcm_files:
        print("[ERROR] No .dcm files found.")
        return {}

    print(f"[{time.strftime('%H:%M:%S')}] Found {len(dcm_files)} DICOM files. Indexing metadata...")
    
    # Structure: registry[patient_id][study_uid][series_uid] = [(instance_num, path), ...]
    registry = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    ct_count = 0
    skip_count = 0
    
    for path in dcm_files:
        try:
            # Read only headers to index quickly
            ds = pydicom.dcmread(path, force=True, stop_before_pixels=True)
            
            modality = getattr(ds, "Modality", "Unknown")
            if modality != "CT":
                skip_count += 1
                continue
                
            patient_id = getattr(ds, "PatientID", "UnknownPatient")
            study_uid = getattr(ds, "StudyInstanceUID", "UnknownStudy")
            series_uid = getattr(ds, "SeriesInstanceUID", "UnknownSeries")
            instance_num = int(getattr(ds, "InstanceNumber", 0))
            
            registry[patient_id][study_uid][series_uid].append({
                "instance_num": instance_num,
                "filepath": path
            })
            ct_count += 1
            
            if ct_count % 5000 == 0:
                print(f"  ... Indexed {ct_count} CT slices ...")
                
        except Exception:
            skip_count += 1
            
    print(f"[{time.strftime('%H:%M:%S')}] Indexing complete. Found {ct_count} CT slices (skipped {skip_count} non-CT).")
    return registry


def format_dicom_age(age_str: str) -> str:
    """Formats age string into DICOM standard 4-character VR AS (e.g. '21' -> '021Y', '45Y' -> '045Y')."""
    if not age_str:
        return ""
    clean = age_str.strip().upper()
    unit = "Y"
    if clean.endswith(("Y", "M", "W", "D")):
        unit = clean[-1]
        digits = clean[:-1]
    else:
        digits = clean
    try:
        val = int(digits)
        return f"{val:03d}{unit}"
    except ValueError:
        return clean[:4]

def simulate_ct_scan(
    patient_id: str,
    study_dict: dict,
    host: str = "127.0.0.1",
    port: int = 11112,
    delay_ms: int = 30,
    inject_id: str = "",
    inject_name: str = "",
    inject_age: str = "",
    inject_sex: str = "",
):
    """
    Simulates a live CT scan for a specific patient.
    Sends slices strictly in order of InstanceNumber.
    """
    active_patient_id = inject_id if inject_id else patient_id
    dicom_age = format_dicom_age(inject_age)
    print(f"\n=======================================================")
    print(f"[Live CT/MRI Scanner]")
    print(f"| (Continuous C-STORE Stream via Port 11112)")
    print(f"=======================================================")
    print(f"  Patient ID:   {active_patient_id}")
    if inject_name: print(f"  Patient Name: {inject_name}")
    if inject_age:  print(f"  Patient Age:  {dicom_age}")
    if inject_sex:  print(f"  Patient Sex:  {inject_sex}")
    print(f"  Target:       {host}:{port}")
    print(f"  Scan Speed:   {delay_ms}ms per slice")
    print(f"=======================================================\n")
    
    ae = AE(ae_title="CT_SCANNER")
    ae.add_requested_context(CTImageStorage)
    ae.add_requested_context("1.2.840.10008.5.1.4.1.1.2") # CT Image Storage explicitly
    
    assoc = ae.associate(host, port, ae_title="PRISM_SCP")
    if not assoc.is_established:
        print(f"[ERROR] Failed to connect to DICOM listener at {host}:{port}")
        return

    try:
        total_sent = 0
        
        for study_uid, series_dict in study_dict.items():
            print(f"  Study: {study_uid}")
            
            for series_uid, slices in series_dict.items():
                # SORT slices by InstanceNumber!
                slices.sort(key=lambda x: x["instance_num"])
                
                print(f"  +-- Series: {series_uid} ({len(slices)} slices)")
                
                for i, slice_meta in enumerate(slices):
                    dcm_path = slice_meta["filepath"]
                    inst_num = slice_meta["instance_num"]
                    
                    try:
                        # Full read including pixels for sending
                        ds = pydicom.dcmread(dcm_path, force=True)
                        
                        # INJECT custom patient metadata
                        if inject_id:
                            ds.PatientID = inject_id
                        if inject_name:
                            ds.PatientName = inject_name
                        if dicom_age:
                            ds.PatientAge = dicom_age
                        if inject_sex:
                            ds.PatientSex = inject_sex
                            
                        status = assoc.send_c_store(ds)
                        
                        if status and status.Status == 0x0000:
                            print(f"  |  [OK] Sent Slice {inst_num:3d} ({i+1}/{len(slices)})")
                            total_sent += 1
                        else:
                            print(f"  |  [FAIL] Slice {inst_num:3d} (Status: {status.Status if status else 'None'})")
                            
                        # Simulate physical machine rotation/scan delay
                        time.sleep(delay_ms / 1000.0)
                        
                    except Exception as e:
                        print(f"  |  [ERROR] Slice {inst_num:3d}: {e}")
                        
    finally:
        assoc.release()
        print(f"\n[DONE] Scan complete. {total_sent} CT slices sent for patient {active_patient_id}.")


def main():
    parser = argparse.ArgumentParser(description="CT Machine Emulator")
    parser.add_argument("--dir", required=True, help="Path to TCIA or dataset directory")
    parser.add_argument("--host", default="127.0.0.1", help="SCP Host")
    parser.add_argument("--port", type=int, default=11112, help="SCP Port")
    parser.add_argument("--delay", type=int, default=100, help="Delay between slices in ms")
    parser.add_argument("--patient", type=str, default="", help="Specific Patient ID to scan")
    parser.add_argument("--name", type=str, default="", help="Override Patient Name")
    parser.add_argument("--age", type=str, default="", help="Override Patient Age (e.g. 45Y)")
    parser.add_argument("--sex", type=str, default="", help="Override Patient Sex (M/F/O)")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    # 1. Index the dataset
    registry = build_patient_registry(args.dir)
    
    if not registry:
        return
        
    patients = list(registry.keys())
    
    # 2. Select patients to scan
    patients_to_scan = patients
    if args.patient and args.patient in registry:
        patients_to_scan = [args.patient]
        
    # 3. Run the scan
    for patient_id in patients_to_scan:
        simulate_ct_scan(
            patient_id=patient_id,
            study_dict=registry[patient_id],
            host=args.host,
            port=args.port,
            delay_ms=args.delay,
            inject_id=args.patient,
            inject_name=args.name,
            inject_age=args.age,
            inject_sex=args.sex,
        )


if __name__ == "__main__":
    main()
