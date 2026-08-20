"""
ct_machine_emulator.py — Clinical CT Scanner Emulator

This script acts as a real CT machine for Phase 1 testing. 
Instead of sending random slices from a folder, it:
  1. Scans a dataset directory (like TCIA).
  2. Groups all files logically by Patient ID -> Study UID -> Series UID.
  3. Sorts the slices within a series by their exact InstanceNumber.
  4. Connects to our DICOM Listener (SCP) and streams the slices in order,
     simulating the precise timing and sequence of a live hospital CT scan.

Usage:
    # Run the ingestion pipeline first in another terminal:
    python -m phase1_ingestion.pipeline

    # Then run the emulator:
    python -m phase1_ingestion.ct_machine_emulator --dir "path/to/dataset"
"""

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
    """
    Scans the directory and builds a hierarchical registry:
    PatientID -> StudyInstanceUID -> SeriesInstanceUID -> list of dicts(InstanceNumber, filepath)
    """
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
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            
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


def simulate_ct_scan(
    patient_id: str,
    study_dict: dict,
    host: str = "127.0.0.1",
    port: int = 11112,
    delay_ms: int =  30
):
    """
    Simulates a live CT scan for a specific patient.
    Sends slices strictly in order of InstanceNumber.
    """
    print(f"\n=======================================================")
    print(f"[Live CT/MRI Scanner]")
    print(f"| (Continuous C-STORE Stream via Port 11112)")
    print(f"=======================================================")
    print(f"  Patient ID: {patient_id}")
    print(f"  Target:     {host}:{port}")
    print(f"  Scan Speed: {delay_ms}ms per slice")
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
                        ds = pydicom.dcmread(dcm_path)
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
        print(f"\n[DONE] Scan complete. {total_sent} CT slices sent for patient {patient_id}.")


def main():
    parser = argparse.ArgumentParser(description="CT Machine Emulator")
    parser.add_argument("--dir", required=True, help="Path to TCIA or dataset directory")
    parser.add_argument("--host", default="127.0.0.1", help="SCP Host")
    parser.add_argument("--port", type=int, default=11112, help="SCP Port")
    parser.add_argument("--delay", type=int, default=100, help="Delay between slices in ms")
    parser.add_argument("--patient", type=str, default="", help="Specific Patient ID to scan")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    # 1. Index the dataset
    registry = build_patient_registry(args.dir)
    
    if not registry:
        return
        
    patients = list(registry.keys())
    
    # 2. Select patients to scan
    patients_to_scan = []
    if args.patient:
        if args.patient in registry:
            patients_to_scan = [args.patient]
        else:
            print(f"[WARN] Patient '{args.patient}' not found in the directory.")
            return
    else:
        patients_to_scan = patients
        print(f"\nFound {len(patients)} patient(s) in the dataset. Proceeding with all of them.")
        
    # 3. Run the scan
    for patient_id in patients_to_scan:
        simulate_ct_scan(
            patient_id=patient_id,
            study_dict=registry[patient_id],
            host=args.host,
            port=args.port,
            delay_ms=args.delay
        )


if __name__ == "__main__":
    main()
