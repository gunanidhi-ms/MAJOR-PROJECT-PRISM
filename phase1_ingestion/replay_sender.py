"""
replay_sender.py — DICOM C-STORE SCU Replay Script

Reads .dcm files from the sample_dicoms/ directory and sends them to
the DICOM listener (SCP) via C-STORE, simulating a live CT scanner stream.

Features:
  - Shuffles slice order to test the sorting buffer
  - Adds random delays (0.2–1.5s) between sends to simulate network jitter
  - Reports send status for each slice
  - Configurable target host/port and delay range

Usage:
    python replay_sender.py                          # defaults
    python replay_sender.py --port 11112 --delay 0.5 # custom
"""

import os
import sys
import time
import random
import argparse
import logging
from pathlib import Path
from glob import glob

try:
    from pynetdicom import AE
    from pynetdicom.sop_class import CTImageStorage
except ImportError:
    raise ImportError("pynetdicom is required: pip install pynetdicom")

try:
    import pydicom
except ImportError:
    raise ImportError("pydicom is required: pip install pydicom")

logger = logging.getLogger(__name__)


def find_dicom_files(directory: str) -> list[str]:
    """Find all .dcm files in a directory, sorted by name."""
    dcm_files = sorted(glob(os.path.join(directory, "*.dcm")))
    if not dcm_files:
        # Also check subdirectories
        dcm_files = sorted(glob(os.path.join(directory, "**", "*.dcm"), recursive=True))
    return dcm_files


def replay_dicoms(
    dicom_dir: str,
    host: str = "127.0.0.1",
    port: int = 11112,
    ae_title: str = "PRISM_SCU",
    called_ae_title: str = "PRISM_SCP",
    min_delay: float = 0.2,
    max_delay: float = 1.5,
    shuffle: bool = True,
    seed: int | None = 42,
) -> dict:
    """
    Replay DICOM files to a C-STORE SCP, simulating a live stream.
    
    Args:
        dicom_dir: Directory containing .dcm files.
        host: SCP hostname (default localhost).
        port: SCP port (default 11112).
        ae_title: This sender's AE title.
        called_ae_title: Target SCP's AE title.
        min_delay: Minimum delay between sends (seconds).
        max_delay: Maximum delay between sends (seconds).
        shuffle: If True, randomize send order to test sorting buffer.
        seed: Random seed for reproducible shuffling.
    
    Returns:
        Dict with send statistics.
    """
    # Find DICOM files
    dcm_files = find_dicom_files(dicom_dir)
    if not dcm_files:
        print(f"❌ No .dcm files found in: {dicom_dir}")
        print("   Run generate_test_dicoms.py first to create test data.")
        return {"sent": 0, "failed": 0, "total": 0}

    print(f"📂 Found {len(dcm_files)} DICOM files in: {dicom_dir}")

    # Shuffle for out-of-order testing
    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(dcm_files)
        print(f"🔀 Shuffled send order (seed={seed})")

    # Set up C-STORE SCU
    ae = AE(ae_title=ae_title)
    ae.add_requested_context(CTImageStorage)
    # Also add common CT-related SOP classes
    ae.add_requested_context("1.2.840.10008.5.1.4.1.1.2")  # CT Image Storage

    stats = {"sent": 0, "failed": 0, "total": len(dcm_files)}

    print(f"\n🚀 Connecting to {called_ae_title}@{host}:{port}...")
    print(f"   Delay range: {min_delay:.1f}s – {max_delay:.1f}s")
    print(f"   {'─' * 50}\n")

    # Establish association
    assoc = ae.associate(host, port, ae_title=called_ae_title)
    if not assoc.is_established:
        print(f"❌ Failed to connect to {host}:{port}")
        print("   Make sure the DICOM listener is running.")
        print("   Start it with: python pipeline.py")
        return stats

    try:
        for i, dcm_path in enumerate(dcm_files):
            try:
                ds = pydicom.dcmread(dcm_path, force=True)
                instance_num = getattr(ds, "InstanceNumber", "?")

                # Filter out non-CT files
                modality = getattr(ds, "Modality", "")
                if modality != "CT":
                    print(f"  ⏭ [{i+1:3d}/{len(dcm_files)}] Skipping non-CT file (Modality={modality})")
                    continue

                # Send the dataset
                status = assoc.send_c_store(ds)

                if status and status.Status == 0x0000:
                    stats["sent"] += 1
                    print(
                        f"  ✓ [{i+1:3d}/{len(dcm_files)}] "
                        f"Sent slice InstanceNumber={instance_num} "
                        f"({os.path.basename(dcm_path)})"
                    )
                else:
                    stats["failed"] += 1
                    status_val = status.Status if status else "None"
                    print(
                        f"  ✗ [{i+1:3d}/{len(dcm_files)}] "
                        f"FAILED slice {instance_num} (status={status_val})"
                    )

                # Random delay to simulate network jitter
                if i < len(dcm_files) - 1:  # no delay after last slice
                    delay = random.uniform(min_delay, max_delay)
                    time.sleep(delay)

            except Exception as e:
                stats["failed"] += 1
                logger.error("Error sending %s: %s", dcm_path, e)
                print(f"  ✗ [{i+1:3d}/{len(dcm_files)}] ERROR: {e}")

    finally:
        assoc.release()

    print(f"\n{'─' * 50}")
    print(f"📊 Replay complete: {stats['sent']} sent, {stats['failed']} failed, "
          f"{stats['total']} total")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Replay DICOM files to a C-STORE SCP"
    )
    parser.add_argument(
        "--dir",
        default=os.path.join(os.path.dirname(__file__), "sample_dicoms"),
        help="Directory containing .dcm files (default: sample_dicoms/)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="SCP host")
    parser.add_argument("--port", type=int, default=11112, help="SCP port")
    parser.add_argument(
        "--min-delay", type=float, default=0.2, help="Min delay between sends (s)"
    )
    parser.add_argument(
        "--max-delay", type=float, default=1.5, help="Max delay between sends (s)"
    )
    parser.add_argument(
        "--no-shuffle", action="store_true", help="Send in order (don't shuffle)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    replay_dicoms(
        dicom_dir=args.dir,
        host=args.host,
        port=args.port,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
        shuffle=not args.no_shuffle,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
