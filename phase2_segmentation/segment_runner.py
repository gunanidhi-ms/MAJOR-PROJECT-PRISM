"""
segment_runner.py — TotalSegmentator Subprocess Wrapper

Runs TotalSegmentator as an isolated subprocess (never as an in-process Python
import) to ensure C++-backed tensor memory is fully released after each run.

Routes:
    CT  → --task total
    MR  → --task total_mr

Flags: --fast --statistics --radiomics --ml
Timeout: 120 seconds hard kill

The output stats dict is transformed to match schemas/organ_statistics.json.

Usage:
    from phase2_segmentation.segment_runner import run_totalsegmentator

    out_dir, stats = run_totalsegmentator("volume.nii.gz", modality="CT")
"""

import os
import sys
import json
import time
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Timeout defaults (overridden at runtime by compute_dynamic_timeout) ────────
# These fallbacks are used only when hardware detection fails completely.
_TIMEOUT_GPU_DEFAULT = 120   # seconds — GPU with enough VRAM
_TIMEOUT_CPU_DEFAULT = 600   # seconds — CPU-only (no GPU detected)

# TotalSegmentator flags per the specification.
# NOTE: --radiomics excluded because pyradiomics fails to build on Python 3.12.
# The TS CLI flag itself works but requires pyradiomics C extension at import time.
TS_FLAGS = ["--fast", "--statistics", "--ml"]

DEFAULT_OUT_DIR = os.path.join("phase2_work", "temp_seg")


def compute_dynamic_timeout(nifti_path: Optional[str] = None) -> int:
    """
    Compute a hardware-adaptive subprocess timeout for TotalSegmentator.

    Logic:
      1. Detect GPU (CUDA / ROCm / MPS / DirectML).
         GPU present  → base 120 s
         CPU-only     → base 300 s
      2. Scale by available system RAM:
         RAM < 8 GB  → multiply by 3.0  (very constrained)
         RAM < 16 GB → multiply by 2.0  (typical laptop)
         RAM < 32 GB → multiply by 1.5  (standard desktop)
         RAM >= 32 GB → multiply by 1.0  (workstation, no penalty)
      3. Scale by NIfTI file size (proxy for volume size):
         file > 200 MB → add  120 s
         file > 100 MB → add   60 s
         file > 50  MB → add   30 s
      4. Cap at 1800 s (30 min) — protects against infinite hang.
         Floor at 60 s.

    Returns:
        Timeout in whole seconds.
    """
    has_gpu = False
    gpu_label = "CPU"

    # ── GPU detection ───────────────────────────────────────────────────────
    # 1. Try torch (most reliable — respects CUDA_VISIBLE_DEVICES etc.)
    try:
        import torch
        if torch.cuda.is_available():
            has_gpu = True
            name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "CUDA"
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.device_count() > 0 else 0
            gpu_label = f"CUDA ({name}, {vram_gb:.1f}GB VRAM)"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            has_gpu = True
            gpu_label = "Apple MPS"
    except ImportError:
        pass

    # 2. Fallback: check nvidia-smi (works even without torch)
    if not has_gpu:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0 and r.stdout.strip():
                has_gpu = True
                gpu_label = f"CUDA via nvidia-smi ({r.stdout.strip().splitlines()[0]})"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # 3. Fallback: ROCm / AMD — check rocm-smi
    if not has_gpu:
        try:
            r = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0 and r.stdout.strip():
                has_gpu = True
                gpu_label = "ROCm (AMD)"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    base_timeout = _TIMEOUT_GPU_DEFAULT if has_gpu else _TIMEOUT_CPU_DEFAULT

    # ── RAM scaling ─────────────────────────────────────────────────────────
    ram_multiplier = 1.0
    ram_label = "unknown RAM"
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
        total_gb = psutil.virtual_memory().total / (1024**3)
        ram_label = f"{available_gb:.1f}GB avail / {total_gb:.1f}GB total"
        if available_gb < 4:
            ram_multiplier = 4.0
        elif available_gb < 8:
            ram_multiplier = 3.0
        elif available_gb < 16:
            ram_multiplier = 2.0
        elif available_gb < 32:
            ram_multiplier = 1.5
        else:
            ram_multiplier = 1.0
    except ImportError:
        pass  # No psutil — keep default multiplier

    timeout = int(base_timeout * ram_multiplier)

    # ── Volume size bonus ────────────────────────────────────────────────────
    size_bonus = 0
    file_mb = 0.0
    if nifti_path and os.path.isfile(nifti_path):
        file_mb = os.path.getsize(nifti_path) / (1024 * 1024)
        if file_mb > 200:
            size_bonus = 180
        elif file_mb > 100:
            size_bonus = 90
        elif file_mb > 50:
            size_bonus = 45

    timeout += size_bonus

    # ── Clamp ───────────────────────────────────────────────────────────────
    timeout = max(60, min(timeout, 1800))

    logger.info(
        "TotalSegmentator dynamic timeout: %ds  |  GPU=%s (%s)  |  RAM scale=%.1fx (%s)  |  "
        "Volume=%.1f MB (+%ds bonus)",
        timeout, has_gpu, gpu_label, ram_multiplier, ram_label, file_mb, size_bonus,
    )

    return timeout


def _find_totalsegmentator_command() -> str:
    """
    Locate the TotalSegmentator command-line executable.

    Returns:
        The command string to invoke TotalSegmentator.

    Raises:
        RuntimeError: If TotalSegmentator is not found on PATH.
    """
    # Try the standard CLI entry point
    ts_path = shutil.which("TotalSegmentator")
    if ts_path:
        return ts_path

    # On some installs, it's lowercase
    ts_path = shutil.which("totalsegmentator")
    if ts_path:
        return ts_path

    # Fallback: check Python user Scripts directory (pip install --user)
    import site
    user_scripts = os.path.join(site.getusersitepackages().replace("site-packages", "Scripts"))
    for name in ("TotalSegmentator.exe", "TotalSegmentator", "totalsegmentator.exe", "totalsegmentator"):
        candidate = shutil.which(os.path.join(user_scripts, name))
        if candidate:
            logger.info("Found TotalSegmentator in user Scripts: %s", candidate)
            return candidate

    # Fallback: check common Windows user Scripts paths
    appdata_roaming = os.environ.get("APPDATA", "")
    if appdata_roaming:
        py_ver = f"Python{sys.version_info.major}{sys.version_info.minor}"
        roaming_scripts = os.path.join(appdata_roaming, "Python", py_ver, "Scripts")
        for name in ("TotalSegmentator.exe", "TotalSegmentator"):
            candidate = shutil.which(os.path.join(roaming_scripts, name))
            if candidate:
                logger.info("Found TotalSegmentator in Roaming Scripts: %s", candidate)
                return candidate

    # Fallback: try as a Python module
    # TotalSegmentator can be invoked as `python -m totalsegmentator`
    try:
        result = subprocess.run(
            [sys.executable, "-m", "totalsegmentator", "--help"],
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            return f"{sys.executable} -m totalsegmentator"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    raise RuntimeError(
        "TotalSegmentator not found. Install it with: pip install TotalSegmentator\n"
        "Then verify: TotalSegmentator --help"
    )


def _build_command(
    nifti_path: str,
    out_dir: str,
    modality: str = "CT",
) -> list:
    """
    Build the TotalSegmentator command-line arguments.

    Args:
        nifti_path: Path to input NIfTI volume.
        out_dir: Directory for segmentation output.
        modality: "CT" or "MR" — routes to the correct task.

    Returns:
        List of command-line arguments.
    """
    # Determine task based on modality
    modality_upper = modality.upper().strip()
    if modality_upper in ("MR", "MRI"):
        task = "total_mr"
    else:
        # Default to CT
        task = "total"

    ts_cmd = _find_totalsegmentator_command()

    # If the command is a "python -m ..." invocation, split it
    if ts_cmd.startswith(sys.executable):
        cmd_parts = ts_cmd.split()
    else:
        cmd_parts = [ts_cmd]

    cmd = cmd_parts + [
        "-i", os.path.abspath(nifti_path),
        "-o", os.path.abspath(out_dir),
        "--task", task,
    ] + TS_FLAGS

    logger.info("TotalSegmentator command: %s", " ".join(cmd))
    return cmd


def _parse_statistics(out_dir: str, out_arg: str = None) -> Dict[str, Dict[str, Any]]:
    """
    Parse TotalSegmentator's statistics output into the organ_statistics.json
    schema shape.

    TotalSegmentator 2.17+ output layout:
        When -o is a directory path ending in /:
            <out_dir>/statistics.json
        When -o is a file path (multi-label mode, default):
            <out_dir>/../statistics.json   (BESIDE the -o file, not inside it)

    We check both locations for robustness.

    Args:
        out_dir: TotalSegmentator output directory passed as -o.
        out_arg: The actual -o argument passed to TS (may differ from out_dir).

    Returns:
        Dict matching the organ_statistics.json schema.
    """
    result = {}

    # TS 2.17 in multi-label mode writes statistics.json BESIDE the -o path,
    # not inside it. Check both locations.
    candidate_paths = [
        os.path.join(out_dir, "statistics.json"),                     # old behavior
        os.path.join(os.path.dirname(out_dir), "statistics.json"),    # TS 2.17 beside
    ]
    if out_arg:
        candidate_paths.append(
            os.path.join(os.path.dirname(os.path.abspath(out_arg)), "statistics.json")
        )

    stats_path = None
    for p in candidate_paths:
        if os.path.isfile(p):
            stats_path = p
            break

    if stats_path is None:
        logger.warning(
            "statistics.json not found. Checked: %s. "
            "Ensure --statistics flag was used and TS completed successfully.",
            candidate_paths,
        )
        return result

    logger.info("Found statistics.json at: %s", stats_path)

    with open(stats_path, "r") as f:
        raw_stats = json.load(f)

    logger.info(
        "Parsed statistics.json: %d organ entries",
        len(raw_stats) if isinstance(raw_stats, dict) else 0,
    )

    if isinstance(raw_stats, dict):
        for organ_label, organ_data in raw_stats.items():
            if not isinstance(organ_data, dict):
                continue
            result[organ_label] = _normalize_organ_entry(organ_data)
    elif isinstance(raw_stats, list):
        for entry in raw_stats:
            if isinstance(entry, dict) and "name" in entry:
                organ_label = entry["name"]
                result[organ_label] = _normalize_organ_entry(entry)

    # Check for radiomics stats (optional — only present if pyradiomics installed)
    radiomics_candidates = [
        os.path.join(out_dir, "statistics_radiomics.json"),
        os.path.join(os.path.dirname(out_dir), "statistics_radiomics.json"),
    ]
    for rpath in radiomics_candidates:
        if os.path.isfile(rpath):
            try:
                with open(rpath, "r") as f:
                    json.load(f)
                logger.info("Found statistics_radiomics.json at: %s", rpath)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Failed to parse statistics_radiomics.json: %s", e)
            break

    return result


def _normalize_organ_entry(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a single organ's raw stats into the organ_statistics.json schema.

    Handles TotalSegmentator 2.17+ output format:
        {"volume": <voxel_count>, "intensity": <mean_HU>}
    as well as richer older formats with percentile fields.

    Any missing values default to 0.0.
    """
    def get_float(keys, default=0.0):
        """Try multiple key names, return the first match."""
        if isinstance(keys, str):
            keys = [keys]
        for k in keys:
            if k in raw:
                try:
                    return float(raw[k])
                except (ValueError, TypeError):
                    pass
        return default

    def get_int(keys, default=0):
        if isinstance(keys, str):
            keys = [keys]
        for k in keys:
            if k in raw:
                try:
                    return int(raw[k])
                except (ValueError, TypeError):
                    pass
        return default

    # TS 2.17 uses {"volume": N, "intensity": N}
    # "intensity" = mean HU,  "volume" = voxel count (not cc!)
    # Compute spacing-independent volume_cc only if spacing info is available
    # (we don't have it here, so volume stays in voxels).
    ts_mean = get_float(["intensity", "trimmed_mean", "mean", "intensity_mean"])
    ts_voxels = get_int(["volume", "voxel_count", "count", "n_voxels", "num_voxels"])

    # For TS 2.17 minimal format, we can only fill the fields we have.
    # trimmed_mean = intensity (mean HU), volume_cc = voxel_count (in voxels,
    # not cc — the caller should convert using spacing if needed).
    q1 = get_float(["q1", "percentile_25", "p25"])
    q3 = get_float(["q3", "percentile_75", "p75"])
    iqr = q3 - q1 if (q3 != 0.0 or q1 != 0.0) else get_float(["iqr"])

    return {
        "trimmed_mean": ts_mean,
        "trimmed_std": get_float(["trimmed_std", "std", "intensity_std", "stdev"]),
        "median": get_float(["median", "intensity_median"]),
        "mad": get_float(["mad", "median_absolute_deviation"]),
        "p5": get_float(["p5", "percentile_5", "p05"]),
        "p95": get_float(["p95", "percentile_95"]),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "voxel_count": ts_voxels,
        # volume_cc: TS 2.17 doesn't give cc directly; store voxel_count here
        # and Package 3 will convert using voxel_spacing.
        "volume_cc": get_float(["volume_cc", "volume_ml", "volume_l"]),
    }


def run_totalsegmentator(
    nifti_path: str,
    out_dir: str = DEFAULT_OUT_DIR,
    modality: str = "CT",
    timeout: Optional[int] = None,
) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """
    Run TotalSegmentator on a NIfTI volume as an isolated subprocess.

    This function NEVER imports TotalSegmentator as a Python library.
    Subprocess isolation is required to guarantee C++-backed tensor memory
    is released after each run.

    Args:
        nifti_path: Path to the input NIfTI volume (.nii or .nii.gz).
        out_dir: Directory for TotalSegmentator output. Created if needed.
        modality: "CT" or "MR" — routes to the correct segmentation task.
        timeout: Hard timeout in seconds. If None (default), compute_dynamic_timeout()
                 is called to auto-detect based on GPU presence, available RAM, and
                 NIfTI file size.


    Returns:
        Tuple of:
            - out_dir: Path to the segmentation output directory
            - stats_dict: Dict matching schemas/organ_statistics.json shape

    Raises:
        FileNotFoundError: If the input NIfTI doesn't exist.
        RuntimeError: If TotalSegmentator is not installed or fails.
        subprocess.TimeoutExpired: If the segmentation exceeds timeout.
    """
    # ── Validate input ──
    nifti_abs = os.path.abspath(nifti_path)
    if not os.path.isfile(nifti_abs):
        raise FileNotFoundError(f"Input NIfTI not found: {nifti_abs}")

    file_size_mb = os.path.getsize(nifti_abs) / (1024 * 1024)
    if file_size_mb < 0.01:
        raise ValueError(
            f"Input NIfTI is suspiciously small ({file_size_mb:.3f} MB). "
            f"Likely corrupted or empty."
        )

    # ── Prepare output directory ──
    out_abs = os.path.abspath(out_dir)
    os.makedirs(out_abs, exist_ok=True)

    # ── Resolve timeout (auto-detect if not explicitly set) ──
    if timeout is None:
        timeout = compute_dynamic_timeout(nifti_abs)

    # ── Build and run command ──
    cmd = _build_command(nifti_abs, out_abs, modality)
    t_start = time.perf_counter()

    logger.info(
        "Starting TotalSegmentator: input=%s (%.1f MB), modality=%s, timeout=%ds",
        nifti_abs,
        file_size_mb,
        modality,
        timeout,
    )


    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            # Ensure the subprocess doesn't inherit our env in a way that
            # could cause GPU conflicts
            env={**os.environ},
        )
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t_start
        logger.error(
            "TotalSegmentator TIMED OUT after %.1fs (hardware-adaptive limit: %ds). "
            "Consider running on a GPU, closing other applications to free RAM, "
            "or reducing the volume size. The subprocess has been killed.",
            elapsed,
            timeout,
        )
        raise
    except FileNotFoundError:
        raise RuntimeError(
            "TotalSegmentator executable not found. "
            "Install with: pip install TotalSegmentator"
        )

    elapsed = time.perf_counter() - t_start

    # ── Check exit code ──
    if result.returncode != 0:
        stderr_snippet = (result.stderr or "")[:1000]
        logger.error(
            "TotalSegmentator FAILED (exit code %d, %.1fs):\n%s",
            result.returncode,
            elapsed,
            stderr_snippet,
        )
        raise RuntimeError(
            f"TotalSegmentator failed with exit code {result.returncode}. "
            f"stderr: {stderr_snippet}"
        )

    logger.info(
        "TotalSegmentator completed successfully in %.1fs",
        elapsed,
    )

    # Log stdout/stderr for debugging (truncated)
    if result.stdout:
        logger.debug("TS stdout (first 500 chars): %s", result.stdout[:500])
    if result.stderr:
        logger.debug("TS stderr (first 500 chars): %s", result.stderr[:500])

    # ── Parse statistics ──
    stats_dict = _parse_statistics(out_abs, out_arg=out_abs)

    if not stats_dict:
        logger.warning(
            "TotalSegmentator produced no parseable organ statistics. "
            "Check if --statistics flag is supported in this version and "
            "that statistics.json was written next to the output path."
        )

    logger.info(
        "Segmentation complete: %d organs detected, output at %s",
        len(stats_dict),
        out_abs,
    )

    return out_abs, stats_dict
