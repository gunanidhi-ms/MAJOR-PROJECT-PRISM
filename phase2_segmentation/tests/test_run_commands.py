"""
test_run_commands.py — Tests for running actual Package 2 commands

These tests can be used to verify the implementation works with real dependencies.
Run them manually when TotalSegmentator and SimpleITK are properly installed.

Usage:
    pytest phase2_segmentation/tests/test_run_commands.py -v -s
    pytest phase2_segmentation/tests/test_run_commands.py::test_check_dependencies -v
"""

import os
import sys
import tempfile
import numpy as np
import pytest


def test_check_dependencies():
    """Check if required dependencies are available."""
    
    # Test SimpleITK
    try:
        import SimpleITK as sitk
        print(f"✓ SimpleITK {sitk.Version.VersionString()} available")
        
        # Test basic functionality
        test_array = np.ones((10, 20, 30), dtype=np.float32)
        image = sitk.GetImageFromArray(test_array)
        assert image.GetSize() == (30, 20, 10)  # SimpleITK reverses dimensions
        print("✓ SimpleITK basic functionality works")
        
    except ImportError:
        pytest.skip("SimpleITK not installed")
    
    # Test psutil
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"✓ psutil available, system RAM: {memory.total / (1024**3):.1f} GB")
    except ImportError:
        pytest.skip("psutil not installed")
    
    # Test TotalSegmentator (optional)
    try:
        import subprocess
        result = subprocess.run(
            ["TotalSegmentator", "--help"],
            capture_output=True,
            timeout=10
        )
        if result.returncode == 0:
            print("✓ TotalSegmentator CLI available")
        else:
            print("⚠ TotalSegmentator CLI found but returned error")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        try:
            result = subprocess.run(
                [sys.executable, "-m", "totalsegmentator", "--help"],
                capture_output=True,
                timeout=10
            )
            if result.returncode == 0:
                print("✓ TotalSegmentator Python module available")
            else:
                print("⚠ TotalSegmentator module found but returned error")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("✗ TotalSegmentator not available (install with: pip install TotalSegmentator)")


def test_create_test_nifti():
    """Test creating a NIfTI file with realistic test data."""
    try:
        import SimpleITK as sitk
    except ImportError:
        pytest.skip("SimpleITK not available")
    
    from phase2_segmentation.volume_builder import assemble_nifti
    
    # Create test volume with realistic HU values
    volume = np.zeros((30, 128, 128), dtype=np.float32)
    volume.fill(-1000)  # Air
    
    # Add soft tissue region
    volume[10:20, 40:80, 40:80] = np.random.normal(50, 15, (10, 40, 40))
    
    # Add bone region  
    volume[15:18, 55:65, 55:65] = np.random.normal(400, 50, (3, 10, 10))
    
    spacing = (0.742, 0.742, 1.25)
    
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        temp_path = f.name
    
    try:
        result_path = assemble_nifti(volume, spacing, out_path=temp_path)
        
        # Verify file was created
        assert os.path.isfile(result_path)
        file_size = os.path.getsize(result_path)
        print(f"✓ Created NIfTI: {result_path} ({file_size} bytes)")
        
        # Verify we can read it back
        image = sitk.ReadImage(result_path)
        print(f"✓ NIfTI dimensions: {image.GetSize()}")
        print(f"✓ NIfTI spacing: {image.GetSpacing()}")
        
        return result_path
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise e


@pytest.mark.slow
def test_totalsegmentator_command_discovery():
    """Test finding TotalSegmentator executable."""
    from phase2_segmentation.segment_runner import _find_totalsegmentator_command
    
    try:
        command = _find_totalsegmentator_command()
        print(f"✓ Found TotalSegmentator command: {command}")
        
        # Test building a command
        from phase2_segmentation.segment_runner import _build_command
        cmd = _build_command("test.nii.gz", "test_out", modality="CT")
        print(f"✓ Built command: {' '.join(cmd[:5])}... (truncated)")
        
    except RuntimeError as e:
        pytest.skip(f"TotalSegmentator not available: {e}")


@pytest.mark.slow  
@pytest.mark.skip(reason="Requires TotalSegmentator installation and may be slow")
def test_run_totalsegmentator_on_test_data():
    """Test running TotalSegmentator on actual test data."""
    
    # First create test NIfTI
    nifti_path = test_create_test_nifti()
    
    try:
        from phase2_segmentation.segment_runner import run_totalsegmentator
        
        with tempfile.TemporaryDirectory() as temp_dir:
            seg_dir = os.path.join(temp_dir, "segmentation")
            
            print(f"Running TotalSegmentator on {nifti_path}...")
            print("This may take several minutes...")
            
            out_dir, stats = run_totalsegmentator(
                nifti_path,
                out_dir=seg_dir,
                modality="CT",
                timeout=300  # 5 minutes
            )
            
            print(f"✓ Segmentation completed: {out_dir}")
            print(f"✓ Detected {len(stats)} organs")
            
            # Show some results
            for organ, organ_stats in list(stats.items())[:3]:
                volume_cc = organ_stats.get("volume_cc", 0)
                print(f"  {organ}: {volume_cc:.1f} cc")
            
            # Verify output files exist
            assert os.path.isdir(out_dir)
            segmentation_files = [f for f in os.listdir(out_dir) if f.endswith('.nii.gz')]
            print(f"✓ Created {len(segmentation_files)} segmentation files")
            
    finally:
        # Clean up
        if os.path.exists(nifti_path):
            os.unlink(nifti_path)


def test_full_lifecycle_with_test_data():
    """Test the complete Phase 2 lifecycle with mocked segmentation."""
    from phase2_segmentation.lifecycle_manager import Phase2LifecycleManager
    
    # Create test volume
    volume = np.zeros((20, 64, 64), dtype=np.float32)
    volume.fill(-1000)
    volume[8:12, 20:40, 20:40] = np.random.normal(50, 15, (4, 20, 20))
    
    spacing = (1.0, 1.0, 2.0)
    series_meta = {
        "series_instance_uid": "test.series.123",
        "modality": "CT"
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock the segmentation step to avoid requiring TotalSegmentator
        from unittest.mock import patch
        
        with patch('phase2_segmentation.lifecycle_manager.Phase2LifecycleManager._segment') as mock_segment:
            # Mock successful segmentation
            mock_segment.return_value = (
                os.path.join(temp_dir, "seg"),
                {"liver": {"volume_cc": 1200.0, "trimmed_mean": 55.0}}
            )
            
            manager = Phase2LifecycleManager(work_dir=temp_dir)
            result = manager.run(volume, spacing, series_meta)
            
            print(f"✓ Lifecycle completed successfully: {result.success}")
            print(f"✓ Total time: {result.total_time_sec:.2f}s")
            print(f"✓ Peak RAM: {result.peak_ram_gb:.2f} GB")
            print(f"✓ Organs detected: {result.num_organs_detected}")
            
            assert result.success
            assert result.num_organs_detected == 1
            # File existence is verified by test_volume_builder.py unit tests.
            # Here we only verify orchestration: success flag and organ count.



if __name__ == "__main__":
    # Allow running this file directly for manual testing
    print("=== Package 2 Manual Testing ===")
    
    print("\n1. Checking dependencies...")
    test_check_dependencies()
    
    print("\n2. Testing NIfTI creation...")
    nifti_path = test_create_test_nifti()
    print(f"Test NIfTI created at: {nifti_path}")
    
    print("\n3. Testing command discovery...")
    test_totalsegmentator_command_discovery()
    
    print("\n4. Testing full lifecycle...")
    test_full_lifecycle_with_test_data()
    
    print("\n✓ All manual tests completed successfully!")
    
    # Clean up
    if 'nifti_path' in locals() and os.path.exists(nifti_path):
        os.unlink(nifti_path)