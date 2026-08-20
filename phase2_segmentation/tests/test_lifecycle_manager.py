"""
test_lifecycle_manager.py — Unit tests for Phase 2 lifecycle orchestration

Tests:
    - RAMWatchdog monitoring and peak detection
    - Phase2LifecycleManager orchestration
    - End-to-end pipeline integration
    - Cleanup behavior on success/failure
    - Multiprocessing entry point function
"""

import os
import time
import threading
import tempfile
from unittest.mock import patch, MagicMock
import pytest
import numpy as np

from phase2_segmentation.lifecycle_manager import (
    RAMWatchdog,
    Phase2LifecycleManager, 
    Phase2Result,
    run_phase2,
)


class TestRAMWatchdog:
    """Test RAM monitoring functionality."""

    def test_ram_watchdog_basic_monitoring(self):
        """Test basic RAM monitoring starts, runs, and returns a positive peak.
        
        Uses real psutil (installed) — no mocking needed.
        """
        watchdog = RAMWatchdog(budget_gb=64.0, interval_sec=0.05)  # High budget so no alarm
        watchdog.start()
        time.sleep(0.2)  # Let it poll a couple of times
        peak = watchdog.stop()

        assert peak > 0.0, f"Peak RAM should be positive, got {peak}"
        assert not watchdog.exceeded_budget

    def test_ram_watchdog_budget_exceeded(self, caplog):
        """Test budget exceeded detection by setting budget below current process RSS.
        
        Since _monitor_loop now tracks process RSS (not system-wide RAM),
        we set budget=0.0 which is always below any real process RSS.
        """
        # A budget of 0.0 GB is always below the process's RSS (even at 0.3GB idle)
        watchdog = RAMWatchdog(budget_gb=0.0, interval_sec=0.05)
        watchdog.start()
        time.sleep(0.2)
        watchdog.stop()

        assert watchdog.exceeded_budget, (
            f"Budget 0.0 GB should always be exceeded by process RSS "
            f"(peak={watchdog.peak_gb:.3f} GB)"
        )

    def test_ram_watchdog_without_psutil(self, caplog):
        """Test graceful handling when psutil is not available.
        
        Patches the psutil import inside start() via sys.modules.
        """
        import sys
        real_psutil = sys.modules.get("psutil")
        sys.modules["psutil"] = None  # Causes 'import psutil' to fail
        try:
            watchdog = RAMWatchdog()
            watchdog.start()  # Should log a warning and return without crashing
            peak = watchdog.stop()
            # Peak stays at 0 since baseline was never set
            assert peak == 0.0
        finally:
            if real_psutil is None:
                sys.modules.pop("psutil", None)
            else:
                sys.modules["psutil"] = real_psutil

    def test_ram_watchdog_thread_cleanup(self):
        """Test that monitoring thread is properly stopped after stop() is called."""
        watchdog = RAMWatchdog(budget_gb=64.0, interval_sec=0.05)
        watchdog.start()

        # Thread must be running
        assert watchdog._thread is not None
        assert watchdog._thread.is_alive()

        watchdog.stop()

        # Running flag must be cleared immediately
        assert not watchdog._running
        # Give thread time to exit (interval is 0.05s, join timeout is 2s)
        time.sleep(0.2)
        assert not watchdog._thread.is_alive()


class TestPhase2Result:
    """Test Phase2Result dataclass functionality."""

    def test_default_result(self):
        """Test default Phase2Result initialization."""
        result = Phase2Result()
        
        assert not result.success
        assert result.nifti_path == ""
        assert result.organ_stats == {}
        assert result.peak_ram_gb == 0.0
        assert result.total_time_sec == 0.0
        assert result.num_organs_detected == 0

    def test_result_with_data(self):
        """Test Phase2Result with populated data."""
        organ_stats = {"liver": {"volume_cc": 1500.0}}
        result = Phase2Result(
            success=True,
            nifti_path="/path/to/volume.nii.gz",
            organ_stats=organ_stats,
            peak_ram_gb=2.5,
            num_organs_detected=len(organ_stats),
        )
        
        assert result.success
        assert result.nifti_path == "/path/to/volume.nii.gz"
        assert result.num_organs_detected == 1
        assert result.peak_ram_gb == 2.5


class TestPhase2LifecycleManager:
    """Test the main lifecycle management class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_volume = np.random.randn(20, 128, 128).astype(np.float32) * 100
        self.test_spacing = (0.7, 0.7, 2.0)
        self.test_series_meta = {
            "series_instance_uid": "1.2.3.4.5.6.7.8.9",
            "modality": "CT"
        }

    @patch('phase2_segmentation.lifecycle_manager.Phase2LifecycleManager._segment')
    @patch('phase2_segmentation.lifecycle_manager.Phase2LifecycleManager._assemble')
    @patch('phase2_segmentation.lifecycle_manager.RAMWatchdog')
    def test_successful_pipeline(self, mock_watchdog_class, mock_assemble, mock_segment):
        """Test successful end-to-end pipeline execution."""
        # Mock watchdog
        mock_watchdog = MagicMock()
        mock_watchdog.stop.return_value = 2.1
        mock_watchdog_class.return_value = mock_watchdog
        
        # Mock assembly
        nifti_path = os.path.join(self.temp_dir, "volume.nii.gz")
        mock_assemble.return_value = nifti_path
        
        # Mock segmentation 
        seg_dir = os.path.join(self.temp_dir, "seg")
        organ_stats = {"liver": {"volume_cc": 1500.0}, "kidney_left": {"volume_cc": 150.0}}
        mock_segment.return_value = (seg_dir, organ_stats)
        
        manager = Phase2LifecycleManager(work_dir=self.temp_dir)
        result = manager.run(self.test_volume, self.test_spacing, self.test_series_meta)
        
        assert result.success
        assert result.nifti_path == nifti_path
        assert result.segmentation_dir == seg_dir
        assert result.num_organs_detected == 2
        assert result.peak_ram_gb == 2.1
        assert result.total_time_sec > 0
        
        # Verify methods were called
        mock_assemble.assert_called_once()
        mock_segment.assert_called_once()
        mock_watchdog.start.assert_called_once()
        mock_watchdog.stop.assert_called_once()

    @patch('phase2_segmentation.lifecycle_manager.Phase2LifecycleManager._segment')
    @patch('phase2_segmentation.lifecycle_manager.Phase2LifecycleManager._assemble')
    @patch('phase2_segmentation.lifecycle_manager.RAMWatchdog')
    def test_assembly_failure(self, mock_watchdog_class, mock_assemble, mock_segment):
        """Test handling of NIfTI assembly failure.""" 
        # Mock watchdog
        mock_watchdog = MagicMock()
        mock_watchdog.stop.return_value = 1.5
        mock_watchdog_class.return_value = mock_watchdog
        
        # Mock assembly failure
        mock_assemble.side_effect = ValueError("Invalid volume dimensions")
        
        manager = Phase2LifecycleManager(work_dir=self.temp_dir)
        result = manager.run(self.test_volume, self.test_spacing)
        
        assert not result.success
        assert "Invalid volume dimensions" in result.error_message
        assert result.peak_ram_gb == 1.5  # Watchdog still recorded
        
        # Segmentation should not have been called
        mock_segment.assert_not_called()

    @patch('phase2_segmentation.lifecycle_manager.Phase2LifecycleManager._segment')
    @patch('phase2_segmentation.lifecycle_manager.Phase2LifecycleManager._assemble') 
    @patch('phase2_segmentation.lifecycle_manager.Phase2LifecycleManager._cleanup')
    @patch('phase2_segmentation.lifecycle_manager.RAMWatchdog')
    def test_cleanup_on_success(self, mock_watchdog_class, mock_cleanup, mock_assemble, mock_segment):
        """Test cleanup behavior when cleanup_on_success=True."""
        # Setup mocks
        mock_watchdog = MagicMock()
        mock_watchdog.stop.return_value = 2.0
        mock_watchdog_class.return_value = mock_watchdog
        
        mock_assemble.return_value = "volume.nii.gz"
        mock_segment.return_value = ("seg_dir", {})
        
        manager = Phase2LifecycleManager(
            work_dir=self.temp_dir,
            cleanup_on_success=True
        )
        result = manager.run(self.test_volume, self.test_spacing)
        
        assert result.success
        mock_cleanup.assert_called_once()

    @patch('phase2_segmentation.lifecycle_manager.Phase2LifecycleManager._assemble')
    @patch('phase2_segmentation.lifecycle_manager.Phase2LifecycleManager._cleanup') 
    @patch('phase2_segmentation.lifecycle_manager.RAMWatchdog')
    def test_cleanup_on_failure(self, mock_watchdog_class, mock_cleanup, mock_assemble):
        """Test cleanup behavior when cleanup_on_failure=True.""" 
        # Setup mocks
        mock_watchdog = MagicMock()
        mock_watchdog.stop.return_value = 2.0
        mock_watchdog_class.return_value = mock_watchdog
        
        mock_assemble.side_effect = RuntimeError("Assembly failed")
        
        manager = Phase2LifecycleManager(
            work_dir=self.temp_dir,
            cleanup_on_failure=True
        )
        result = manager.run(self.test_volume, self.test_spacing)
        
        assert not result.success
        mock_cleanup.assert_called_once()

    def test_work_directory_creation(self):
        """Test that work directory is created if it doesn't exist."""
        nonexistent_dir = os.path.join(self.temp_dir, "nonexistent", "deep", "path")
        
        manager = Phase2LifecycleManager(work_dir=nonexistent_dir)
        
        # This should create the directory structure
        with patch.object(manager, '_assemble') as mock_assemble, \
             patch.object(manager, '_segment') as mock_segment:
            mock_assemble.side_effect = RuntimeError("Stop early")  # Fail fast
            
            manager.run(self.test_volume, self.test_spacing)
            
            assert os.path.isdir(nonexistent_dir)

    @patch('phase2_segmentation.lifecycle_manager.Phase2LifecycleManager._segment')
    @patch('phase2_segmentation.lifecycle_manager.Phase2LifecycleManager._assemble')
    @patch('phase2_segmentation.lifecycle_manager.RAMWatchdog')
    def test_successful_pipeline_with_baselines(self, mock_watchdog_class, mock_assemble, mock_segment):
        """Test pipeline run with baseline stats and region detection (when temp_seg.nii exists)."""
        import SimpleITK as sitk
        # Mock watchdog
        mock_watchdog = MagicMock()
        mock_watchdog.stop.return_value = 1.5
        mock_watchdog_class.return_value = mock_watchdog
        
        # Mock assembly
        nifti_path = os.path.join(self.temp_dir, "volume.nii.gz")
        mock_assemble.return_value = nifti_path
        
        # Mock segmentation
        seg_dir = os.path.join(self.temp_dir, "seg")
        ts_stats = {"liver": {"volume": 100, "intensity": 50}}
        mock_segment.return_value = (seg_dir, ts_stats)
        
        # Create a fake temp_seg.nii containing label 5 (liver)
        seg_arr = np.zeros_like(self.test_volume, dtype=np.uint8)
        # Mark a portion of the volume as liver (label 5)
        seg_arr[2:8, 2:8, 2:8] = 5
        
        # Write to self.temp_dir as temp_seg.nii
        seg_img = sitk.GetImageFromArray(seg_arr)
        seg_nii_path = os.path.join(self.temp_dir, "temp_seg.nii")
        sitk.WriteImage(seg_img, seg_nii_path)
        
        manager = Phase2LifecycleManager(work_dir=self.temp_dir)
        result = manager.run(self.test_volume, self.test_spacing, self.test_series_meta)
        
        assert result.success
        assert result.num_organs_detected == 1
        assert "liver" in result.organ_stats
        
        # Verify stats populated conform to organ_statistics.json schema
        liver_stats = result.organ_stats["liver"]
        assert "trimmed_mean" in liver_stats
        assert "trimmed_std" in liver_stats
        assert "median" in liver_stats
        assert "mad" in liver_stats
        assert "volume_cc" in liver_stats
        assert liver_stats["voxel_count"] == 216  # 6 * 6 * 6 = 216
        
        # Region classification check (liver present only -> abdomen_pelvis)
        assert result.region == "abdomen_pelvis"

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


class TestMultiprocessingEntryPoint:
    """Test the run_phase2 function used by multiprocessing."""

    @patch('phase2_segmentation.lifecycle_manager.Phase2LifecycleManager')
    def test_run_phase2_signature(self, mock_manager_class):
        """Test that run_phase2 has the correct signature for multiprocessing."""
        mock_manager = MagicMock()
        mock_result = Phase2Result(success=True, num_organs_detected=5)
        mock_manager.run.return_value = mock_result
        mock_manager_class.return_value = mock_manager
        
        volume = np.zeros((20, 128, 128))
        findings_per_slice = [[] for _ in range(20)]
        instance_numbers = list(range(1, 21))
        spacing = (0.7, 0.7, 2.0)
        series_meta = {"modality": "CT"}
        
        result = run_phase2(volume, findings_per_slice, instance_numbers, spacing, series_meta)
        
        assert isinstance(result, Phase2Result)
        assert result.success
        assert result.num_organs_detected == 5
        
        # Verify manager was created and called
        mock_manager_class.assert_called_once()
        mock_manager.run.assert_called_once_with(volume, spacing, series_meta, findings_per_slice=findings_per_slice)

    @patch('phase2_segmentation.lifecycle_manager.logging.basicConfig')
    @patch('phase2_segmentation.lifecycle_manager.Phase2LifecycleManager')
    def test_run_phase2_logging_setup(self, mock_manager_class, mock_logging_config):
        """Test that run_phase2 sets up subprocess logging correctly."""
        mock_manager = MagicMock()
        mock_manager.run.return_value = Phase2Result(success=True)
        mock_manager_class.return_value = mock_manager
        
        volume = np.zeros((10, 64, 64))
        result = run_phase2(volume, [], [], (1.0, 1.0, 1.0))
        
        # Verify logging was configured
        mock_logging_config.assert_called_once()
        config_call = mock_logging_config.call_args
        assert 'level' in config_call[1]
        assert 'format' in config_call[1]
        assert 'Phase2' in config_call[1]['format']

    @patch('phase2_segmentation.lifecycle_manager.Phase2LifecycleManager')
    def test_run_phase2_work_dir_creation(self, mock_manager_class):
        """Test that run_phase2 creates unique work directories."""
        mock_manager_class.return_value.run.return_value = Phase2Result(success=True)
        
        volume = np.zeros((10, 64, 64))
        
        # Call run_phase2 multiple times
        with patch('time.time', side_effect=[1000, 2000]):
            result1 = run_phase2(volume, [], [], (1.0, 1.0, 1.0))
            result2 = run_phase2(volume, [], [], (1.0, 1.0, 1.0))
        
        # Verify different work directories were used
        calls = mock_manager_class.call_args_list
        work_dir1 = calls[0][1]['work_dir'] 
        work_dir2 = calls[1][1]['work_dir']
        
        assert work_dir1 != work_dir2
        assert "run_1000" in work_dir1
        assert "run_2000" in work_dir2


@pytest.fixture
def temp_dir():
    """Fixture providing a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    import shutil
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_volume():
    """Fixture providing a sample test volume."""
    return np.random.randn(20, 128, 128).astype(np.float32) * 100