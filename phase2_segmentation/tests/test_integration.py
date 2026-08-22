"""
test_integration.py — Integration tests for Package 2 end-to-end functionality

These tests verify the complete Package 2 pipeline works correctly when all
components are integrated together. They require external dependencies and
are marked as slow/integration tests.

Run with: pytest -m integration
Skip with: pytest -m "not integration"
"""

import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from phase2_segmentation.volume_builder import assemble_nifti
from phase2_segmentation.segment_runner import run_totalsegmentator
from phase2_segmentation.lifecycle_manager import Phase2LifecycleManager, run_phase2


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndPipeline:
    """Integration tests for the complete Phase 2 pipeline."""

    def test_volume_to_nifti_pipeline(self, sample_hu_volume, realistic_spacing, temp_work_dir):
        """Test volume assembly to NIfTI with real SimpleITK."""
        nifti_path = os.path.join(temp_work_dir, "test_volume.nii.gz")
        
        # This requires actual SimpleITK installation
        result_path = assemble_nifti(
            sample_hu_volume, 
            realistic_spacing, 
            out_path=nifti_path
        )
        
        assert os.path.isfile(result_path)
        assert os.path.getsize(result_path) > 10000  # Should be substantial
        
        # Verify we can read it back with SimpleITK
        try:
            import SimpleITK as sitk
            image = sitk.ReadImage(result_path)
            
            # Check dimensions match
            size = image.GetSize()  # (X, Y, Z)
            expected_shape = sample_hu_volume.shape  # (Z, Y, X)
            assert size == (expected_shape[2], expected_shape[1], expected_shape[0])
            
            # Check spacing was set correctly
            spacing = image.GetSpacing()  # (X, Y, Z)
            expected_spacing = (realistic_spacing[1], realistic_spacing[0], realistic_spacing[2])
            np.testing.assert_allclose(spacing, expected_spacing, rtol=1e-6)
            
        except ImportError:
            pytest.skip("SimpleITK not available for verification")

    @pytest.mark.skip(reason="Requires TotalSegmentator installation and GPU")
    def test_full_segmentation_pipeline(self, sample_hu_volume, realistic_spacing, temp_work_dir):
        """Test complete NIfTI → TotalSegmentator → statistics pipeline."""
        # Step 1: Create NIfTI
        nifti_path = os.path.join(temp_work_dir, "input.nii.gz")
        assemble_nifti(sample_hu_volume, realistic_spacing, out_path=nifti_path)
        
        # Step 2: Run segmentation (this will fail without actual TotalSegmentator)
        seg_dir = os.path.join(temp_work_dir, "segmentation")
        out_dir, stats = run_totalsegmentator(nifti_path, out_dir=seg_dir, modality="CT")
        
        # Step 3: Verify outputs
        assert os.path.isdir(out_dir)
        assert isinstance(stats, dict)
        assert len(stats) > 0  # Should detect some organs
        
        # Verify statistics format
        for organ_name, organ_stats in stats.items():
            assert "volume_cc" in organ_stats
            assert "trimmed_mean" in organ_stats
            assert isinstance(organ_stats["volume_cc"], (int, float))

    def test_lifecycle_manager_with_mocked_dependencies(
        self, 
        sample_hu_volume, 
        realistic_spacing, 
        sample_series_metadata,
        temp_work_dir
    ):
        """Test Phase2LifecycleManager with mocked external dependencies."""
        
        with patch('phase2_segmentation.lifecycle_manager.Phase2LifecycleManager._assemble') as mock_assemble, \
             patch('phase2_segmentation.lifecycle_manager.Phase2LifecycleManager._segment') as mock_segment:
            
            # Mock successful assembly
            nifti_path = os.path.join(temp_work_dir, "volume.nii.gz")
            mock_assemble.return_value = nifti_path
            
            # Mock successful segmentation
            seg_dir = os.path.join(temp_work_dir, "seg")
            organ_stats = {
                "liver": {"volume_cc": 1500.0, "trimmed_mean": 55.2},
                "kidney_left": {"volume_cc": 150.0, "trimmed_mean": 32.1},
            }
            mock_segment.return_value = (seg_dir, organ_stats)
            
            # Run the full lifecycle
            manager = Phase2LifecycleManager(work_dir=temp_work_dir)
            result = manager.run(sample_hu_volume, realistic_spacing, sample_series_metadata)
            
            # Verify successful completion
            assert result.success
            assert result.nifti_path == nifti_path
            assert result.segmentation_dir == seg_dir
            assert result.num_organs_detected == 2
            assert result.total_time_sec > 0
            assert result.assembly_time_sec > 0
            assert result.segmentation_time_sec > 0
            assert result.peak_ram_gb >= 0
            
            # Verify calls were made correctly
            mock_assemble.assert_called_once_with(
                sample_hu_volume, 
                realistic_spacing, 
                os.path.join(temp_work_dir, "temp_volume.nii.gz")
            )
            mock_segment.assert_called_once_with(
                nifti_path,
                os.path.join(temp_work_dir, "temp_seg"),
                "CT"
            )

    def test_multiprocessing_entry_point(
        self,
        sample_hu_volume,
        realistic_spacing,
        sample_series_metadata
    ):
        """Test the run_phase2 entry point used by multiprocessing."""
        
        with patch('phase2_segmentation.lifecycle_manager.Phase2LifecycleManager') as mock_manager_class:
            # Mock the manager and its result
            mock_manager = mock_manager_class.return_value
            mock_result = mock_manager.run.return_value
            mock_result.success = True
            mock_result.num_organs_detected = 3
            mock_result.peak_ram_gb = 2.1
            
            # Test data for multiprocessing signature
            findings_per_slice = [[] for _ in range(sample_hu_volume.shape[0])]
            instance_numbers = list(range(1, sample_hu_volume.shape[0] + 1))
            
            # Call the entry point
            result = run_phase2(
                sample_hu_volume,
                findings_per_slice, 
                instance_numbers,
                realistic_spacing,
                sample_series_metadata
            )
            
            # Verify result
            assert result.success
            assert result.num_organs_detected == 3
            assert result.peak_ram_gb == 2.1
            
            # Verify manager was created and called
            mock_manager_class.assert_called_once()
            mock_manager.run.assert_called_once_with(
                sample_hu_volume, 
                realistic_spacing, 
                sample_series_metadata,
                findings_per_slice=findings_per_slice
            )


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Integration tests for error handling across component boundaries."""

    def test_corrupted_volume_handling(self, temp_work_dir):
        """Test handling of corrupted volume data throughout the pipeline."""
        # Create a problematic volume (constant values)
        corrupted_volume = np.full((10, 64, 64), -1024.0, dtype=np.float32)
        spacing = (1.0, 1.0, 2.0)
        
        manager = Phase2LifecycleManager(work_dir=temp_work_dir)
        result = manager.run(corrupted_volume, spacing)
        
        # Should fail gracefully
        assert not result.success
        assert "constant value" in result.error_message

    def test_invalid_spacing_handling(self, sample_hu_volume, temp_work_dir):
        """Test handling of invalid spacing values."""
        invalid_spacing = (0.0, 0.7, 1.5)  # Zero spacing is invalid
        
        manager = Phase2LifecycleManager(work_dir=temp_work_dir)
        result = manager.run(sample_hu_volume, invalid_spacing)
        
        # Should fail gracefully
        assert not result.success
        assert "must be positive" in result.error_message

    def test_disk_space_simulation(self, sample_hu_volume, realistic_spacing):
        """Test handling when output directory cannot be created."""
        # Try to use a path that can't be created (root filesystem)
        if os.name == 'nt':  # Windows
            invalid_path = "C:/Windows/System32/invalid_phase2_dir"
        else:  # Unix-like
            invalid_path = "/root/invalid_phase2_dir"
        
        manager = Phase2LifecycleManager(work_dir=invalid_path)
        result = manager.run(sample_hu_volume, realistic_spacing)
        
        # Should either succeed (if permissions allow) or fail gracefully
        if not result.success:
            assert len(result.error_message) > 0


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceCharacteristics:
    """Integration tests for performance and resource usage."""

    def test_ram_monitoring_accuracy(self, sample_hu_volume, realistic_spacing, temp_work_dir):
        """Test that RAM monitoring provides reasonable measurements."""
        
        with patch('phase2_segmentation.lifecycle_manager.Phase2LifecycleManager._assemble') as mock_assemble, \
             patch('phase2_segmentation.lifecycle_manager.Phase2LifecycleManager._segment') as mock_segment:
            
            # Mock quick operations
            mock_assemble.return_value = "test.nii.gz"
            mock_segment.return_value = ("test_seg", {})
            
            manager = Phase2LifecycleManager(
                work_dir=temp_work_dir,
                ram_budget_gb=1.0  # Low budget for testing
            )
            result = manager.run(sample_hu_volume, realistic_spacing)
            
            # Should complete and report RAM usage
            assert result.success
            assert result.peak_ram_gb > 0  # Should measure something
            assert result.total_time_sec > 0
            assert result.assembly_time_sec >= 0
            assert result.segmentation_time_sec >= 0

    def test_large_volume_handling(self, realistic_spacing, temp_work_dir):
        """Test handling of large volume data."""
        # Create a larger volume (but not too large for CI)
        large_volume = np.random.randn(100, 256, 256).astype(np.float32) * 100
        
        with patch('phase2_segmentation.lifecycle_manager.Phase2LifecycleManager._assemble') as mock_assemble, \
             patch('phase2_segmentation.lifecycle_manager.Phase2LifecycleManager._segment') as mock_segment:
            
            mock_assemble.return_value = "large.nii.gz"
            mock_segment.return_value = ("large_seg", {"liver": {"volume_cc": 2000.0}})
            
            manager = Phase2LifecycleManager(work_dir=temp_work_dir)
            result = manager.run(large_volume, realistic_spacing)
            
            assert result.success
            assert result.volume_shape == (100, 256, 256)
            assert result.num_organs_detected == 1

    def test_concurrent_safety(self, sample_hu_volume, realistic_spacing):
        """Test that multiple Phase2LifecycleManager instances don't interfere."""
        import threading
        
        results = {}
        
        def run_manager(manager_id):
            temp_dir = f"temp_concurrent_{manager_id}"
            os.makedirs(temp_dir, exist_ok=True)
            
            try:
                with patch('phase2_segmentation.lifecycle_manager.Phase2LifecycleManager._assemble') as mock_assemble, \
                     patch('phase2_segmentation.lifecycle_manager.Phase2LifecycleManager._segment') as mock_segment:
                    
                    mock_assemble.return_value = f"test_{manager_id}.nii.gz"
                    mock_segment.return_value = (f"seg_{manager_id}", {"organ": {"volume_cc": 100.0}})
                    
                    manager = Phase2LifecycleManager(work_dir=temp_dir)
                    result = manager.run(sample_hu_volume, realistic_spacing)
                    results[manager_id] = result
            finally:
                import shutil
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Run multiple managers concurrently
        threads = []
        for i in range(3):
            thread = threading.Thread(target=run_manager, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join(timeout=30)  # Reasonable timeout
        
        # All should succeed independently
        assert len(results) == 3
        for result in results.values():
            assert result.success


@pytest.mark.integration
class TestPackage7Integration:
    """Integration tests covering Stage 4 and Stage 5 (Scoring & Handoff)."""

    def test_full_phase2_pipeline_p7_outputs(self, sample_hu_volume, realistic_spacing, temp_work_dir):
        """After run(), Phase2Result.findings_path exists and JSON is schema-valid."""
        from schemas.candidate import Candidate
        
        with patch.dict('sys.modules', {'SimpleITK': MagicMock()}), \
             patch('phase2_segmentation.lifecycle_manager.Phase2LifecycleManager._assemble') as mock_assemble, \
             patch('phase2_segmentation.lifecycle_manager.Phase2LifecycleManager._segment') as mock_segment, \
             patch('phase1_ingestion.finding_tracker.link_findings_across_slices', return_value=[]), \
             patch('phase2_segmentation.seed_clusterer.cluster_phase1_seeds', return_value=[]), \
             patch('phase2_segmentation.organ_sweep.sweep_organs', return_value=[]), \
             patch('phase2_segmentation.candidate_merger.merge_candidates', return_value=[Candidate(phase1_confidence=0.8)]):
             
            mock_assemble.return_value = "temp.nii"
            
            def mock_segment_side_effect(nifti_path, seg_dir, modality):
                os.makedirs(seg_dir, exist_ok=True)
                with open(os.path.join(seg_dir, "temp_seg.nii"), "w") as f:
                    f.write("dummy")
                return (seg_dir, {"liver": {"volume_cc": 100.0, "trimmed_mean": 40.0}})
                
            mock_segment.side_effect = mock_segment_side_effect
            
            manager = Phase2LifecycleManager(work_dir=temp_work_dir)
            findings_per_slice = [[Candidate(phase1_confidence=0.8)]]
            
            result = manager.run(sample_hu_volume, realistic_spacing, {}, findings_per_slice=findings_per_slice)
            
            assert result.success
            assert result.findings_path != ""
            assert os.path.exists(result.findings_path)
            
            import json
            with open(result.findings_path, "r") as f:
                data = json.load(f)
            
            assert "findings" in data
            assert data["candidate_count"] == 1



    def test_all_gate_values_are_valid_strings(self, sample_hu_volume, realistic_spacing, temp_work_dir):
        """After Stage 5, every non-suppressed candidate has gate AUTO_CONFIRMED or MANUAL_REVIEW."""
        from schemas.candidate import Candidate
        
        with patch.dict('sys.modules', {'SimpleITK': MagicMock()}), \
             patch('phase2_segmentation.lifecycle_manager.Phase2LifecycleManager._assemble') as mock_assemble, \
             patch('phase2_segmentation.lifecycle_manager.Phase2LifecycleManager._segment') as mock_segment, \
             patch('phase1_ingestion.finding_tracker.link_findings_across_slices', return_value=[]), \
             patch('phase2_segmentation.seed_clusterer.cluster_phase1_seeds', return_value=[]), \
             patch('phase2_segmentation.organ_sweep.sweep_organs', return_value=[]), \
             patch('phase2_segmentation.candidate_merger.merge_candidates', return_value=[Candidate(phase1_confidence=0.9)]):
             
            mock_assemble.return_value = "temp.nii"
            
            def mock_segment_side_effect(nifti_path, seg_dir, modality):
                os.makedirs(seg_dir, exist_ok=True)
                with open(os.path.join(seg_dir, "temp_seg.nii"), "w") as f:
                    f.write("dummy")
                return (seg_dir, {"liver": {"volume_cc": 100.0, "trimmed_mean": 40.0}})
                
            mock_segment.side_effect = mock_segment_side_effect
            
            manager = Phase2LifecycleManager(work_dir=temp_work_dir)
            findings_per_slice = [[Candidate(phase1_confidence=0.9)]]
            
            result = manager.run(sample_hu_volume, realistic_spacing, {}, findings_per_slice=findings_per_slice)
            
            for c in result.candidates:
                if not c.suppressed:
                    assert c.gate in {"AUTO_CONFIRMED", "MANUAL_REVIEW"}