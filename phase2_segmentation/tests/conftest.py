"""
conftest.py — Shared pytest fixtures and configuration for Package 2 tests

Provides common test fixtures and setup for all phase2_segmentation tests.
"""

import os
import tempfile
import shutil
import numpy as np
import pytest


@pytest.fixture(scope="session")
def sample_hu_volume():
    """Fixture providing a realistic HU volume for testing."""
    # Create a volume with realistic CT HU values
    volume = np.zeros((30, 256, 256), dtype=np.float32)
    
    # Air background
    volume.fill(-1000)
    
    # Add some tissue-like structures
    # Soft tissue region (center)
    volume[10:20, 100:150, 100:150] = np.random.normal(50, 15, (10, 50, 50))
    
    # Bone-like structure
    volume[15:18, 120:130, 120:130] = np.random.normal(400, 50, (3, 10, 10))
    
    # Lung tissue
    volume[5:25, 50:80, 180:220] = np.random.normal(-800, 100, (20, 30, 40))
    
    return volume


@pytest.fixture(scope="session") 
def realistic_spacing():
    """Fixture providing realistic CT spacing values."""
    return (0.742, 0.742, 1.25)  # (row_mm, col_mm, slice_thickness_mm)


@pytest.fixture(scope="session")
def sample_series_metadata():
    """Fixture providing sample DICOM series metadata."""
    return {
        "series_instance_uid": "1.2.840.10008.1.2.1.3.123456789.20240101.120000.001",
        "study_instance_uid": "1.2.840.10008.1.2.1.3.123456789.20240101.120000",
        "modality": "CT",
        "patient_id": "TEST_PATIENT_001",
        "study_date": "20240101",
        "series_description": "CHEST CT ANGIO",
        "slice_thickness": "1.25",
        "pixel_spacing": ["0.742", "0.742"],
    }


@pytest.fixture
def temp_work_dir():
    """Fixture providing a temporary working directory."""
    temp_dir = tempfile.mkdtemp(prefix="phase2_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_organ_statistics():
    """Fixture providing sample organ statistics matching the schema."""
    return {
        "liver": {
            "trimmed_mean": 55.2,
            "trimmed_std": 12.8,
            "median": 54.1,
            "mad": 8.5,
            "p5": 35.0,
            "p95": 75.0,
            "q1": 48.5,
            "q3": 62.0,
            "iqr": 13.5,
            "voxel_count": 120450,
            "volume_cc": 1508.2,
        },
        "kidney_left": {
            "trimmed_mean": 32.1,
            "trimmed_std": 18.5,
            "median": 31.8,
            "mad": 12.2,
            "p5": 10.0,
            "p95": 65.0,
            "q1": 22.0,
            "q3": 42.5,
            "iqr": 20.5,
            "voxel_count": 15230,
            "volume_cc": 152.3,
        },
        "kidney_right": {
            "trimmed_mean": 34.5,
            "trimmed_std": 16.8,
            "median": 33.2,
            "mad": 11.8,
            "p5": 12.0,
            "p95": 68.0,
            "q1": 24.5,
            "q3": 44.0,
            "iqr": 19.5,
            "voxel_count": 14820,
            "volume_cc": 148.2,
        }
    }


@pytest.fixture
def mock_totalsegmentator_output(temp_work_dir, sample_organ_statistics):
    """Fixture providing mock TotalSegmentator output files."""
    import json
    
    # Create statistics.json
    stats_file = os.path.join(temp_work_dir, "statistics.json")
    with open(stats_file, "w") as f:
        json.dump(sample_organ_statistics, f, indent=2)
    
    # Create statistics_radiomics.json
    radiomics_data = {
        "liver": {
            "original_firstorder_Mean": 55.18,
            "original_glcm_Contrast": 182.45,
            "original_glrlm_RunLengthNonUniformity": 1850.2,
        }
    }
    radiomics_file = os.path.join(temp_work_dir, "statistics_radiomics.json")  
    with open(radiomics_file, "w") as f:
        json.dump(radiomics_data, f, indent=2)
    
    # Create some dummy segmentation masks
    segmentation_files = [
        "liver.nii.gz",
        "kidney_left.nii.gz", 
        "kidney_right.nii.gz"
    ]
    
    for filename in segmentation_files:
        filepath = os.path.join(temp_work_dir, filename)
        with open(filepath, "wb") as f:
            f.write(b"dummy nifti segmentation data" * 100)
    
    return temp_work_dir


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", 
        "slow: marks tests as slow (require external dependencies like TotalSegmentator)"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (require full system setup)"
    )


def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on naming patterns."""
    for item in items:
        # Mark tests requiring external tools as slow
        if any(keyword in item.name.lower() for keyword in ["totalsegmentator", "subprocess", "command"]):
            item.add_marker(pytest.mark.slow)
        
        # Mark end-to-end tests as integration
        if "integration" in item.name.lower() or "end_to_end" in item.name.lower():
            item.add_marker(pytest.mark.integration)