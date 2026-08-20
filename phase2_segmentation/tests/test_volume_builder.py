"""
test_volume_builder.py — Unit tests for NIfTI volume assembly

Tests:
    - Volume array validation (shape, dtype, HU ranges)
    - Spacing validation (positive values, reasonable ranges)
    - SimpleITK integration (spacing order conversion)
    - File I/O operations
    - Error handling for corrupted inputs
"""

import os
import tempfile
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from phase2_segmentation.volume_builder import (
    validate_volume_array,
    validate_spacing,
    assemble_nifti,
)


class TestVolumeValidation:
    """Test volume array validation logic."""

    def test_valid_volume(self):
        """Test that a valid volume passes validation."""
        volume = np.random.randn(50, 256, 256).astype(np.float32) * 100
        validate_volume_array(volume)  # Should not raise

    def test_none_volume(self):
        """Test that None volume raises ValueError."""
        with pytest.raises(ValueError, match="volume_array is None"):
            validate_volume_array(None)

    def test_wrong_dimensions(self):
        """Test that non-3D arrays raise ValueError."""
        # 2D array
        with pytest.raises(ValueError, match="must be 3D"):
            validate_volume_array(np.zeros((256, 256)))
        
        # 4D array
        with pytest.raises(ValueError, match="must be 3D"):
            validate_volume_array(np.zeros((10, 50, 256, 256)))

    def test_insufficient_slices(self):
        """Test that volumes with too few slices raise ValueError."""
        with pytest.raises(ValueError, match="only 1 slices"):
            validate_volume_array(np.zeros((1, 256, 256)))

    def test_too_small_spatial(self):
        """Test that volumes with tiny spatial dimensions raise ValueError."""
        with pytest.raises(ValueError, match="too small"):
            validate_volume_array(np.zeros((10, 8, 8)))

    def test_empty_volume(self):
        """Test that empty volumes raise ValueError (size 0 or too-small dims)."""
        # np.zeros((0,0,0)) triggers ndim==3 but size==0; hits the size check
        # However shape (0,0,0) also triggers z<2 first — either error is valid
        with pytest.raises(ValueError):
            validate_volume_array(np.zeros((0, 0, 0)))

    def test_constant_volume(self):
        """Test that volumes with constant values raise ValueError."""
        volume = np.full((10, 64, 64), 42.0)
        with pytest.raises(ValueError, match="constant value"):
            validate_volume_array(volume)

    def test_extreme_hu_values(self, caplog):
        """Test that extreme HU values generate warnings but don't fail."""
        # Very negative values
        volume = np.full((10, 64, 64), -3000.0)
        volume[0, 0, 0] = -2500.0  # Make it non-constant
        
        validate_volume_array(volume)
        assert "extreme HU values" in caplog.text


class TestSpacingValidation:
    """Test spacing tuple validation logic."""

    def test_valid_spacing(self):
        """Test that valid spacing passes validation."""
        validate_spacing((0.7, 0.7, 1.5))  # Should not raise

    def test_none_spacing(self):
        """Test that None spacing raises ValueError."""
        with pytest.raises(ValueError, match="spacing_meta is None"):
            validate_spacing(None)

    def test_wrong_length(self):
        """Test that non-3-tuple raises ValueError."""
        with pytest.raises(ValueError, match="must be a 3-tuple"):
            validate_spacing((0.7, 0.7))
        
        with pytest.raises(ValueError, match="must be a 3-tuple"):
            validate_spacing((0.7, 0.7, 1.5, 2.0))

    def test_non_numeric_values(self):
        """Test that non-numeric spacing raises ValueError."""
        with pytest.raises(ValueError, match="must be numeric"):
            validate_spacing(("0.7", 0.7, 1.5))

    def test_zero_or_negative_spacing(self):
        """Test that zero/negative spacing raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_spacing((0.0, 0.7, 1.5))
        
        with pytest.raises(ValueError, match="must be positive"):
            validate_spacing((0.7, -0.5, 1.5))

    def test_unusually_large_spacing(self, caplog):
        """Test that very large spacing generates warnings."""
        validate_spacing((60.0, 0.7, 1.5))
        assert "unusually large" in caplog.text


class TestNIfTIAssembly:
    """Test NIfTI file assembly and SimpleITK integration."""

    def setup_method(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_volume = np.random.randn(20, 128, 128).astype(np.float32) * 100
        self.test_spacing = (0.7, 0.7, 2.0)

    def test_successful_assembly(self):
        """Test successful NIfTI assembly with valid inputs."""
        out_path = os.path.join(self.temp_dir, "test.nii.gz")
        
        result_path = assemble_nifti(self.test_volume, self.test_spacing, out_path)
        
        assert result_path == os.path.abspath(out_path)
        assert os.path.isfile(result_path)
        assert os.path.getsize(result_path) > 1000  # Non-empty file

    def test_spacing_conversion(self):
        """Test that spacing is correctly converted from (row,col,z) to SimpleITK (col,row,z).
        
        Uses a real round-trip via SimpleITK to verify the swap is correct.
        row_mm and col_mm must be deliberately different to catch the swap.
        """
        import SimpleITK as sitk
        out_path = os.path.join(self.temp_dir, "spacing_swap_test.nii.gz")
        
        row_mm, col_mm, z_mm = 0.8, 0.6, 2.0
        spacing = (row_mm, col_mm, z_mm)
        
        result_path = assemble_nifti(self.test_volume, spacing, out_path)
        
        img = sitk.ReadImage(result_path)
        sitk_spacing = img.GetSpacing()  # Returns (X, Y, Z) = (col, row, z)
        
        # SimpleITK X = col_mm, Y = row_mm, Z = z_mm
        assert abs(sitk_spacing[0] - col_mm) < 1e-5, (
            f"X spacing should be col_mm={col_mm}, got {sitk_spacing[0]}"
        )
        assert abs(sitk_spacing[1] - row_mm) < 1e-5, (
            f"Y spacing should be row_mm={row_mm}, got {sitk_spacing[1]}"
        )
        assert abs(sitk_spacing[2] - z_mm) < 1e-5, (
            f"Z spacing should be z_mm={z_mm}, got {sitk_spacing[2]}"
        )

    def test_orientation_handling(self):
        """Test DICOM orientation cosines produce a valid 3x3 direction matrix.
        
        Uses a real round-trip: write NIfTI with explicit orientation, read back,
        and verify the direction is a 9-element tuple.
        """
        import SimpleITK as sitk
        out_path = os.path.join(self.temp_dir, "orientation_test.nii.gz")
        
        # Standard axial orientation cosines: row=[1,0,0], col=[0,1,0]
        orientation = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        
        result_path = assemble_nifti(
            self.test_volume,
            self.test_spacing,
            out_path,
            orientation_patient=orientation
        )
        
        img = sitk.ReadImage(result_path)
        direction = img.GetDirection()  # Should be a 9-element tuple
        assert len(direction) == 9, f"Expected 9-element direction, got {len(direction)}"

    def test_missing_sitk_import(self):
        """Test graceful handling when SimpleITK is not available.
        
        Patches the import inside the function using sys.modules to simulate
        SimpleITK not being installed.
        """
        import sys
        original = sys.modules.get("SimpleITK")
        sys.modules["SimpleITK"] = None  # Forces ImportError on 'import SimpleITK'
        try:
            with pytest.raises((RuntimeError, ImportError)):
                assemble_nifti(self.test_volume, self.test_spacing, "test_missing.nii.gz")
        finally:
            if original is None:
                sys.modules.pop("SimpleITK", None)
            else:
                sys.modules["SimpleITK"] = original

    def test_sitk_write_failure(self):
        """Test handling of bad output path (write failure).
        
        Points assemble_nifti at an invalid path to trigger a write failure.
        """
        # Use a path that exists as a directory (can't write a file there)
        out_path = os.path.join(self.temp_dir, "subdir") + os.sep
        os.makedirs(out_path, exist_ok=True)
        bad_path = os.path.join(out_path, "", "test.nii.gz")  # same dir, valid path
        # Instead: use a truly invalid path — null byte in name (OS-level rejection)
        # On Windows the easiest is write to a path inside a file (not a dir)
        # Create a file, try to write into it as if it's a directory
        blocker = os.path.join(self.temp_dir, "blocker.txt")
        open(blocker, "w").close()
        bad_child = os.path.join(blocker, "child.nii.gz")  # parent exists as file
        with pytest.raises((RuntimeError, OSError, Exception)):
            assemble_nifti(self.test_volume, self.test_spacing, bad_child)

    def test_directory_creation(self):
        """Test that parent directories are created if they don't exist."""
        nested_path = os.path.join(self.temp_dir, "deep", "nested", "test.nii.gz")
        
        result_path = assemble_nifti(self.test_volume, self.test_spacing, nested_path)
        
        assert os.path.isfile(result_path)
        assert os.path.dirname(result_path) == os.path.dirname(nested_path)

    def teardown_method(self):
        """Clean up temporary files."""
        import shutil
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


@pytest.fixture
def temp_dir():
    """Fixture providing a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    import shutil
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_volume():
    """Fixture providing a valid test volume."""
    return np.random.randn(20, 128, 128).astype(np.float32) * 100


@pytest.fixture  
def sample_spacing():
    """Fixture providing valid test spacing."""
    return (0.7, 0.7, 2.0)