import numpy as np
import pytest
from insilicoICH.lesion_definition import FractureLesion

@pytest.fixture
def synthetic_skull():
    """Creates a synthetic spherical skull mask and spacings."""
    shape = (64, 64, 64)
    spacings = (1.0, 1.0, 1.0)
    
    # Create a hollow sphere
    z, y, x = np.ogrid[-32:32, -32:32, -32:32]
    radius_sq = x**2 + y**2 + z**2
    # Skull between radius 25 and 30
    skull_mask = (radius_sq < 30**2) & (radius_sq > 25**2)
    
    return skull_mask, spacings

def test_fracture_generation_defaults(synthetic_skull):
    """Test basic fracture generation with default parameters."""
    skull_mask, spacings = synthetic_skull
    fracture = FractureLesion('linear fracture', skull_mask, spacings, seed=42)
    fracture.generate(fracture_length=50)
    
    assert fracture.mask is not None
    assert fracture.mask.shape == skull_mask.shape
    assert fracture.mask.dtype == bool
    assert np.any(fracture.mask), "Fracture mask should not be empty"
    
    # The current implementation ensures the fracture is within the skull
    assert np.all(skull_mask[fracture.mask]), "Fracture must be contained within the skull mask"

def test_fracture_generation_parameters(synthetic_skull):
    """Test fracture generation with specific parameters."""
    skull_mask, spacings = synthetic_skull
    # Use a fixed seed for reproducibility in tests
    fracture = FractureLesion('linear fracture', skull_mask, spacings, seed=123)
    
    # Test with specific angles, length, and thickness
    length = 40
    thickness = 2
    fracture.generate(fracture_length=length, phi=30, theta=45, thickness=thickness)
    
    assert np.any(fracture.mask)
    assert fracture.volume_ml > 0
    
    # Check if thickness logic produced enough voxels (heuristic check)
    # A single line of 40 voxels would be ~40 voxels. Thickness should increase this.
    assert np.sum(fracture.mask) > length

def test_fracture_consistency(synthetic_skull):
    """Test that specifying a seed produces deterministic results."""
    skull_mask, spacings = synthetic_skull
    seed = 99
    
    f1 = FractureLesion('linear fracture', skull_mask, spacings, seed=seed)
    f1.generate(fracture_length=50)
    
    f2 = FractureLesion('linear fracture', skull_mask, spacings, seed=seed)
    f2.generate(fracture_length=50)
    
    np.testing.assert_array_equal(f1.mask, f2.mask)

def test_fracture_lesion_properties(synthetic_skull):
    """Test that properties are correctly set after generation."""
    skull_mask, spacings = synthetic_skull
    fracture = FractureLesion('linear fracture', skull_mask, spacings, seed=10)
    fracture.generate(fracture_length=30)
    
    # Intensity should be 0 by definition in FractureLesion
    assert fracture.intensity_hu == 0
    
    # Center of mass should be calculated and 3-dimensional
    assert hasattr(fracture, 'coords_voxel')
    assert len(fracture.coords_voxel) == 3
    
    # Volume should be calculated
    assert fracture.volume_ml > 0
    expected_vol = np.sum(fracture.mask) * (spacings[0] * spacings[1] * spacings[2]) / 1000.0
    assert np.isclose(fracture.volume_ml, expected_vol)

def test_fracture_invalid_input(synthetic_skull):
    """Test behavior with likely invalid inputs if edge cases are handled."""
    skull_mask, spacings = synthetic_skull
    fracture = FractureLesion('linear fracture', skull_mask, spacings)
    
    # Passing 0 length should probably result in empty or minimal mask, or error
    # Checking current implementation: 
    # it passes length to random walk. range(length) -> range(0). Loop doesn't run.
    # projected_fractures remains zeros.
    fracture.generate(fracture_length=0)
    assert np.sum(fracture.mask) == 0
