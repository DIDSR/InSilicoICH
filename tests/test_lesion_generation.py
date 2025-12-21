
import pytest
import numpy as np
from insilicoICH.lesion_definition import LesionFactory, DuralLesion, RoundLesion, FractureLesion

# Fixtures
@pytest.fixture
def mock_dura_boundary():
    shape = (100, 100, 100)
    boundary = np.zeros(shape, dtype=bool)
    z, y, x = np.ogrid[:100, :100, :100]
    center = (50, 50, 50)
    # Hollow sphere (shell)
    dist_sq = (z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2
    boundary[(dist_sq <= 45**2) & (dist_sq >= 44**2)] = True
    return boundary

@pytest.fixture
def mock_brain_boundary():
    shape = (100, 100, 100)
    z, y, x = np.ogrid[:100, :100, :100]
    center = (50, 50, 50)
    mask = ((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2) <= 40**2
    return mask

@pytest.fixture
def mock_skull_boundary():
    shape = (100, 100, 100)
    boundary = np.zeros(shape, dtype=np.uint8) 
    z, y, x = np.ogrid[:100, :100, :100]
    center = (50, 50, 50)
    mask = ((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2) <= 45**2
    boundary[mask] = 1
    return boundary

def test_dural_lesion_generation(mock_dura_boundary):
    # Use explicit safe seed
    lesion = LesionFactory.create('EDH', boundary=mock_dura_boundary, spacings=(1, 1, 1), seed=20624)
    target_vol = 10.0
    lesion.generate(volume_ml=target_vol, intensity_hu=60, texture_contrast=0)
    
    assert lesion.mask is not None
    assert lesion.mask.sum() > 0
    assert abs(lesion.volume_ml - target_vol) / target_vol < 0.25 
    assert abs(lesion.intensity_HU - 60) < 1.0

def test_round_lesion_generation(mock_brain_boundary):
    lesion = LesionFactory.create('IPH', boundary=mock_brain_boundary, spacings=(1, 1, 1), seed=20624)
    target_vol = 15.0
    lesion.generate(volume_ml=target_vol, intensity_hu=50)
    
    assert lesion.mask is not None
    assert lesion.mask.sum() > 0
    assert abs(lesion.volume_ml - target_vol) / target_vol < 0.1
    assert abs(lesion.intensity_HU - 50) < 1.0

def test_fracture_lesion_generation(mock_skull_boundary):
    lesion = LesionFactory.create('Fracture', boundary=mock_skull_boundary, spacings=(1, 1, 1))
    lesion.generate(fracture_length=50, thickness=2)
    
    assert lesion.mask is not None
    assert lesion.volume_ml > 0
    assert np.any(mock_skull_boundary[lesion.mask] > 0)

def test_lesion_factory_invalid_type():
    with pytest.raises(ValueError):
        LesionFactory.create('InvalidType')
