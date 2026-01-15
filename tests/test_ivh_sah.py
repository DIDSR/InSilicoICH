
import pytest
import numpy as np
from insilicoICH import lesion_definition as ld
from insilicoICH.lesion_definition import LesionFactory

# Fixtures
@pytest.fixture
def mock_csf_mask():
    """
    Creates a mock CSF mask with:
    1. Central box-like 'ventricles'
    2. Peripheral shell-like 'SAH/sulci'
    """
    shape = (100, 100, 100)
    mask = np.zeros(shape, dtype=bool)
    
    # 1. Ventricles (Central Box)
    # Approx 20x20x20 = 8000 voxels
    mask[40:60, 40:60, 40:60] = True
    
    # 2. SAH (Peripheral Shell spots)
    z, y, x = np.ogrid[:100, :100, :100]
    center = (50, 50, 50)
    radius_sq = (z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2
    # Shell between r=40 and r=45
    shell = (radius_sq >= 40**2) & (radius_sq <= 45**2)
    # Make it sparse to look like sulci
    sparse_shell = shell & ((x + y + z) % 3 == 0)
    
    mask[sparse_shell] = True
    
    return mask

def test_partition_csf(mock_csf_mask):
    ventricles, sah = ld.partition_csf_to_ventricles_and_sah(mock_csf_mask)
    
    # Check disjointness
    assert not np.any(ventricles & sah), "Ventricle and SAH masks should be disjoint"
    
    # Check coverage
    # Note: partition might drop small isolated bits if logic filters them, 
    # but initially should verify major components.
    # The current implementation uses morphological opening which might erode small bits.
    # So we check that the major components are captured.
    
    # Ventricles should capture the central box
    # Center of image is 50,50,50. Central box is 40:60.
    assert ventricles[50, 50, 50], "Central voxel should be ventricle"
    assert np.sum(ventricles) > 1000, "Ventricle volume too small"
    
    # SAH should capture peripheral parts
    # A point on the shell, e.g., z=50, y=50, x=92 (r=42)
    # check if any point in the expected shell area is classified as SAH
    z, y, x = np.ogrid[:100, :100, :100]
    center = (50, 50, 50)
    r2 = (z - 50)**2 + (y - 50)**2 + (x - 50)**2
    shell_mask = (r2 > 40**2) & (r2 < 45**2)
    
    # There should be SAH voxels in the shell region
    assert np.sum(sah[shell_mask]) > 0, "No SAH detected in peripheral region"
    
    # Ventricles should NOT be in the shell
    assert np.sum(ventricles[shell_mask]) == 0, "Ventricle mask extends into periphery"

def test_ivh_generation(mock_csf_mask):
    ventricles, _ = ld.partition_csf_to_ventricles_and_sah(mock_csf_mask)
    
    lesion = LesionFactory.create('IVH', boundary=ventricles, spacings=(1, 1, 1), seed=42)
    target_vol = 5.0 # mL = 5000 voxels
    lesion.generate(volume_ml=target_vol)
    
    assert lesion.mask is not None
    assert lesion.mask.sum() > 0
    
    # Fluid level check:
    # IVH fills from bottom (lowest z) up.
    # If mask is present at z=Z, it should be present at z=Z-1 (mostly)
    # Find min and max z of the lesion
    lesion_coords = np.argwhere(lesion.mask)
    min_z = lesion_coords[:, 0].min()
    max_z = lesion_coords[:, 0].max()

    # Check that a slice in the middle is fuller than a slice above?
    # Or simpler: check that it didn't fill the top of the ventricle box if volume was small.
    ventricle_coords = np.argwhere(ventricles)
    max_v_z = ventricle_coords[:, 0].max()

    # If target volume < total ventricle volume, we shouldn't reach the top
    total_vent_vol = np.sum(ventricles) * 0.001 # approx L
    if target_vol < total_vent_vol:
        assert max_z < max_v_z, "IVH should not fill entire ventricle if target volume is small (fluid level effect)"

def test_sah_generation(mock_csf_mask):
    _, sah = ld.partition_csf_to_ventricles_and_sah(mock_csf_mask)
    
    # Ensure we have enough SAH space
    if np.sum(sah) == 0:
        pytest.skip("Mock SAH mask is empty")

    lesion = LesionFactory.create('SAH', boundary=sah, spacings=(1, 1, 1), seed=42)
    target_vol = 1.0 
    lesion.generate(volume_ml=target_vol)
    
    assert lesion.mask is not None
    assert lesion.mask.sum() > 0
    
    # Constraint check
    # Lesion mask must be subset of boundary (SAH space)
    assert np.all(sah[lesion.mask]), "SAH lesion leaked outside boundary"
