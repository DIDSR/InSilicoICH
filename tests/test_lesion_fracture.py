import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath('src/insilicoICH'))
from lesion_definition import FractureLesion

def test_fracture_lesion_instantiation():
    skull_shape = (50, 50, 50)
    skull = np.zeros(skull_shape, dtype=int)
    # Create a simple skull shell
    center = np.array(skull_shape) / 2
    Z, Y, X = np.ogrid[:50, :50, :50]
    dist = np.sqrt((X-center[2])**2 + (Y-center[1])**2 + (Z-center[0])**2)
    skull[(dist >= 20) & (dist <= 25)] = 1
    
    lesion = FractureLesion('Fracture', skull, spacings=(1, 1, 1))
    assert lesion.lesion_type == 'Fracture'
    assert np.array_equal(lesion.skull, skull)

def test_fracture_generation():
    skull_shape = (100, 100, 100)
    skull = np.zeros(skull_shape, dtype=int)
    # Create a simple spherical skull
    center = np.array(skull_shape) / 2
    Z, Y, X = np.ogrid[:100, :100, :100]
    dist_from_center = np.sqrt((X - center[2])**2 + (Y-center[1])**2 + (Z-center[0])**2)
    skull[(dist_from_center > 40) & (dist_from_center < 45)] = 1
    
    lesion = FractureLesion('Fracture', skull, spacings=(1, 1, 1), seed=42)
    
    # Generate fracture
    lesion.generate(fracture_length=50, thickness=2)
    
    # Check if mask is generated
    assert lesion.mask is not None
    assert lesion.image is not None
    
    # Check if mask is within skull
    # lesion.mask is boolean, skull is int (0 or 1). 
    # lesion.mask should only be True where skull is 1.
    assert np.all(lesion.mask <= (skull > 0)) 
    
    # Check volume
    assert lesion.volume_ml > 0
    
    # Check reproducibility
    mask1 = lesion.mask.copy()
    lesion.generate(fracture_length=50, thickness=2)
    # Since we re-used same seed in generate (via same instance rng state? No, rng advances)
    # Re-instantiate for strict reproducibility check or reset seed?
    # The class initializes rng with seed. Each call advances it.
    
    lesion2 = FractureLesion('Fracture', skull, spacings=(1, 1, 1), seed=42)
    lesion2.generate(fracture_length=50, thickness=2)
    assert np.array_equal(mask1, lesion2.mask)

def test_fracture_thickness():
    skull = np.ones((50, 50, 50), dtype=int)
    lesion = FractureLesion('Fracture', skull,  seed=123)
    
    lesion.generate(fracture_length=20, thickness=1)
    vol1 = lesion.volume_ml
    
    # Reset rng for fair comparison (though path might differ if rng is used for thickness dilation? No)
    lesion2 = FractureLesion('Fracture', skull, seed=123)
    lesion2.generate(fracture_length=20, thickness=3)
    vol2 = lesion2.volume_ml
    
    assert vol2 > vol1
