
import numpy as np
import pytest
from scipy.ndimage import zoom
import scipy.ndimage # ensure available
# Mock VITools before importing insilicoICH
import sys
from unittest.mock import MagicMock

# Create a more robust Mock for VITools
class MockVIToolsPhantom:
    ages = [0, 1]  # Dummy ages
    def __init__(self, *args, **kwargs):
        pass

# Mock VITools structure
mock_vitools = MagicMock()
mock_vitools.Phantom = MockVIToolsPhantom
sys.modules['VITools'] = mock_vitools
sys.modules['VITools.hooks'] = MagicMock()

from insilicoICH.transforms import resize, RandAffine, ResizeWithPadOrCrop, Affine

def create_phantom(shape=(100, 100, 100), radius=30, intensity=100):
    img = np.zeros(shape, dtype=np.float32)
    center = np.array(shape) / 2
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
    img[dist <= radius] = intensity
    return img

def test_resize_stats():
    phantom = create_phantom(shape=(50, 50, 50), radius=10)
    target_shape = (100, 100, 100)

    resized = resize(phantom, target_shape, order=1)

    assert np.allclose(resized.shape, target_shape)
    assert np.isclose(resized.mean(), phantom.mean(), rtol=0.1)
    assert np.isclose(resized.max(), phantom.max(), rtol=0.1)

def test_rand_affine_stats():
    phantom = create_phantom(shape=(64, 64, 64), radius=15)

    rotate_range = [np.pi/4, np.pi/20, np.pi/20]
    translate_range = [10, 10, 10]
    scale_range = [0.1, 0.1, 0.1]

    transform = RandAffine(
        prob=1.0,
        rotate_range=rotate_range,
        translate_range=translate_range,
        scale_range=scale_range,
        padding_mode="border",
        mode='nearest'
    )
    transform.set_random_state(seed=42)
    out = transform(phantom)

    # Check shape preserved
    assert out.shape == phantom.shape

    # Check volume (sum > threshold) conservation roughly
    vol_in = np.sum(phantom > 50)
    vol_out = np.sum(out > 50)

    vol_diff_percent = abs(vol_in - vol_out) / vol_in
    assert vol_diff_percent < 0.25

    # Determinism check
    transform.set_random_state(seed=42)
    out2 = transform(phantom)
    assert np.allclose(out, out2)

def test_resize_pad_crop():
    phantom = create_phantom(shape=(50, 50, 50), radius=10)

    # Crop
    cropper = ResizeWithPadOrCrop(spatial_size=(30, 30, 30))
    cropped = cropper(phantom)
    assert cropped.shape == (30, 30, 30)
    assert cropped[15, 15, 15] == 100 # Center preserved

    # Pad
    padder = ResizeWithPadOrCrop(spatial_size=(70, 70, 70))
    padded = padder(phantom)
    assert padded.shape == (70, 70, 70)
    assert padded[35, 35, 35] == 100

def test_affine_deterministic():
    phantom = create_phantom(shape=(50, 50, 50), radius=10)
    # Simple translation
    translate = [5, 0, 0]
    affine = Affine(translate_params=translate, padding_mode='zeros', mode='nearest')
    out = affine(phantom)

    # Check center has moved
    # axis 0 is Z. +5 translation.
    com_in = scipy.ndimage.center_of_mass(phantom)
    com_out = scipy.ndimage.center_of_mass(out)

    # Translation affects coordinates.
    # Center of mass should shift by translation vector.
    # Note: RandAffine implementation:
    # x_out = (R * S) * (x_in - center) + translation + center
    # If R=I, S=I: x_out = x_in + translation
    # So if we translate by +5, the image content moves by +5?
    # Or is it coordinate transform?
    # x_out = x_in + t
    # x_in = x_out - t
    # Input at (x) moves to Output at (x+t).
    # So COM should increase by t.

    # However, my implementation passes 'translate' directly.
    # Let's verify direction.

    diff0 = com_out[0] - com_in[0]
    print(f"COM Shift: {diff0}")

    assert abs(diff0 - 5) < 1.0
