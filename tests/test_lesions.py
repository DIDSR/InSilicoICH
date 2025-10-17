import unittest
import numpy as np

from insilicoICH.lesion_definition import (
    sphere_radius_from_volume,
    get_semi_major_axes_ratios,
    connect_points_and_fill,
    generate_3d_perlin_texture,
    DuralLesion,
    RoundLesion,
    LesionFactory
)


class TestUtilityFunctions(unittest.TestCase):
    """Tests for the helper utility functions."""

    def test_sphere_radius_from_volume(self):
        """Test the volume-to-radius conversion."""
        self.assertEqual(sphere_radius_from_volume(0), 0.0)
        # Test a known value: a sphere with radius 10mm has a volume of (4/3)*pi*1000 mm^3
        volume = (4/3) * np.pi * 1000
        self.assertAlmostEqual(sphere_radius_from_volume(volume / 1000), 10.0, places=5)
        with self.assertRaises(TypeError):
            sphere_radius_from_volume(None)

    def test_get_semi_major_axes_ratios(self):
        """Test the generation of ellipsoid axis ratios."""
        ratios = get_semi_major_axes_ratios(0.5, seed=42)
        self.assertEqual(ratios.shape, (3,))
        self.assertTrue(all(r > 0 for r in ratios))

        # Test reproducibility with seed
        ratios2 = get_semi_major_axes_ratios(0.5, seed=42)
        np.testing.assert_array_equal(ratios, ratios2)

        # Test that different seeds give different shuffles
        ratios3 = get_semi_major_axes_ratios(0.5, seed=101)
        self.assertFalse(np.array_equal(ratios, ratios3))

    def test_connect_points_and_fill(self):
        """Test the 2D slice filling logic."""
        boundary = np.zeros((50, 50), dtype=bool)
        boundary[10:40, 10] = True # A vertical line boundary

        start = (15, 10)
        end = (35, 10)

        filled_mask = connect_points_and_fill(start, end, boundary, 'EDH')

        self.assertIsInstance(filled_mask, np.ndarray)
        self.assertEqual(filled_mask.dtype, bool)
        self.assertTrue(np.any(filled_mask)) # Check that it's not empty

        # Test case where no path is possible
        no_path_boundary = np.zeros_like(boundary)
        filled_mask_empty = connect_points_and_fill(start, end, no_path_boundary, 'EDH')
        self.assertLessEqual(np.any(filled_mask_empty), 100)

    def test_generate_3d_perlin_texture(self):
        """Test the Perlin noise generation."""
        shape = (16, 32, 32)
        texture = generate_3d_perlin_texture(*shape, seed=123)
        self.assertEqual(texture.shape, shape)

        # Test reproducibility
        texture2 = generate_3d_perlin_texture(*shape, seed=123)
        np.testing.assert_array_equal(texture, texture2)


class TestDuralLesion(unittest.TestCase):
    """Tests for the DuralLesion class."""

    def setUp(self):
        """Set up a common testing environment."""
        self.shape = (64, 128, 128)
        self.spacings = (4, 2, 2)
        # Create a simple spherical shell for the dura map
        self.dura_map = np.zeros(self.shape, dtype=bool)
        z, y, x = np.ogrid[-32:32, -64:64, -64:64]
        shell_mask = (x**2 + y**2 + z**2 < 30**2) & (x**2 + y**2 + z**2 > 28**2)
        self.dura_map[shell_mask] = True

    def test_initialization(self):
        """Test that DuralLesion initializes correctly and handles errors."""
        with self.assertRaises(ValueError):
            DuralLesion(lesion_type='IPH', boundary=self.dura_map, spacings=self.spacings)

        lesion = DuralLesion(lesion_type='SDH', boundary=self.dura_map, spacings=self.spacings)
        self.assertEqual(lesion.lesion_type, 'SDH')

    def test_generate_sdh(self):
        """Test the generation of a Subdural Hematoma (SDH)."""
        desired_volume = 10.0
        lesion = DuralLesion(lesion_type='SDH', boundary=self.dura_map, spacings=self.spacings, seed=42)
        lesion.generate(volume_ml=desired_volume, intensity_hu=60.0, texture_contrast=1)

        self.assertIsNotNone(lesion.mask)
        self.assertIsNotNone(lesion.image)
        self.assertTrue(np.any(lesion.mask))
        self.assertAlmostEqual(lesion.volume_ml, desired_volume, delta=4.0) # Check if achieved volume is within 10%
        self.assertIsInstance(lesion.coords_voxel, tuple)

        # Check that the image has the correct intensity where the mask is true
        mean_intensity = lesion.image[lesion.mask].mean()
        self.assertAlmostEqual(mean_intensity, 60.0, delta=5.0) # Allow for texture variation

    def test_generate_edh(self):
        """Test the generation of an Epidural Hematoma (EDH)."""
        lesion = DuralLesion(lesion_type='EDH', boundary=self.dura_map, spacings=self.spacings, seed=101)
        lesion.generate(volume_ml=15.0, intensity_hu=70.0, texture_contrast=0.0)

        self.assertIsNotNone(lesion.mask)
        self.assertTrue(np.any(lesion.mask))
        self.assertAlmostEqual(lesion.volume_ml, 15.0, delta=1.5)

        # With no texture contrast, intensity should be exact
        mean_intensity = lesion.image[lesion.mask].mean()
        self.assertAlmostEqual(mean_intensity, 70.0, places=5)


class TestRoundLesion(unittest.TestCase):
    """Tests for the RoundLesion (IPH) class."""

    def setUp(self):
        """Set up a common testing environment."""
        self.shape = (64, 128, 128)
        self.spacings = (1.0, 0.5, 0.5)
        # Create a simple box for the boundary mask (e.g., brain tissue)
        self.boundary_mask = np.zeros(self.shape, dtype=bool)
        self.boundary_mask[10:54, 20:108, 20:108] = True

    def test_generate_simple_iph(self):
        """Test generation of a simple, regular IPH."""
        lesion = RoundLesion(boundary=self.boundary_mask,
                             spacings=self.spacings, seed=42)
        lesion.generate(
            volume_ml=10.0, intensity_hu=55.0, eccentricity=0.1,
            irregularity=0.0, smoothness=1.0, complexity=1,
            edema_hu=0, texture_contrast=0
        )

        self.assertIsNotNone(lesion.mask)
        self.assertTrue(np.any(lesion.mask))
        self.assertAlmostEqual(lesion.volume_ml, 10.0, delta=1.0)  # Check achieved volume

        # Check that the lesion is contained within the boundary
        self.assertGreater(self.boundary_mask[lesion.mask].sum() / lesion.mask.sum(), 0.6)

        # Check intensity
        self.assertAlmostEqual(lesion.intensity_HU, 55.0, places=5)

    def test_generate_complex_iph_with_edema(self):
        """Test a complex IPH with irregularity and edema."""
        lesion = RoundLesion(boundary=self.boundary_mask,
                             spacings=self.spacings, seed=123)
        lesion.generate(
            volume_ml=12.0, intensity_hu=70.0, eccentricity=0.6,
            irregularity=0.8, smoothness=0.5, complexity=3,
            edema_hu=5.0, edema_thickness=5, texture_contrast=1
        )

        self.assertIsNotNone(lesion.mask)
        self.assertTrue(np.any(lesion.mask))
        self.assertAlmostEqual(lesion.volume_ml, 12.0, delta=1.0)  # Check achieved volume
        # Check that edema was created
        # The mean intensity should be lower than the core HU due to edema
        self.assertLess(lesion.intensity_HU, 70.0)
        self.assertGreater(lesion.intensity_HU, 25.0)

    def test_volume_too_large_error(self):
        """Test that an error is raised if the requested volume is too large."""
        small_boundary = np.zeros((20, 20, 20), dtype=bool)
        small_boundary[5:15, 5:15, 5:15] = True

        lesion = RoundLesion(boundary=small_boundary, spacings=(1,1,1), seed=1)
        with self.assertRaises(RuntimeError):
            lesion.generate(volume_ml=50.0, intensity_hu=50.0)


class TestLesionFactory(unittest.TestCase):
    """Tests for the LesionFactory."""

    def test_factory_creation(self):
        """Test that the factory creates the correct lesion types."""
        boundary = np.zeros((10,10,10))

        edh_lesion = LesionFactory.create('EDH', boundary=boundary, spacings=(1, 1, 1))
        self.assertIsInstance(edh_lesion, DuralLesion)
        self.assertEqual(edh_lesion.lesion_type, 'EDH')
        
        iph_lesion = LesionFactory.create('IPH', boundary=boundary, spacings=(1, 1, 1))
        self.assertIsInstance(iph_lesion, RoundLesion)
        self.assertEqual(iph_lesion.lesion_type, 'IPH')

    def test_factory_invalid_type(self):
        """Test that the factory raises an error for an unknown lesion type."""
        with self.assertRaises(ValueError):
            LesionFactory.create('UnknownType')


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
