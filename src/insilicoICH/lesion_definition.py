"""
Module responsible for the procedural generation of intracranial hemorrhage lesions.
"""
# --- Standard Library Imports ---
import abc
import math
from typing import Tuple, Dict, Type, Optional

# --- Third Party Imports ---
import numpy as np
import skimage as ski
from scipy.ndimage import (
    center_of_mass,
    distance_transform_edt
)
import noise

# --- Type Aliases for Clarity ---
Shape3D = Tuple[int, int, int]
Point3D = Tuple[int, int, int]

# =============================================================================
# 1. UTILITY FUNCTIONS
# Helper functions used across different lesion generation classes.
# =============================================================================


def sphere_radius_from_volume(volume_ml: float) -> float:
    """Converts volume in milliliters (mL) to a sphere's radius in millimeters (mm)."""
    if volume_ml <= 0:
        return 0.0
    # 1 mL = 1000 mm^3. Volume of sphere V = 4/3 * pi * r^3
    return np.cbrt(0.75 * volume_ml * 1000 / np.pi)


def get_semi_major_axes_ratios(eccentricity: float, seed: int = None) -> np.ndarray:
    """
    Generates three semi-major axis ratios for an ellipsoid with a target mean eccentricity.
    Uses a lookup table for pragmatic, fast generation.
    """
    # Maps mean eccentricity to a plausible set of axis ratios [a, b, c].
    eccentricity_map = {
        0.0: [1.0, 1.0, 1.0], 0.2: [1.0, 1.0, 0.9], 0.4: [1.0, 0.9, 0.8],
        0.6: [1.0, 0.8, 0.6], 0.8: [1.0, 0.7, 0.4], 0.9: [1.0, 0.6, 0.2]
    }
    closest_key = min(eccentricity_map.keys(), key=lambda k: abs(k - eccentricity))
    ratios = eccentricity_map[closest_key]
    rng = np.random.default_rng(seed)
    rng.shuffle(ratios)
    return np.array(ratios)


def connect_points_and_fill(
    start: Point3D, end: Point3D, boundary: np.ndarray, hematoma_type: str
) -> np.ndarray:
    """
    Creates a filled 2D lesion slice by connecting two points.
    One path follows the provided boundary, the other is a curved connector.
    """
    costs = np.where(boundary, 1, 1e6)  # High cost for non-boundary points
    try:
        path, _ = ski.graph.route_through_array(costs, start=start, end=end, fully_connected=True)
        boundary_coords = np.array(path)
    except (ValueError, IndexError):
        return np.zeros_like(boundary, dtype=bool) # Cannot find path

    if len(boundary_coords) < 2:
        return np.zeros_like(boundary, dtype=bool)

    # Define Bézier curve parameters for different shapes
    bezier_params = {
        'EDH': {'weight': 0.14, 'middle': (boundary.shape[0]/2, boundary.shape[1]/2)},
        'SDH': {'weight': 2.0, 'middle': boundary_coords[len(boundary_coords) // 2]},
    }
    params = bezier_params.get(hematoma_type, {'weight': 0.0, 'middle': boundary_coords[len(boundary_coords) // 2]})

    rr, cc = ski.draw.bezier_curve(
        r0=start[0], c0=start[1], r1=int(params['middle'][0]), c1=int(params['middle'][1]),
        r2=end[0], c2=end[1], weight=params['weight']
    )
    connect_coords = np.vstack((rr, cc)).T

    # Combine paths to form a closed loop polygon
    # Note: connect_coords is reversed to trace from end back to start
    full_polygon = np.concatenate((boundary_coords, connect_coords[::-1]), axis=0)

    # --- MAJOR IMPROVEMENT: Use polygon fill instead of convex hull ---
    # This preserves the concave shape of lesions like SDH, which is more realistic.
    return ski.draw.polygon2mask(boundary.shape, full_polygon)


def generate_3d_perlin_texture(depth=32, height=128, width=128, scale=50.0,
                               octaves=4, persistence=0.5, lacunarity=2.0,
                               seed=None) -> np.ndarray:
    """
    Generates a 3D texture using Perlin noise.

    Args:
        depth (int): The depth of the texture.
        height (int): The height of the texture.
        width (int): The width of the texture.
        scale (float): The 'zoom' level of the noise. Larger values result in lower frequency.
        octaves (int): The number of noise layers to combine.
        persistence (float): The amplitude multiplier for each subsequent octave.
        lacunarity (float): The frequency multiplier for each subsequent octave.
        seed (int, optional): Seed for reproducibility. If None, a random seed is used.
    Returns:
        np.ndarray: A 3D numpy array containing the generated texture.
    If `seed` is provided, it will be used to ensure reproducibility.
    If `seed` is None, a random seed will be used.
    """
    texture_array = np.zeros((depth, height, width))
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                texture_array[z][y][x] = noise.pnoise3(x / scale,
                                                       y / scale,
                                                       z / scale,
                                                       octaves=octaves,
                                                       persistence=persistence,
                                                       lacunarity=lacunarity,
                                                       repeatx=width,
                                                       repeaty=height,
                                                       repeatz=depth,
                                                       base=seed)
    return texture_array
# =============================================================================
# 2. ABSTRACT BASE CLASS FOR LESIONS
# Defines the common interface and shared functionality for all lesion types.
# =============================================================================


class Lesion(abc.ABC):
    """Abstract base class for a procedurally generated lesion."""

    def __init__(self, lesion_type: str, spacings: tuple, seed: Optional[int] = None, **kwargs):
        self.lesion_type = lesion_type
        self.dz, self.dx, self.dy = spacings
        self.voxel_volume_ml = (self.dx * self.dy * self.dz) / 1000.0
        self.volume_ml = 0.
        self.seed = seed if seed is not None else np.random.randint(0, 2**31 - 1)
        self.rng = np.random.default_rng(self.seed)

        # These will be populated by the generate() method
        self.mask: Optional[np.ndarray] = None
        self.image: Optional[np.ndarray] = None
        self.coords_voxel: Optional[Point3D] = None
        self.achieved_volume_ml: float = 0.0

    @abc.abstractmethod
    def generate(self, **kwargs) -> "Lesion":
        """
        Generates the lesion mask and textured image.
        This method must be implemented by all subclasses.
        It should populate self.mask, self.image, and other attributes.
        """
        pass

    def _get_noise_texture(self, shape: Shape3D, base_intensity: float, std_dev_pct: float, scale: float) -> np.ndarray:
        """Generates a 3D Perlin noise texture to simulate tissue heterogeneity."""
        if std_dev_pct <= 0:
            return np.full(shape, base_intensity, dtype=np.float32)

        noise_texture = generate_3d_perlin_texture(
            depth=shape[0], height=shape[1], width=shape[2],
            scale=scale, octaves=4, persistence=0.5, lacunarity=2.0,
            seed=self.seed
        )

        # Scale noise to be +/- 1, then apply std_dev and add to base intensity
        return base_intensity + (noise_texture * base_intensity * std_dev_pct)

    @property
    def intensity_HU(self) -> float:
        return self.image[self.mask].mean()

    def __repr__(self):
        """Provides a string representation of the lesion."""
        return (f"Lesion(type={self.lesion_type}, "
                f"volume_ml={self.volume_ml:.2f}, "
                f"coords_voxel={self.coords_voxel}, "
                f"seed={self.seed})")
# =============================================================================
# 3. CONCRETE LESION IMPLEMENTATIONS
# =============================================================================


class DuralLesion(Lesion):
    """Generates dural-based lesions (EDH, SDH) that conform to a boundary."""

    # Class constants for clarity
    SHAPE_CONFIG = {
        'EDH': {'length_vol_ratio': 4, 'coverage_log_coeff': 10.23, 'coverage_log_intercept': 19.09},
        'SDH': {'length_vol_ratio': 11, 'coverage_log_coeff': 10.38, 'coverage_log_intercept': 24.48},
    }

    def __init__(self, lesion_type: str, boundary: np.ndarray, **kwargs):
        super().__init__(lesion_type=lesion_type, **kwargs)
        if lesion_type not in self.SHAPE_CONFIG:
            raise ValueError(f"DuralLesion requires type in {list(self.SHAPE_CONFIG.keys())}")
        self.dura_map = boundary

    def generate(self, volume_ml: float, intensity_hu: float = 40,
                 texture_contrast: float = 0., texture_scale: float = 15, **kwargs) -> "DuralLesion":
        """
        Generates the 3D lesion mask and textured image for a dural-based hemorrhage.

        This method orchestrates the procedural generation of a dural lesion (EDH or SDH).
        The process begins by estimating the required z-axis coverage and central
        slice area based on the target volume and lesion type. It then enters an
        iterative loop to generate a 3D mask that accurately matches the target volume.

        In each iteration, it finds appropriate start and end points on a central
        slice of the dura map. A 2D lesion shape is created between these points
        and then propagated to adjacent slices with a tapering effect to create a
        realistic, three-dimensional object. The volume of this generated mask is
        measured, and if it is not within a 10% tolerance of the target volume,
        the parameters are adjusted, and the process repeats.

        Once a suitable mask is generated, a procedural noise texture is applied
        to create the final image. This method modifies the instance in place,
        setting the `mask`, `image`, `volume_ml`, and `coords_voxel` attributes.

        Args:
            volume_ml (float): The target volume of the lesion in milliliters (mL).
            intensity_hu (float): The mean Hounsfield Unit (HU) value for the
                lesion's texture.
            texture_contrast (float): The standard deviation of the texture as a
                percentage of the mean intensity. A value of 0 results in a
                uniform (flat) lesion intensity.
            texture_scale (float, optional): The characteristic scale of the noise
                texture features. Defaults to 15.
            **kwargs: Catches unused keyword arguments for forward compatibility.

        Returns:
            DuralLesion: The instance of the class (`self`), allowing for method chaining.
                         The instance's `mask`, `image`, `volume_ml`, and
                         `coords_voxel` attributes are populated.
        """
        config = self.SHAPE_CONFIG[self.lesion_type]

        # 1. Estimate lesion dimensions from volume
        z_coverage_mm = config['coverage_log_coeff'] * math.log(volume_ml) + config['coverage_log_intercept']
        num_slices = max(1, round(z_coverage_mm / self.dz))

        avg_area_mm2 = (volume_ml * 1000) / (num_slices * self.dz)
        center_slice_area_mm2 = 1.5 * avg_area_mm2

        # 2. Iteratively generate mask to match target volume
        final_mask = np.zeros_like(self.dura_map, dtype=bool)

        # Use a while loop for more robust volume correction
        max_tries, tries = 5, 0
        while tries < max_tries:
            start_pt, end_pt, center_z = self._find_initial_dural_endpoints(center_slice_area_mm2)

            # Propagate up and down from the center slice
            all_filled_arrays = {}
            for direction in [-1, 1]: # -1 for up, 1 for down
                prev_start, prev_end = start_pt, end_pt
                for j in range(int(np.ceil(num_slices / 2.0))):
                    slice_idx = center_z + (j * direction)
                    if not (0 <= slice_idx < self.dura_map.shape[0]): break

                    # On the first slice (j=0), don't taper
                    scale = max(0, 1 - (j / (num_slices / 2.0))**2) if j > 0 else 1.0

                    filled, prev_start, prev_end = self._propagate_and_taper_slice(
                        self.dura_map[slice_idx], prev_start, prev_end, scale
                    )
                    if filled is None:
                        break
                    all_filled_arrays[slice_idx] = filled

            current_mask = np.zeros_like(self.dura_map, dtype=bool)
            for idx, filled_array in all_filled_arrays.items():
                current_mask[idx] = filled_array

            current_volume_ml = np.sum(current_mask) * self.voxel_volume_ml

            if abs(current_volume_ml - volume_ml) / volume_ml < 0.10:  # 10% tolerance
                final_mask = current_mask
                break

            # Adjust area estimate for next try
            if current_volume_ml > 0:
                correction = volume_ml / current_volume_ml
                center_slice_area_mm2 *= correction
            tries += 1
        else:  # If loop finishes without breaking
            final_mask = current_mask  # Keep the last attempt

        self.mask = final_mask
        self.volume_ml = np.sum(self.mask) * self.voxel_volume_ml
        self.coords_voxel = tuple(map(int, center_of_mass(self.mask)))

        # 3. Generate textured image
        texture = self._get_noise_texture(self.mask.shape, intensity_hu,
                                          texture_contrast,
                                          scale=texture_scale)
        self.image = np.where(self.mask, texture, 0).astype(np.float32)
        self.intensity_hu = intensity_hu
        return self

    def _find_initial_dural_endpoints(self, center_slice_area_mm2: float) -> Tuple[Point3D, Point3D, int]:
        """Finds a valid starting slice and two endpoints for the dural lesion."""
        config = self.SHAPE_CONFIG[self.lesion_type]
        desired_dist_mm = np.sqrt(config['length_vol_ratio'] * center_slice_area_mm2)
        desired_dist_vox = desired_dist_mm / self.dx

        valid_slice_indices = np.where(self.dura_map.sum(axis=(1, 2)) > desired_dist_vox)[0]
        if len(valid_slice_indices) == 0:
            raise RuntimeError("Could not find any suitable slices for dural lesion.")

        for _ in range(50):  # Try 50 times to find a good pair
            init_slice_idx = self.rng.choice(valid_slice_indices)
            dura_points = np.argwhere(self.dura_map[init_slice_idx])
            if len(dura_points) < 2:
                continue

            start_point = dura_points[self.rng.integers(len(dura_points))]
            distances = np.linalg.norm(dura_points - start_point, axis=1)

            valid_ends = dura_points[np.isclose(distances, desired_dist_vox, atol=max(5.0, desired_dist_vox * 0.1))]
            if len(valid_ends) > 0:
                end_point = valid_ends[self.rng.integers(len(valid_ends))]
                return start_point, end_point, init_slice_idx

        raise RuntimeError('Failed to find suitable start/end points for the requested lesion size.')

    def _propagate_and_taper_slice(self, dura_slice_map, prev_start, prev_end, scale_factor):
        """Propagates a lesion to an adjacent slice, scaling its cross-section."""
        dura_points = np.argwhere(dura_slice_map)
        if len(dura_points) < 2:
            return None, None, None

        new_start = dura_points[np.argmin(np.linalg.norm(dura_points - prev_start, axis=1))]
        new_end = dura_points[np.argmin(np.linalg.norm(dura_points - prev_end, axis=1))]
        if np.array_equal(new_start, new_end): return None, None, None

        costs = np.where(dura_slice_map, 1, 1e6)
        try:
            path, _ = ski.graph.route_through_array(costs, start=tuple(new_start), end=tuple(new_end))
        except (ValueError, IndexError): return None, None, None

        path = np.array(path)
        if len(path) < 2:
            return None, None, None

        new_len = int(len(path) * scale_factor)
        if new_len < 2:
            return None, None, None

        trim = (len(path) - new_len) // 2
        final_start, final_end = path[trim], path[len(path) - 1 - trim]

        filled_array = connect_points_and_fill(final_start, final_end, dura_slice_map, self.lesion_type)
        return filled_array, final_start, final_end


class RoundLesion(Lesion):
    """Generates parenchymal lesions (IPH) using deformed implicit surfaces."""

    def __init__(self, boundary: np.ndarray, lesion_type='IPH', **kwargs):
        super().__init__(lesion_type=lesion_type, **kwargs)
        self.boundary_mask = boundary

    def generate(self, volume_ml: float, intensity_hu: float, eccentricity: float = 0.5,
                 irregularity: float = 0.5, smoothness: float = 0.5, complexity: int = 3,
                 edema_hu: float = 0, edema_thickness: int = 5, texture_contrast: float = 0,
                 texture_scale: float = 10.0, **kwargs) -> "RoundLesion":
        """
        Generates the 3D mask and textured image for an intraparenchymal hemorrhage.

        This method uses a sophisticated procedural technique based on deformed
        implicit surfaces to create realistic, irregular lesion shapes. The core
        process is as follows:

        1.  **Placement**: A valid center for the lesion is randomly chosen within
            the provided `boundary_mask`, ensuring the lesion fits.
        2.  **Base Shape**: One or more base ellipsoids are defined. Their shape is
            controlled by `volume_ml` and `eccentricity`. If `complexity` is
            greater than one, multiple smaller, slightly offset ellipsoids are used.
        3.  **Deformation**: Each ellipsoid's surface is treated as an implicit
            surface and is deformed using 3D Perlin noise. The `irregularity`
            parameter controls the amplitude of this deformation, while `smoothness`
            controls the frequency (scale) of the noise features.
        4.  **Volume Matching**: The deformed implicit surfaces are combined, and a
            binary search algorithm is used to find the precise threshold that
            creates a 3D mask matching the target `volume_ml`.
        5.  **Edema & Texture**: An optional layer of perihematomal edema is added
            around the core lesion, with a smooth intensity falloff. A final
            heterogeneous texture is applied to the lesion core itself.

        This method modifies the instance in place, setting the `mask`, `image`,
        `volume_ml`, and `coords_voxel` attributes.

        Args:
            volume_ml (float): The target volume of the lesion's core in mL.
            intensity_hu (float): The mean Hounsfield Unit (HU) for the lesion core.
            eccentricity (float, optional): Controls the elongation of the base
                ellipsoid(s). 0 is a sphere, ~0.9 is highly elongated. Defaults to 0.5.
            irregularity (float, optional): The magnitude of surface deformation.
                0 results in a smooth ellipsoid. Defaults to 0.5.
            smoothness (float, optional): The scale of the surface noise features.
                Higher values create larger, smoother bumps. Defaults to 0.5.
            complexity (int, optional): The number of overlapping deformed ellipsoids
                to combine for the final shape. Defaults to 3.
            edema_hu (float, optional): The peak HU value of the surrounding edema.
                If 0, no edema is added. Defaults to 0.
            edema_thickness (int, optional): The thickness of the edema layer in voxels.
                Defaults to 5. Adding edema thickness will keep a constant
                volume but may decrease average attenuation as a result.
            texture_contrast (float, optional): The standard deviation of the core
                texture as a percentage of `intensity_hu`. 0 creates a flat texture.
                Defaults to 0.
            texture_scale (float, optional): The characteristic scale of the noise
                texture features. Defaults to 10.0.
            **kwargs: Catches unused keyword arguments for forward compatibility.

        Returns:
            RoundLesion: The instance of the class (`self`), allowing for method chaining.
        """

        target_voxel_count = volume_ml / self.voxel_volume_ml

        # 1. Find valid center point
        base_radius_vox = np.cbrt(target_voxel_count * 0.75 / np.pi)
        valid_indices = np.argwhere(distance_transform_edt(self.boundary_mask) > base_radius_vox * 0.4)
        if len(valid_indices) == 0:
            raise RuntimeError(f'Requested volume {volume_ml}mL is too large for the available space.')
        center_coords = tuple(valid_indices[self.rng.integers(len(valid_indices))])

        # 2. Generate and combine deformed implicit surfaces
        all_surfaces = []
        complexity = int(complexity)
        radius_scaler = np.cbrt(1 / complexity)
        for i in range(complexity):
            sub_seed = self.seed + i
            axes_ratios = get_semi_major_axes_ratios(eccentricity, sub_seed)
            radii = base_radius_vox * axes_ratios * radius_scaler

            coords = np.indices(self.boundary_mask.shape)
            shift = self.rng.uniform(-0.2, 0.2, 3) * radii if complexity > 1 else np.zeros(3)
            current_center = np.array(center_coords) + shift

            ellipsoid_sdf = np.sum(((coords - current_center[:, None, None, None]) / radii[:, None, None, None])**2, axis=0) - 1.0

            noise_scale = 15 * smoothness
            perlin_noise = self._get_noise_texture(self.boundary_mask.shape,
                                                   base_intensity=1,
                                                   std_dev_pct=1,
                                                   scale=noise_scale)
            deformed_surface = ellipsoid_sdf - irregularity * perlin_noise
            all_surfaces.append(deformed_surface)

        final_surface = np.min(all_surfaces, axis=0)

        # 3. Binary search for threshold to match volume
        low, high = -2.0, 2.0
        for _ in range(10):
            mid = (low + high) / 2
            if np.sum(final_surface < mid) < target_voxel_count: low = mid
            else: high = mid

        lesion_core_mask = final_surface < high

        # 4. Generate edema and final textured image
        self.mask = lesion_core_mask.copy()
        self.image = np.zeros_like(self.boundary_mask, dtype=np.float32)

        if (edema_thickness > 0) & (edema_hu > 0):
            # Create a smooth falloff for the edema
            dist_transform = distance_transform_edt(lesion_core_mask)
            edema_mask = (dist_transform > 0) & (dist_transform < edema_thickness)
            edema_intensity = np.exp(-0.125 * (dist_transform)) * edema_hu
            self.image[edema_mask] = edema_intensity[edema_mask]
            self.mask |= edema_mask
            lesion_core_mask &= ~edema_mask  # Core is only the high-intensity part

        texture = self._get_noise_texture(self.mask.shape, intensity_hu, texture_contrast, scale=texture_scale)
        self.image[lesion_core_mask] = texture[lesion_core_mask]  # Apply texture only to the core

        self.volume_ml = np.sum(self.mask) * self.voxel_volume_ml
        self.coords_voxel = tuple(map(int, center_of_mass(self.mask)))

        return self

# =============================================================================
# 4. LESION FACTORY
# A simple factory to create the correct lesion object based on type.
# =============================================================================


class LesionFactory:
    """Creates lesion objects based on the specified type."""

    _lesion_map: Dict[str, Type[Lesion]] = {
        'EDH': DuralLesion,
        'SDH': DuralLesion,
        'IPH': RoundLesion,
    }

    @staticmethod
    def create(lesion_type: str, **kwargs) -> Lesion:
        """
        Factory method to instantiate a lesion object.

        Args:
            lesion_type (str): The type of lesion to create ('EDH', 'SDH', 'IPH').
            **kwargs: Arguments specific to the lesion type constructor.
                      For RoundLesion, DuralLesion, expects 'boundary'.

        Returns:
            An instance of a Lesion subclass.
        """
        lesion_class = LesionFactory._lesion_map.get(lesion_type)
        if not lesion_class:
            raise ValueError(f"Invalid lesion type: {lesion_type}. "
                             f"Valid types are {list(LesionFactory._lesion_map.keys())}")

        # Pass the lesion_type to the constructor along with other kwargs
        return lesion_class(lesion_type=lesion_type, **kwargs)
