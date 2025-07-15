"""
Module responsible for lesion definition
"""
# --- Standard Library Imports ---
import abc
import math
from typing import Tuple, Union, Dict, Any

# -- Third Party Imports ---
from scipy.interpolate import interp1d
import numpy as np
import skimage as ski
from scipy.ndimage import (gaussian_filter,
                           center_of_mass,
                           distance_transform_edt)
from monai.transforms import RandAffine
import noise
from skimage.graph import route_through_array
from skimage.morphology import binary_erosion

# --- Type Aliases for Clarity ---
Shape3D = Tuple[int, int, int]
Point3D = Tuple[int, int, int]


def sphere_radius_from_volume(volume_ml: float) -> float:
    """Converts volume in milliliters (mL) to a sphere's radius in millimeters (mm)."""
    # 1 mL = 1000 mm^3. Volume of sphere V = 4/3 * pi * r^3
    return np.cbrt(0.75 * volume_ml * 1000 / np.pi)


def calculate_eccentricity(a: float, b: float) -> float:
    """Calculates the eccentricity of an ellipse with semi-axes a and b."""
    # Ensure a is the major axis for the formula.
    major, minor = (a, b) if a > b else (b, a)
    if major == 0:
        return 0.0
    return np.sqrt(1 - (minor**2 / major**2))


def get_closest_key(target_key: float, dictionary: Dict[float, Any]) -> float:
    """Finds the key in a dictionary closest to the target key."""
    if not dictionary:
        raise ValueError("Cannot find closest key in an empty dictionary.")
    return min(dictionary.keys(), key=lambda k: abs(k - target_key))


def get_semi_major_axes_ratios(eccentricity: float, seed: int = None) -> np.ndarray:
    """
    Generates three semi-major axis ratios for an ellipsoid with a target mean eccentricity.

    Note: This is a simplified approach. A more robust method would involve
    optimization to find axes that precisely match the target eccentricity.
    """
    # Pre-calculated lookup table for common eccentricities.
    # Maps mean eccentricity to a plausible set of axis ratios [a, b, c].
    eccentricity_map = {
        0.2: [1.0, 1.0, 0.9], 0.4: [1.0, 0.9, 0.8], 0.6: [1.0, 0.8, 0.6],
        0.8: [1.0, 0.7, 0.4], 0.9: [1.0, 0.6, 0.2], 1.0: [1.0, 0.5, 0.1]
    }
    closest_eccentricity = get_closest_key(eccentricity, eccentricity_map)
    ratios = eccentricity_map[closest_eccentricity]

    rng = np.random.default_rng(seed)
    rng.shuffle(ratios)
    return np.array(ratios)


def get_perimeter(lesion):
    return ski.morphology.binary_dilation(lesion, np.ones((3, 3))) ^\
           ski.morphology.binary_erosion(lesion, np.ones((3, 3)))


def elliptical_lesion(shape: tuple | list,
                      center: tuple | None = None,
                      radius: tuple | None = None,
                      random_rotate: bool | int = True):
    """Generates a binary elliptical mask.

    This function creates a binary elliptical mask based on the specified shape,
    center, and radii. The sphere is defined by the equation:
    $r^2 = z^2 + x^2 + y^2$.

    Args:
        shape (Sequence[int]): The shape of the output array.
        center (Sequence[int]): The coordinates of the center of the ellipse.
        radius (Sequence[int]): A sequence of 3 integers specifying the three
            semi-major axes.
        random_rotate (Union[bool, int]): If True, applies a random rotation.
            If an integer, it is used as the random seed for the transform
            to ensure repeatability.

    Returns:
        np.ndarray: A binary array representing the elliptical mask.

    """
    if isinstance(radius, np.ndarray):
        radius = list(radius)
    center = center or [dim//2 for dim in shape]
    radius = radius or [dim//10 for dim in shape]
    if not isinstance(radius, list | tuple):
        radius = 3*[radius]
    ell = ski.draw.ellipsoid(*radius)
    if random_rotate:
        transform = RandAffine(prob=1, rotate_range=[np.pi/2, np.pi/2, np.pi/2],
                               scale_range=[0.1, 0.1, 0.1], padding_mode="zeros")
        if isinstance(random_rotate, int):
            transform.set_random_state(seed=random_rotate)

        ell = np.pad(ell, ((int(max(radius)-radius[0]),),
                           (int(max(radius)-radius[1]),),
                           (int(max(radius)-radius[2]),)))
        ell = transform(ell)

    starts = center - np.array(ell.shape)//2
    ends = center + np.array(ell.shape)//2 + 1
    lesion_only = np.zeros(shape)
    lesion_only[starts[0]:ends[0],
                starts[1]:ends[1],
                starts[2]:ends[2]] = ell
    return np.where(lesion_only > 0, True, False)


def connect_points(
    start: Union[Tuple[int, int], np.ndarray],
    end: Union[Tuple[int, int], np.ndarray],
    boundary: np.ndarray,
    hematoma_type: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates two aligned coordinate paths connecting start and end points.

    One path follows a high-cost boundary in an image, and the second is a
    Bézier curve running between the points. Both output paths are resampled
    to have the same number of points and are de-duplicated. The paths form
    a closed loop suitable for filling.

    Args:
        start: A tuple or NumPy array (row, col) for the starting coordinate.
        end: A tuple or NumPy array (row, col) for the ending coordinate.
        boundary: A 2D NumPy array where the desired path has a low cost
                  (e.g., 1s on a background of 0s).
        hematoma_type: A string ('EDH', 'SDH', or other) that determines the
                       shape of the Bézier curve.

    Returns:
        A tuple containing:
        - filled_array: A 2D NumPy array with the area between the two
                        paths filled in.
        - boundary_coords: An (N, 2) array of unique coordinates for the path
                           along the boundary.
        - connect_coords: An (N, 2) array of unique coordinates for the Bézier
                          curve, with length N matching boundary_coords.
    """
    # --- 1. Find the path along the boundary (more efficiently) ---
    costs = np.where(boundary == 1, 0, 10000)
    path, _ = ski.graph.route_through_array(costs, start=tuple(start),
                                            end=tuple(end),
                                            fully_connected=False)

    boundary_coords = np.array(path)
    if len(boundary_coords) == 0:
        raise ValueError(
            "Could not find a path between start and end points on boundary.")

    # --- 2. Define Bézier Curve Parameters in a structured way ---
    rows, cols = boundary.shape
    bezier_params = {
        'EDH': {'weight': 0.14, 'middle': (rows / 2, cols / 2)},
        'SDH': {'weight': 2.0, 'middle': boundary_coords[len(boundary_coords) // 2]},
    }
    params = bezier_params.get(hematoma_type, {'weight': 0.0, 'middle': (rows / 2, cols / 2)})
    bezier_weight = params['weight']
    bezier_middle = params['middle']

    # --- 3. Generate the Bézier Curve with a more robust retry mechanism ---
    for i, weight in enumerate([bezier_weight, 0.0]):
        rr, cc = ski.draw.bezier_curve(
            r0=start[0], c0=start[1],
            r1=int(bezier_middle[0]), c1=int(bezier_middle[1]),
            r2=end[0], c2=end[1],
            weight=weight
        )

        generated_points = set(zip(rr, cc))
        if tuple(start) in generated_points and tuple(end) in generated_points:
            connect_coords_raw = np.stack((rr, cc), axis=-1)
            break

        if i == 0 and bezier_weight == 0.0:
            break
    else:
        raise RuntimeError(
            f"Unable to create a valid Bézier curve between {start} and {end}."
            )

    # --- 4. Reorder and Resample the Bézier Curve (Vectorized) ---
    distances = np.linalg.norm(connect_coords_raw - np.array(start), axis=1)
    ordered_connect_coords = connect_coords_raw[np.argsort(distances)]

    num_points_boundary = len(boundary_coords)
    if len(ordered_connect_coords) > 1:
        t_original = np.linspace(0, 1, len(ordered_connect_coords))

        f_row = interp1d(t_original, ordered_connect_coords[:, 0], kind='linear')
        f_col = interp1d(t_original, ordered_connect_coords[:, 1], kind='linear')

        t_new = np.linspace(0, 1, num_points_boundary)

        row_new = f_row(t_new)
        col_new = f_col(t_new)

        connect_coords = np.vstack((row_new, col_new)).T
        connect_coords[0] = start
        connect_coords[-1] = end
        connect_coords = connect_coords.astype(int)
    else:
        connect_coords = np.repeat(ordered_connect_coords, num_points_boundary, axis=0)

    # --- 5. Remove duplicate coordinates from both paths to ensure uniqueness ---
    # A coordinate is a duplicate if it's identical to a preceding coordinate.
    # We find duplicates in each path and remove the corresponding indices from 
    # both paths to maintain alignment for transformations like ThinPlateSpline.

    def get_duplicate_mask(arr):
        # Using a structured array to treat rows as single entities for finding uniqueness
        # This is more robust and often faster than lexsort for this purpose.
        _, unique_indices = np.unique(arr, axis=0, return_index=True)
        duplicate_mask = np.ones(len(arr), dtype=bool)
        duplicate_mask[np.sort(unique_indices)] = False
        return duplicate_mask

    duplicate_mask_b = get_duplicate_mask(boundary_coords)
    duplicate_mask_c = get_duplicate_mask(connect_coords)

    # Combine the masks: an index is removed if it's a duplicate in *either* path
    combined_duplicate_mask = duplicate_mask_b | duplicate_mask_c

    # Keep only the non-duplicate indices by inverting the mask
    boundary_coords = boundary_coords[~combined_duplicate_mask]
    connect_coords = connect_coords[~combined_duplicate_mask]
    connect_coords[0] = boundary_coords[0]  # Ensure the first point matches the boundary start
    connect_coords[-1] = boundary_coords[-1]  # Ensure the last point matches the boundary end
    # --- 6. Create Final Filled Array from a Guaranteed Closed Loop ---
    # This step is performed *after* de-duplication to ensure the final paths are used.
    # To ensure a perfectly closed loop for filling, we combine the two paths.
    # We take the boundary path, and append the connecting path in reverse order
    # (so it goes from 'end' back to 'start'), removing the shared endpoints
    # from the reversed path to avoid overlap.
    if len(boundary_coords) > 0 and len(connect_coords) > 0:
        closed_loop_coords = np.concatenate(
            (boundary_coords, connect_coords[::-1][1:-1])
        )

        # Create the perimeter on an empty array
        perimeter_array = np.zeros_like(boundary, dtype=float)
        rows, cols = closed_loop_coords[:, 0], closed_loop_coords[:, 1]
        perimeter_array[rows, cols] = 1.0
        # perimeter_array = ski.morphology.closing(perimeter_array, np.ones((3, 3)))
        # Fill the area enclosed by the perimeter
        filled_array = ski.morphology.convex_hull_image(perimeter_array)
    else:
        # If de-duplication resulted in an empty path, return an empty filled_array
        filled_array = np.zeros_like(boundary, dtype=int)

    return filled_array, boundary_coords, connect_coords


def coverage_from_volume(volume, hematoma_type, slice_thickness):
    '''
    The hemorrhage masks from the BHSD dataset were used to define a logarithmic
    relationship between hemorrhage volume and slice coverage (in z).
    '''
    if hematoma_type == 'EDH':
        z_coverage = 10.231*math.log(volume) + 19.094
    elif hematoma_type == 'SDH':
        z_coverage = 10.380*math.log(volume) + 24.480
    elif hematoma_type == 'IPH':
        z_coverage = 6.925*math.log(volume) + 17.315
    # currently unused
    elif hematoma_type == 'SAH':
        z_coverage = 5.383*math.log(volume) + 18.237
    elif hematoma_type == 'IVH':
        z_coverage = 6.657*math.log(volume) + 20.492
    # convert units from mm to number of slices
    slice_coverage = z_coverage / slice_thickness

    # round to nearest odd number
    slice_coverage = math.ceil(slice_coverage)
    if slice_coverage % 2 == 0:
        slice_coverage = slice_coverage - 1

    return slice_coverage


# --- Method 1: 3D Perlin Noise ---
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


# --- Method 2: 3D Simplex Noise ---
def generate_3d_simplex_texture(depth=32, height=128, width=128, scale=50.0,
                                octaves=4, persistence=0.5,
                                lacunarity=2.0) -> np.ndarray:
    """
    Generates a texture using a single layer of Simplex noise.

    Args:
        depth (int): The depth of the texture.
        height (int): The height of the texture.
        width (int): The width of the texture.
        scale (float): The 'zoom' level of the noise.
        octaves (int): The number of noise layers to combine.
        persistence (float): The amplitude multiplier for each subsequent octave.
        lacunarity (float): The frequency multiplier for each subsequent octave.
    Returns:
        np.ndarray: A 3D numpy array containing the generated texture.
    """
    texture_array = np.zeros((depth, height, width))
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                texture_array[z][y][x] = noise.snoise3(x / scale, 
                                                       y / scale,
                                                       z / scale,
                                                       octaves=octaves,
                                                       persistence=persistence,
                                                       lacunarity=lacunarity)
    return texture_array


# --- Method 3: 3D Fractional Brownian Motion (fBm) / Turbulence ---
def generate_3d_fbm_texture(depth=32, height=128, width=128, scale=75.0,
                            octaves=6, persistence=0.5, lacunarity=2.0,
                            seed=None) -> np.ndarray:
    """
    Generates a more complex, mottled texture using Fractional Brownian Motion (fBm).
    This is essentially layering multiple octaves of noise.

    Args:
        depth (int): The depth of the texture.
        height (int): The height of the texture.
        width (int): The width of the texture.
        scale (float): The base 'zoom' level of the noise.
        octaves (int): The number of noise layers.
        persistence (float): How much each octave contributes to the overall shape.
        lacunarity (float): How much detail is added with each octave.
        seed (int, optional): Seed for reproducibility. If None, a random seed is used.
    Returns:
        np.ndarray: A 3D numpy array containing the generated texture.
    If `seed` is provided, it will be used to ensure reproducibility.
    If `seed` is None, a random seed will be used.
    """
    # This is identical in implementation to the multi-octave Perlin noise function,
    # as pnoise2 with octaves > 1 is an implementation of fBm.
    # We use different parameters here to highlight its use for mottled textures.
    return generate_3d_perlin_texture(depth, height, width, scale, octaves, persistence, lacunarity, seed=seed)


# --- Method 4: 3D Filtered Random Noise ---
def generate_3d_filtered_noise_texture(depth=32, height=128, width=128, sigma=10.0):
    """
    Generates a texture by applying a Gaussian low-pass filter to random noise.

    Args:
        width (int): The width of the texture.
        height (int): The height of the texture.
        sigma (float): The standard deviation for the Gaussian kernel. 
                       Larger values create smoother, lower-frequency textures.
    """
    # Create a 3D array of random values
    random_noise = np.random.rand(depth, height, width)
    # Apply the 3D Gaussian filter
    filtered_noise = gaussian_filter(random_noise, sigma=sigma)
    return filtered_noise


class Lesion(abc.ABC):
    """Abstract base class for lesions

    Lesions are responsible for generating their own binary masks and metadata.
    """

    def __init__(self, lesion_type: str, spacings: tuple[float] = (1., 1., 1.),
                 texture_scale: float = 3., texture_contrast: float = 0, seed: int = None):
        self.lesion_type = lesion_type
        self.dz, self.dx, self.dy = spacings
        self.texture_scale = texture_scale
        self.texture_contrast = texture_contrast
        self.seed = seed if seed is not None else np.random.randint(0, 2**31 - 1)
        self.mask = None
        self.coords_voxel = None
        self.mass_effect = 0.0

    @abc.abstractmethod
    def generate(self):
        """generates the lesion mask and stores it in self.mask, also sets self.coords_voxel"""
        pass

    def get_noise_texture(self, contrast: float = 80.0, **texture_kwargs) -> np.ndarray:
        """
        Generates a 3D noise texture.

        Args:
            contrast: The mean intensity (HU) of the noise texture.
            **texture_kwargs: Keyword arguments for generate_... functions,
                                e.g., noise_type, scale, contrast_std, seed.
        """
        # Set defaults if not provided
        defaults = {'noise_type': 'perlin', 'scale': 15, 'contrast_std': 1.0, 'seed': None}
        config = {**defaults, **texture_kwargs}

        if config['contrast_std'] <= 0:
            return np.full(self.mask.shape, contrast, dtype=np.float32)

        # Use a dictionary dispatch to map noise_type to function
        noise_generators = {
            'perlin': generate_3d_perlin_texture,
            'simplex': generate_3d_simplex_texture,
            'fbm': generate_3d_fbm_texture,
            'filtered_noise': generate_3d_filtered_noise_texture,
        }

        generator = noise_generators.get(config['noise_type'])
        if not generator:
            raise ValueError(f"Unknown noise type: {config['noise_type']}")

        noise_texture = generator(
            *self.mask.shape, scale=config['scale'], seed=config['seed']
        )
        return noise_texture * config['contrast_std'] * contrast + contrast


class DuralLesion(Lesion):
    def __init__(self, dura_map, lesion_type: str, spacings: tuple[float] = (1., 1., 1.),
                 volume_ml: float = 2., intensity_hu: float = 40, seed: int = None, **kwargs):
        """
        could store volume, atten, etc in this object, currently like the Lesion class

        Core logic to generate a tapered dural lesion mask and the associated warped brain.
        Uses an iterative approach to match the desired volume.

        Args:
            dura_map (np.ndarray): 3D binary array indicating the dura locations, 
                this the boundary where the lesion will start from.
            spacings (tuple): voxel spacings in mm (dz, dx, dy)
            lesion_type (str): 'SDH' or 'EDH'
            volume_ml (float): desired lesion volume in mL
            intensity_hu (float): desired lesion intensity in HU
            seed (int): random seed for reproducibility
        """
        super().__init__(lesion_type=lesion_type, spacings=spacings, seed=seed, **kwargs)
        if lesion_type not in ['SDH', 'EDH']:
            raise ValueError(f"DuralLesion only supports 'SDH' or 'EDH', got {lesion_type}")
        self.dura_map = dura_map
        self.volume_ml = volume_ml
        self.intensity_hu = intensity_hu

    def generate(self):
        rng = np.random.default_rng(self.seed)
        dura_map = self.dura_map
        voxel_volume_ml = (self.dx * self.dy * self.dz) / 1000.0
        hematoma_type = self.lesion_type
        desired_volume_ml = self.volume_ml
        num_slices = coverage_from_volume(
            volume=self.volume_ml, hematoma_type=self.lesion_type, slice_thickness=self.dz
        )
        if num_slices == 0:
            return np.zeros_like(dura_map, dtype=bool)

        # Initial estimate for the central slice area
        avg_area_mm2 = (self.volume_ml * 1000) / (num_slices * self.dz)
        center_slice_area_mm2 = 1.5 * avg_area_mm2

        final_lesion_mask = np.zeros_like(dura_map, dtype=bool)

        # Iteratively correct the volume
        for i in range(3):  # Try up to 3 times to correct the volume
            # --- Generate the lesion mask based on the current area estimate ---
            hemisphere_mask = np.zeros_like(dura_map, dtype=bool)
            mid_y = self.dura_map.shape[2] // 2
            if rng.choice(['left', 'right']) == 'left':
                hemisphere_mask[:, :, :mid_y] = True
            else:
                hemisphere_mask[:, :, mid_y:] = True

            # Use a copy for modification in the loop
            current_dura_map = dura_map.copy()
            current_dura_map[~hemisphere_mask] = 0

            start_point, end_point, init_slice_idx = self._find_initial_dural_endpoints(
                current_dura_map, center_slice_area_mm2, hematoma_type, rng
            )

            all_filled_arrays = {
                init_slice_idx: connect_points(
                    start=start_point, end=end_point, boundary=current_dura_map[init_slice_idx],
                    hematoma_type=hematoma_type
                )[0]
            }

            half_slices = num_slices / 2.0

            # Propagate downwards
            prev_start, prev_end = start_point, end_point
            for j in range(1, int(np.ceil(half_slices))):
                slice_idx = init_slice_idx + j
                if slice_idx >= self.dura_map.shape[0]: break
                scale = max(0, 1 - (j / half_slices)**2)
                filled, prev_start, prev_end = self._propagate_and_taper_slice(
                    current_dura_map[slice_idx], prev_start, prev_end, hematoma_type, scale
                )
                if filled is None: break
                all_filled_arrays[slice_idx] = filled

            # Propagate upwards
            prev_start, prev_end = start_point, end_point
            for j in range(1, int(np.ceil(half_slices))):
                slice_idx = init_slice_idx - j
                if slice_idx < 0: break
                scale = max(0, 1 - (j / half_slices)**2)
                filled, prev_start, prev_end = self._propagate_and_taper_slice(
                    current_dura_map[slice_idx], prev_start, prev_end, hematoma_type, scale
                )
                if filled is None: break
                all_filled_arrays[slice_idx] = filled

            current_mask = np.zeros_like(dura_map, dtype=bool)
            for idx, filled_array in all_filled_arrays.items():
                current_mask[idx] = filled_array

            # --- Measure and correct ---
            current_volume_ml = np.sum(current_mask) * voxel_volume_ml

            # If volume is within 5% of target, we're done
            if abs(current_volume_ml - desired_volume_ml) / desired_volume_ml < 0.05:
                final_lesion_mask = current_mask
                break

            # If not, calculate correction factor and adjust for the next iteration
            if current_volume_ml > 0:
                correction_factor = desired_volume_ml / current_volume_ml
                center_slice_area_mm2 *= correction_factor

            final_lesion_mask = current_mask  # Keep the last result in case loop finishes

        self.mask = final_lesion_mask

        image = np.where(final_lesion_mask, self.intensity_hu, 0).astype(np.float32)
        lesion_texture = self.get_noise_texture(
            contrast=self.intensity_hu, seed=self.seed,
            contrast_std=self.texture_contrast, scale=self.texture_scale
        )
        image[final_lesion_mask] = lesion_texture[final_lesion_mask]

        self.image = image
        self.coords_voxel = tuple(map(int, center_of_mass(self.mask)))
        self.volume_ml = current_volume_ml  # actual achieved volume

        return self

    def _find_initial_dural_endpoints(self, dura_map, center_slice_area_mm2, h_type, rng):
        """Finds a valid starting slice and two endpoints for the dural lesion."""
        ratio = 4 if h_type == 'EDH' else 11
        desired_dist_mm = np.sqrt(ratio * center_slice_area_mm2)
        desired_dist_vox = desired_dist_mm / self.dx

        valid_slice_indices = np.where(dura_map.sum(axis=(1, 2)) > desired_dist_vox)[0]
        if len(valid_slice_indices) == 0:
            raise RuntimeError("Could not find any suitable slices for dural lesion.")

        for _ in range(50):
            init_slice_idx = rng.choice(valid_slice_indices)
            dura_points = np.argwhere(dura_map[init_slice_idx])
            if len(dura_points) < 2: continue

            start_point = dura_points[rng.integers(len(dura_points))]
            distances = np.linalg.norm(dura_points - start_point, axis=1)

            valid_end_points_mask = np.isclose(distances, desired_dist_vox, atol=max(5.0, desired_dist_vox * 0.1))
            valid_end_points = dura_points[valid_end_points_mask]

            if len(valid_end_points) > 0:
                end_point = valid_end_points[rng.integers(len(valid_end_points))]
                return start_point, end_point, init_slice_idx

        raise RuntimeError(f'Failed to find suitable start/end points for the requested lesion size.')

    def _propagate_and_taper_slice(self, dura_slice_map, prev_start, prev_end, h_type, scale_factor):
        """
        Propagates a lesion to an adjacent slice, scaling its cross-section.
        """
        dura_points = np.argwhere(dura_slice_map)
        if len(dura_points) < 2:
            return None, None, None

        dist_to_prev_start = np.linalg.norm(dura_points - prev_start, axis=1)
        dist_to_prev_end = np.linalg.norm(dura_points - prev_end, axis=1)

        initial_new_start = dura_points[np.argmin(dist_to_prev_start)]
        initial_new_end = dura_points[np.argmin(dist_to_prev_end)]

        if np.array_equal(initial_new_start, initial_new_end):
            return None, None, None

        # Find the path along the dura between the new points
        costs = np.where(dura_slice_map, 0, 10000)
        try:
            path_indices, _ = route_through_array(
                costs, start=tuple(initial_new_start),
                end=tuple(initial_new_end)
            )
            path = np.array(path_indices)
        except (ValueError, IndexError):
            # No path found between points, cannot taper.
            return None, None, None

        if len(path) < 2:
            return None, None, None

        # Taper the path by shortening it from both ends
        original_length = len(path)
        new_length = int(original_length * scale_factor)
        if new_length < 2:
            return None, None, None

        trim_amount = (original_length - new_length) // 2

        final_start = path[trim_amount]
        final_end = path[original_length - 1 - trim_amount]

        filled_array, _, _ = connect_points(
            start=final_start, end=final_end,
            boundary=dura_slice_map, hematoma_type=h_type
        )
        return filled_array, final_start, final_end


class RoundLesion(Lesion):
    def __init__(self, boundary_mask: np.ndarray, spacings: tuple[float] = (1., 1., 1.), volume_ml: float = 5.0, intensity_hu: float = 50.0, seed: int = None,
                 eccentricity: float = 0.5, irregularity: float = 0.5, smoothness: float = 1.0,
                 complexity: int = 1, edema: Union[bool, int] = False, **kwargs):
        super().__init__(lesion_type='IPH', spacings=spacings, seed=seed, **kwargs)
        self.boundary_mask = boundary_mask
        self.volume_ml = volume_ml
        self.intensity_hu = intensity_hu
        self.eccentricity = eccentricity
        self.irregularity = irregularity
        self.smoothness = smoothness
        self.complexity = complexity
        self.edema = edema
        self.overlap = 0.4
        """
        Generates an irregular, blob-like lesion using one or more deformed implicit surfaces.

        Args:
            boundary_mask: A binary mask indicating the valid region for lesion placement.
                           The lesion center will be chosen within this mask.
            volume_ml: Desired lesion volume in mL.
            intensity_hu: Desired lesion intensity in HU.
            eccentricity: Controls the elongation of the base ellipsoid. 0 is a sphere,
                          0.9 is a very elongated ellipsoid. Should be in [0, 0.9).
                          If eccentricity is 0, the lesion will be spherical before deformation.
            irregularity: Magnitude of surface deformation. 0 is a perfect
                          ellipsoid, higher values are more irregular.
            smoothness: Scale of the surface features. Higher values lead to
                        smoother, more blob-like features.
            complexity: The number of overlapping, irregular ellipsoids to
                        combine for the final shape.
            overlap: The allowed proportion of overlap of the lesion outside
                     the material of interest
        """
    def generate(self) -> np.ndarray:
        seed = self.seed
        volume = self.volume_ml
        voxel_volume_mm3 = self.dx * self.dy * self.dz
        target_voxel_count = self.volume_ml * 1000 / voxel_volume_mm3
        rng = np.random.default_rng(seed)

        # --- 1. Find a valid center point for the lesion ---
        base_radius_vox = np.cbrt(target_voxel_count * 0.75 / np.pi)
        material_mask = self.boundary_mask
        if material_mask.sum() < target_voxel_count:
            raise RuntimeError(f'Requested volume {self.volume_ml} mL is too large for the available space.')
        valid_points = distance_transform_edt(material_mask) > (base_radius_vox * self.overlap)
        valid_indices = np.argwhere(valid_points)
        if len(valid_indices) == 0:
            raise RuntimeError(f'Requested volume {volume} mL is too large for the available space.')

        center_coords = tuple(valid_indices[rng.integers(len(valid_indices))])

        # --- 2. Generate one or more deformed implicit surfaces ---
        all_implicit_surfaces = []
        # Scale the radius for each component to keep the total volume appropriate
        radius_scaler = np.cbrt(1 / self.complexity)

        for i in range(self.complexity):
            sub_seed = seed + i if seed is not None else None

            # Create a base implicit ellipsoid
            axes_ratios = get_semi_major_axes_ratios(self.eccentricity, sub_seed)
            radii = base_radius_vox * axes_ratios * radius_scaler
            radii[radii == 0] = 1e-6

            coords = np.indices(material_mask.shape, dtype=float)

            # Slightly shift center for each component if complexity > 1
            if self.complexity > 1:
                shift = rng.uniform(-0.2, 0.2, 3) * radii
                current_center = np.array(center_coords) + shift
            else:
                current_center = np.array(center_coords)

            centered_coords = coords - current_center[:, np.newaxis, np.newaxis, np.newaxis]
            ellipsoid_sdf = np.sum((centered_coords / radii[:, np.newaxis, np.newaxis, np.newaxis])**2, axis=0) - 1.0

            # Deform the surface with Perlin noise
            noise_scale = 15 * self.smoothness
            perlin_noise = generate_3d_perlin_texture(*material_mask.shape, scale=noise_scale, seed=sub_seed)
            deformed_surface = ellipsoid_sdf - self.irregularity * perlin_noise
            all_implicit_surfaces.append(deformed_surface)

        # --- 3. Combine surfaces and find the final mask via binary search ---
        # Take the minimum value at each point to create a smooth union of the shapes
        final_implicit_surface = np.min(all_implicit_surfaces, axis=0)

        low_thresh, high_thresh = -2.0, 2.0
        for _ in range(10):  # Binary search for the correct volume
            mid_thresh = (low_thresh + high_thresh) / 2
            current_mask = final_implicit_surface < mid_thresh
            if np.sum(current_mask) < target_voxel_count:
                low_thresh = mid_thresh
            else:
                high_thresh = mid_thresh

        final_mask = final_implicit_surface < high_thresh

        current_volume_ml = np.sum(current_mask) * voxel_volume_mm3 / 1000.0

        # --- 4. edema ---
        lesion_vol = np.zeros_like(material_mask, dtype=np.float32)

        if self.edema:
            edema_pixels = 5 if self.edema is True else int(self.edema)
            edema_mask = binary_erosion(final_mask, np.ones((edema_pixels,)*3)) ^ final_mask
            lesion_vol[edema_mask] = 10
            final_mask |= edema_mask
        lesion_vol[final_mask] = self.intensity_hu

        image = np.where(final_mask, lesion_vol, 0).astype(np.float32)
        self.mask = final_mask
        # --- 5. Fill in texture ---
        lesion_texture = self.get_noise_texture(
            contrast=self.intensity_hu, seed=self.seed,
            contrast_std=self.texture_contrast, scale=self.texture_scale
        )
        image[final_mask] = lesion_texture[final_mask]

        self.image = image
        self.coords_voxel = tuple(map(int, center_of_mass(self.mask)))
        self.volume_ml = current_volume_ml  # actual achieved volume
        return self
