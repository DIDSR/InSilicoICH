"""
Module responsible for lesion definition
"""
import math
from dataclasses import dataclass
from typing import Tuple, Union, Dict, Any

from scipy.interpolate import interp1d
import numpy as np
import skimage as ski
from scipy.ndimage import gaussian_filter
from monai.transforms import RandAffine
import noise

# --- Type Aliases for Clarity ---
Shape3D = Tuple[int, int, int]
Point3D = Tuple[int, int, int]


@dataclass
class Lesion:
    """A data class to hold all information about a single lesion."""
    lesion_type: str
    mask: np.ndarray
    coords_voxel: Point3D
    intensity_hu: float
    volume_ml: float
    mass_effect: float
    seed: int


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
