"""
Refactored module for creating and manipulating medical imaging phantoms with
lesions.
"""

from typing import List, Tuple, Dict, Any, Union

import numpy as np
from monai.transforms import Resize, RandAffine, Affine
from scipy.ndimage import (center_of_mass,
                           distance_transform_edt,
                           map_coordinates)
from skimage.morphology import binary_erosion
from skimage.graph import route_through_array

from VITools import Phantom

# Assuming these are local modules with the specified functions.
# For this example, placeholder functions are used where necessary.
from .. import lesion_definition as ld

# --- Type Aliases for Clarity ---
Shape3D = Tuple[int, int, int]
Spacing3D = Tuple[float, float, float]


def resize(array: np.ndarray, shape: Shape3D, **kwargs) -> np.ndarray:
    """Internal helper to resize a 3D array using MONAI."""
    resize_transform = Resize(max(shape), size_mode='longest', **kwargs)
    # MONAI transforms expect a channel-first format (C, H, W, D)
    resized_array = resize_transform(array[None])[0]
    return resized_array


def get_mean_age(age_range: str):
    return (float(age_range.split('-')[1])+float(age_range.split('-')[0]))/2


class LesionPhantom(Phantom):
    """
    A Phantom object with methods for inserting realistic lesions.
    """
    lesion_types = {'IPH': {
        'volume': [0, 100],
        'intensity': [-100, 100],
        'mass_effect': [True, False],
        'seed': None
        },
                    'SDH': {
        'volume': [0, 100],
        'intensity': [-100, 100],
        'mass_effect': [True, False],
        'seed': None
        },
                    'EDH': {
        'volume': [0, 100],
        'intensity': [-100, 100],
        'mass_effect': [True, False],
        'seed': None
        }
                    }

    def __init__(self, img: np.ndarray, spacings: Spacing3D = (1.0, 1.0, 1.0), **kwargs):
        super().__init__(img, spacings, **kwargs)
        self.lesions: List[ld.Lesion] = []

    @property
    def warp_exclusion_mask(self) -> np.ndarray:
        """Defines a mask of pixels that should not be warped (e.g., skull)."""
        # This could be customized in a subclass, e.g., by segmenting the skull.
        return self.get_skull_map() > 0

    @property
    def warp_inclusion_mask(self) -> np.ndarray:
        """Defines a mask of pixels that are allowed to be warped (e.g., brain tissue)."""
        # Typically, this is the brain mask, excluding skull and air.
        return self.get_brain_mask() > 0

    def resize(self, shape: Shape3D, **kwargs) -> None:
        """Resizes the phantom array and adjusts spacings."""
        # Note: Resizing will invalidate existing lesion masks.
        if self.lesions:
            print("Warning: Resizing phantom after lesion insertion. Lesion masks are now invalid.")
            self.lesions = []
        super().resize(shape, **kwargs)

    def get_lesion_volume(self) -> float:
        """Calculates the total volume of all lesions in mL."""
        return sum(lesion.volume_ml for lesion in self.lesions)

    def __repr__(self) -> str:
        """Provides a detailed string representation of the phantom."""
        base_repr = super().__repr__()
        lesion_details = f"Number of lesions: {len(self.lesions)}\n"
        if self.lesions:
            for i, lesion in enumerate(self.lesions):
                lesion_details += (
                    f"  - Lesion {i+1}: {lesion.lesion_type}, "
                    f"Volume: {lesion.volume_ml:.2f} mL, "
                    f"Center: {lesion.coords_voxel}\n"
                    f"Mass effect strength: {lesion.mass_effect}"
                )
        return f"{base_repr}\n{lesion_details}"

    def get_noise_texture(self, contrast: float = 80.0, **texture_kwargs) -> np.ndarray:
        """
        Generates a 3D noise texture.

        Args:
            contrast: The mean intensity (HU) of the noise texture.
            **texture_kwargs: Keyword arguments for ld.generate_... functions,
                              e.g., noise_type, scale, contrast_std, seed.
        """
        # Set defaults if not provided
        defaults = {'noise_type': 'perlin', 'scale': 15, 'contrast_std': 1.0, 'seed': None}
        config = {**defaults, **texture_kwargs}

        if config['contrast_std'] <= 0:
            return np.full(self.shape, contrast, dtype=np.float32)

        # Use a dictionary dispatch to map noise_type to function
        noise_generators = {
            'perlin': ld.generate_3d_perlin_texture,
            'simplex': ld.generate_3d_simplex_texture,
            'fbm': ld.generate_3d_fbm_texture,
            'filtered_noise': ld.generate_3d_filtered_noise_texture,
        }

        generator = noise_generators.get(config['noise_type'])
        if not generator:
            raise ValueError(f"Unknown noise type: {config['noise_type']}")

        noise_texture = generator(
            *self.shape, scale=config['scale'], seed=config['seed']
        )
        return noise_texture * config['contrast_std'] * contrast + contrast

    def insert_lesion(
        self,
        lesion_type: str,
        volume: float = 5.0,
        intensity: float = 50.0,
        mass_effect: Union[bool, float] = False,
        seed: int = None,
        texture_kwargs: Dict[str, Any] = None,
        **lesion_specific_kwargs
    ):
        """
        Primary method to insert a lesion of a specified type into the phantom.

        Args:
            lesion_type: Type of lesion ('IPH', 'EDH', 'SDH').
            volume: Volume of the lesion in mL.
            intensity: Base CT number of the lesion in HU.
            mass_effect: If False/0.0, no mass effect. If True, uses 1.0.
                         A float controls the warp strength.
            seed: Seed for all random operations.
            texture_kwargs: Arguments for noise texture generation.
            **lesion_specific_kwargs: Arguments for specific lesion types.
                For IPH: `eccentricity`, `irregularity`, `smoothness`,
                         `complexity`, `edema`.
        """
        if volume <= 0:
            return self
        if lesion_type not in self.lesion_types:
            raise ValueError(f"Unknown lesion type: {lesion_type}")

        if seed is None:
            seed = np.random.randint(0, 1e6)

        if lesion_type == 'IPH':
            img_w_lesion, lesion_mask, lesion_coords = self._add_round_lesion(
                volume, intensity, seed, **lesion_specific_kwargs
            )
        else:  # EDH or SDH
            img_w_lesion, lesion_mask, lesion_coords = self._add_dural_lesion(
                lesion_type, volume, seed,
            )

        # add mass_effect
        mass_effect_strength = 0
        if mass_effect is True:
            mass_effect_strength = 1.0
        elif isinstance(mass_effect, (float, int)):
            mass_effect_strength = float(mass_effect)

        if mass_effect_strength > 0:
            img_w_lesion = self._apply_mass_effect(lesion_mask,
                                                   mass_effect_strength)
        # correct and erroneaous voxels
        diff = self.get_CT_number_phantom() - img_w_lesion
        img_w_lesion[abs(diff) > intensity] =\
            self.get_CT_number_phantom()[abs(diff) > intensity]

        # add texture
        if texture_kwargs:
            lesion_texture = self.get_noise_texture(
                contrast=intensity, seed=seed, **texture_kwargs
            )
            img_w_lesion[lesion_mask] = lesion_texture[lesion_mask]
        else:
            img_w_lesion[lesion_mask] = intensity

        # Store the new lesion's information
        lesion_info = ld.Lesion(
            lesion_type=lesion_type,
            mask=lesion_mask,
            coords_voxel=lesion_coords,
            intensity_hu=intensity,
            volume_ml=self.dx * self.dy * self.dz * lesion_mask.sum() / 1000.0,
            mass_effect=mass_effect,
            seed=seed
        )
        self.lesions.append(lesion_info)
        self._phantom = img_w_lesion
        return self

    def apply_transform(self, transform: Union[RandAffine, Affine],
                        seed: int = None):
        """
        Applies an affine transformation to the phantom and all its lesion masks.

        This method applies the given transformation to the internal phantom data
        and to each lesion mask, ensuring they remain aligned. The transformation
        modifies the object's state in-place.

        Args:
            transform (RandAffine | Affine): The transformation to apply.
            seed (int, optional): A seed for the random number generator to ensure
                                  reproducibility, especially for RandAffine.
        """
        if seed is not None:
            transform.set_random_state(seed=seed)

        # 1. Transform the main phantom image. If the transform is random, its
        # parameters are now fixed ("realized") for subsequent calls.
        self._phantom = transform(self._phantom[None])[0]

        # 2. Apply the *same* realized transform to each lesion mask
        for lesion in self.lesions:
            # MONAI expects a channel dimension (C, H, W, D) and float type for interpolation
            transformed_mask = transform(lesion.mask[None].astype(np.float32))[0]

            # Binarize the result after interpolation and update the lesion's mask
            lesion.mask = transformed_mask > 0.5

            # 3. Update lesion metadata to reflect the transformation
            if lesion.mask.any():
                lesion.coords_voxel = tuple(map(int, center_of_mass(lesion.mask)))
                lesion.volume_ml = np.sum(lesion.mask) * (self.dx * self.dy * self.dz) / 1000.0
            else:
                # If the lesion is transformed out of the image, handle it gracefully
                lesion.coords_voxel = (-1, -1, -1)
                lesion.volume_ml = 0.0

    def _add_round_lesion(
        self,
        volume: float,
        intensity: float,
        seed: int,
        material: str = 'white matter',
        eccentricity: float = 0.5,
        irregularity: float = 0.5,
        smoothness: float = 1.0,
        complexity: int = 1,
        overlap: float = 0.4,
        edema: Union[bool, int] = False
    ) -> Tuple[np.ndarray, np.ndarray, ld.Point3D]:
        """
        Generates an irregular, blob-like lesion using one or more deformed implicit surfaces.

        Args:
            irregularity: Magnitude of surface deformation. 0 is a perfect
                          ellipsoid, higher values are more irregular.
            smoothness: Scale of the surface features. Higher values lead to
                        smoother, more blob-like features.
            complexity: The number of overlapping, irregular ellipsoids to
                        combine for the final shape.
            overlap: The allowed proportion of overlap of the lesion outside
                     the material of interest
        """
        img = self.get_CT_number_phantom()
        voxel_volume_mm3 = self.dx * self.dy * self.dz
        target_voxel_count = volume * 1000 / voxel_volume_mm3
        rng = np.random.default_rng(seed)

        # --- 1. Find a valid center point for the lesion ---
        base_radius_vox = np.cbrt(target_voxel_count * 0.75 / np.pi)
        material_mask = self.get_material_mask(material)
        valid_points = distance_transform_edt(material_mask) > (base_radius_vox * overlap)
        valid_indices = np.argwhere(valid_points)
        if len(valid_indices) == 0:
            raise RuntimeError(f'Requested volume {volume} mL is too large for the available space.')

        center_coords = tuple(valid_indices[rng.integers(len(valid_indices))])

        # --- 2. Generate one or more deformed implicit surfaces ---
        all_implicit_surfaces = []
        # Scale the radius for each component to keep the total volume appropriate
        radius_scaler = np.cbrt(1 / complexity)

        for i in range(complexity):
            sub_seed = seed + i if seed is not None else None

            # Create a base implicit ellipsoid
            axes_ratios = ld.get_semi_major_axes_ratios(eccentricity, sub_seed)
            radii = base_radius_vox * axes_ratios * radius_scaler
            radii[radii == 0] = 1e-6

            coords = np.indices(self.shape, dtype=float)

            # Slightly shift center for each component if complexity > 1
            if complexity > 1:
                shift = rng.uniform(-0.2, 0.2, 3) * radii
                current_center = np.array(center_coords) + shift
            else:
                current_center = np.array(center_coords)

            centered_coords = coords - current_center[:, np.newaxis, np.newaxis, np.newaxis]
            ellipsoid_sdf = np.sum((centered_coords / radii[:, np.newaxis, np.newaxis, np.newaxis])**2, axis=0) - 1.0

            # Deform the surface with Perlin noise
            noise_scale = 15 * smoothness
            perlin_noise = ld.generate_3d_perlin_texture(*self.shape, scale=noise_scale, seed=sub_seed)
            deformed_surface = ellipsoid_sdf - irregularity * perlin_noise
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

        # --- 4. edema ---
        lesion_vol = np.zeros_like(img, dtype=np.float32)

        if edema:
            edema_pixels = 5 if edema is True else int(edema)
            edema_mask = binary_erosion(final_mask, np.ones((edema_pixels,)*3)) ^ final_mask
            lesion_vol[edema_mask] = 10
            final_mask |= edema_mask
        lesion_vol[final_mask] = intensity

        img_with_lesion = np.where(final_mask, lesion_vol, img)

        return img_with_lesion, final_mask, center_coords

    def _add_dural_lesion(
        self,
        lesion_type: str,
        volume: float,
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray, ld.Point3D]:
        """Generates a dural-based lesion (SDH or EDH)."""

        lesion_mask, base_img = self._create_dural_lesion(
            volume, lesion_type, seed
        )

        coords = tuple(map(int, center_of_mass(lesion_mask)))
        return base_img, lesion_mask, coords

    def _apply_mass_effect(self, object_mask: np.ndarray,
                           mass_effect: float = 0.2) -> np.ndarray:
        """
        Applies a true 3D mass effect warp, feathered at the boundaries to reduce artifacts.
        """
        img = self.get_CT_number_phantom()
        # --- 1. Calculate the base 3D displacement field ---
        _, indices = distance_transform_edt(~object_mask, return_indices=True)
        output_coords = np.indices(self.shape, dtype=float).transpose(1, 2, 3, 0)
        nearest_object_coords = indices.transpose(1, 2, 3, 0)
        displacement_vectors = output_coords - nearest_object_coords

        # --- 2. Scale the displacement field ---
        num_object_pixels = np.sum(object_mask)
        if num_object_pixels > 0:
            max_displacement = np.cbrt(num_object_pixels)
        else:
            return img  # No warp if object mask is empty.

        decay_factor = 40.0 * mass_effect
        distances = np.linalg.norm(displacement_vectors, axis=-1) + 1e-6
        scale_factors = max_displacement * np.exp(-distances / decay_factor)

        scaled_displacement = (
            displacement_vectors / distances[..., np.newaxis] * scale_factors[..., np.newaxis]
        )

        # --- 3. Create a feathered alpha mask to smooth the warp at the boundary ---
        inclusion_mask = self.warp_inclusion_mask
        if not np.any(inclusion_mask):
            return img  # Nothing to warp

        # Calculate distance from every point inside the inclusion mask to its nearest edge.
        dist_to_edge = distance_transform_edt(inclusion_mask)

        # Feathering transition zone in pixels.
        feather_width = 20.0

        # Create a smooth alpha map: 1.0 deep inside, 0.0 at the edge.
        alpha = np.clip(dist_to_edge / feather_width, 0.0, 1.0)

        # Apply the alpha mask to the displacement field.
        feathered_displacement = scaled_displacement * alpha[..., np.newaxis]

        # --- 4. Create the final blended "flow field" ---
        # The "pull" location for a pixel is its original position minus the displacement.
        # This is applied everywhere, but displacement is zero outside the inclusion zone.
        src_coords = output_coords - feathered_displacement

        # --- 5. Apply the transformation using the flow field ---
        final_src_coords_reshaped = src_coords.transpose(3, 0, 1, 2)

        warped_img = map_coordinates(
            img,
            final_src_coords_reshaped,
            order=1,
            prefilter=True,
            cval=np.min(img)
        )

        return warped_img.astype(img.dtype)

    def _create_dural_lesion(
        self,
        desired_volume_ml: float,
        hematoma_type: str,
        seed: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Core logic to generate a tapered dural lesion mask and the associated warped brain.
        Uses an iterative approach to match the desired volume.
        """
        rng = np.random.default_rng(seed)
        base_img = self.get_CT_number_phantom()
        dura_map = self.get_dura_map()
        voxel_volume_ml = (self.dx * self.dy * self.dz) / 1000.0

        num_slices = ld.coverage_from_volume(
            volume=desired_volume_ml, hematoma_type=hematoma_type, slice_thickness=self.dz
        )
        if num_slices == 0:
            return np.zeros_like(base_img, dtype=bool), base_img

        # Initial estimate for the central slice area
        avg_area_mm2 = (desired_volume_ml * 1000) / (num_slices * self.dz)
        center_slice_area_mm2 = 1.5 * avg_area_mm2

        final_lesion_mask = np.zeros_like(base_img, dtype=bool)

        # Iteratively correct the volume
        for i in range(3):  # Try up to 3 times to correct the volume
            # --- Generate the lesion mask based on the current area estimate ---
            hemisphere_mask = np.zeros_like(dura_map, dtype=bool)
            mid_y = self.shape[2] // 2
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
                init_slice_idx: ld.connect_points(
                    start=start_point, end=end_point, boundary=current_dura_map[init_slice_idx],
                    hematoma_type=hematoma_type
                )[0]
            }

            half_slices = num_slices / 2.0

            # Propagate downwards
            prev_start, prev_end = start_point, end_point
            for j in range(1, int(np.ceil(half_slices))):
                slice_idx = init_slice_idx + j
                if slice_idx >= self.shape[0]: break
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

            current_mask = np.zeros_like(base_img, dtype=bool)
            for idx, filled_array in all_filled_arrays.items():
                current_mask[idx] = filled_array

            current_mask &= self.warp_inclusion_mask
            current_mask &= ~self.warp_exclusion_mask
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

        return final_lesion_mask, base_img

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
                costs, start=tuple(initial_new_start), end=tuple(initial_new_end)
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

        filled_array, _, _ = ld.connect_points(
            start=final_start, end=final_end, boundary=dura_slice_map, hematoma_type=h_type
        )
        return filled_array, final_start, final_end
