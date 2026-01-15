"""
Refactored module for creating and manipulating medical imaging phantoms with
lesions.
"""

from typing import List, Tuple, Union

import numpy as np
from scipy.ndimage import (center_of_mass,
                           distance_transform_edt,
                           map_coordinates)

from VITools import Phantom

# Assuming these are local modules with the specified functions.
# For this example, placeholder functions are used where necessary.
from .. import lesion_definition as ld
from ..transforms import resize, RandAffine, Affine

# --- Type Aliases for Clarity ---
Shape3D = Tuple[int, int, int]
Spacing3D = Tuple[float, float, float]


def get_mean_age(age_range: str):
    return (float(age_range.split('-')[1])+float(age_range.split('-')[0]))/2


class LesionPhantom(Phantom):
    """
    A Phantom object with methods for inserting realistic lesions.

    Phantoms are responsible for loading from different sources, resizing, and
    managing side effects from inserting lesions such as mass effect.
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
        },
                    'IVH': {
        'volume': [0, 100],  # Example range
        'intensity': [40, 80],
        'mass_effect': False, # Typically IVH doesn't deform skull/brain in the same way, but expands ventricles.
        'seed': None
        },
                    'SAH': {
        'volume': [0, 100], # Example range
        'intensity': [40, 80],
        'mass_effect': False,
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

    def insert_lesion(
        self,
        lesion: ld.Lesion,
        mass_effect: Union[bool, float] = False,
    ):
        """
        Primary method to insert a lesion of a specified type into the phantom.

        Args:
            lesion: Lesion object.
            mass_effect: If False/0.0, no mass effect. If True, uses 1.0.
                         A float controls the warp strength.
        """
        # add mass_effect
        img_w_lesion = self.get_CT_number_phantom().copy()
        img_w_lesion[lesion.mask] = lesion.image[lesion.mask]
        mass_effect_strength = 0
        if mass_effect is True:
            mass_effect_strength = 1.0
        elif isinstance(mass_effect, (float, int)):
            mass_effect_strength = float(mass_effect)

        if mass_effect_strength > 0:
            img_w_lesion = self._apply_mass_effect(lesion.mask,
                                                   mass_effect_strength)
        lesion.mass_effect = mass_effect_strength
        # correct and erroneaous voxels
        intensity_hu = lesion.image[lesion.mask].mean()
        diff = self.get_CT_number_phantom() - img_w_lesion
        img_w_lesion[abs(diff) > intensity_hu] =\
            self.get_CT_number_phantom()[abs(diff) > intensity_hu]

        img_w_lesion[lesion.mask] = lesion.image[lesion.mask]

        self.lesions.append(lesion)
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
        seed = seed or np.random.randint(1e6)
        transform.set_random_state(seed=seed)

        # 1. Transform the main phantom image. If the transform is random, its
        # parameters are now fixed ("realized") for subsequent calls.
        self._phantom = transform(self._phantom[None])[0]

        # 2. Apply the *same* realized transform to each lesion mask
        lesions = []
        for lesion in self.lesions:
            # MONAI expects a channel dimension (C, H, W, D) and float type for interpolation
            transform.set_random_state(seed=seed)
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
            lesions.append(lesion)

    def _apply_mass_effect(self, object_mask: np.ndarray,
                           mass_effect: float = 0.2) -> np.ndarray:
        """
        Applies a true 3D mass effect warp, feathered at the boundaries to reduce artifacts.
        """
        # Ensure object mask only contains valid pixels within the phantom and
        # respects inclusion/exclusion zones.
        object_mask &= self.warp_inclusion_mask
        object_mask &= ~self.warp_exclusion_mask

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
