'''
module for working with phantoms
'''

import os

import numpy as np
from skimage.morphology import binary_erosion
from monai.transforms import Resize, RandAffine, Affine

from scipy.ndimage import (center_of_mass,
                           distance_transform_edt)

from VITools import Phantom

from ..artifact_generation import transform_image_label_pair
from .. import lesion_definition as ld


def sphere_radius_from_volume(volume):
    '''
    Converts volume in mL to radii in mm
    '''
    return np.power(3/4/np.pi*volume*1000, 1/3)


def calculate_eccentricity(a, b):
    if a > b:
        return np.sqrt(1 - b**2/a**2)
    else:
        return np.sqrt(1 - a**2/b**2)


def get_closest_key(key, dictionary):
    diffs = {abs(k - key): k for k in dictionary}
    return diffs[min([o for o in diffs])]


def calc_mean_eccentricity(a, b, c):
    return np.mean([calculate_eccentricity(a, b),
                    calculate_eccentricity(a, c),
                    calculate_eccentricity(b, c)])


def get_eccentricity_dict():
    eccentricity_dict = {}
    sample_range = np.linspace(0.1, 1, 10)
    for a in sample_range:
        for b in sample_range:
            for c in sample_range:
                sample_eccentricity = calculate_eccentricity(a, b), \
                                      calculate_eccentricity(a, c), \
                                      calculate_eccentricity(b, c)
                sample_eccentricity = np.round(
                    np.stack(sample_eccentricity).mean(), decimals=2)
                eccentricity_dict[float(sample_eccentricity)] =\
                    list(map(float, (a, b, c)))
    return {c: eccentricity_dict[c] for c in sorted(eccentricity_dict)}


def get_semi_major_axes(eccentricity, seed=None):
    eccentricity_dict = get_eccentricity_dict()
    key = get_closest_key(eccentricity, eccentricity_dict)
    foci = eccentricity_dict[key]
    rng = np.random.default_rng(seed)
    rng.shuffle(foci)
    return np.array(foci)


def get_mean_age(age_range: str):
    return (float(age_range.split('-')[1])+float(age_range.split('-')[0]))/2


def resize(phantom, shape, **kwargs):
    resize = Resize(max(shape), size_mode='longest', **kwargs)
    resized = resize(phantom[None])[0]
    return resized


class LesionPhantom(Phantom):
    '''A Phantom object with methods for inserting lesions.

    '''
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

    def __init__(self, img: np.ndarray, spacings: tuple = (1, 1, 1), **kwargs):
        """Initializes a LesionPhantom object.

        Args:
            img (np.ndarray): The image data of the phantom.
            spacings (tuple, optional): The voxel spacings in (dz, dx, dy)
                format. Defaults to (1, 1, 1).
            **kwargs: Additional keyword arguments passed to the parent class's
                initializer.
        """
        super().__init__(img, spacings, **kwargs)
        self._lesion = []
        self._lesion_coords = []
        self.lesion_type = []
        self.lesion_intensity = []  # HU
        self.mass_effect = False
        self.warp_exclusion_mask = self.get_warp_exclusion_mask()
        self.warp_inclusion_mask = self.get_warp_inclusion_mask()

    def get_warp_exclusion_mask(self):
        return np.zeros(self.shape, dtype=bool)

    def get_warp_inclusion_mask(self):
        return np.ones(self.shape, dtype=bool)

    def resize(self, shape: tuple, **kwargs) -> None:
        """Resizes the phantom array and adjusts spacings.

        This method resizes the phantom array, warp exclusion mask, and warp
        inclusion mask to the specified shape, adjusting spacings accordingly.

        Args:
            shape (tuple): The new shape for the phantom array.
            **kwargs: Additional keyword arguments passed to the resize
                function.

        Returns:
            None: The method modifies the array and masks in place.
        """
        super().resize(shape, **kwargs)
        self.warp_exclusion_mask = resize(
            self.warp_exclusion_mask, shape, **kwargs).astype(bool)
        self.warp_inclusion_mask = resize(
            self.warp_inclusion_mask, shape, **kwargs).astype(bool)

    def get_lesion_volume(self, unit='mL'):
        """Calculates the volume of the lesion.

        The volume can be determined in either milliliters (mL) or cubic 
        millimeters (mm3).

        Args:
            unit (str): The unit of the lesion volume. Defaults to 'mL'.

        Returns:
            float: The volume of the lesion.
        """
        vol_mm3 = self.dx * self.dy * self.dz * self.get_lesion_mask().sum()
        if unit == 'mm3':
            return vol_mm3
        if unit == 'mL':
            return vol_mm3 / 1000

    def __repr__(self) -> str:
        """Returns the string representation of the LesionPhantom object.

        This representation includes details from the parent class, along with
        the number of lesions, their coordinates, and the mass effect status.

        Returns:
            str: The string representation of the object.
        """
        repr = super().__repr__() + f'''
        Number of lesions: {len(self._lesion_coords)}
        Lesion locations [voxel index (z, x, y)]: {self._lesion_coords}
        Mass effect: {self.mass_effect}
        '''
        return repr

    @property
    def spacings(self):
        """The voxel spacings of the phantom.

        Returns:
            tuple: Voxel spacings in (dz, dx, dy) format.
        """
        return self.dz, self.dx, self.dy

    def get_noise_texture(self, noise_type='perlin', contrast=80,
                          contrast_std=1, scale=15, seed=None, **kwargs):
        """Generates a noise texture for the phantom.

        Args:
            noise_type (str): Type of noise to generate. Options include
                'perlin', 'simplex', 'fbm', and 'filtered_noise'.
            contrast (float, optional): Contrast level of the noise texture.
                Defaults to 80.
            contrast_std (float, optional): Standard deviation of the contrast.
                Defaults to 1.
            scale (float, optional): The 'zoom' level of the noise. Larger
                values result in lower frequency. Defaults to 15.
            **kwargs: Additional keyword arguments to pass to the noise
                generation function.

        Returns:
            numpy.ndarray: The generated noise texture for the phantom.

        Raises:
            ValueError: If an unknown noise type is provided or if
                contrast_std is negative.

        """
        if noise_type not in ['perlin', 'simplex', 'fbm', 'filtered_noise']:
            raise ValueError(
                f'Unknown noise type: {noise_type}. '
                  'Options are: perlin, simplex, fbm, filtered_noise.')
        if contrast_std < 0:
            raise ValueError(f'contrast_std {contrast_std} must be >= 0')

        if seed is None:
            seed = np.random.randint(0, 1e6)

        if contrast_std == 0:
            noise_type = 'constant'
            noise_texture = np.full(self.shape, contrast, dtype=np.float32)
        if noise_type == 'perlin':
            noise_texture = ld.generate_3d_perlin_texture(*self.shape,
                                                          scale=scale,
                                                          seed=seed,
                                                          **kwargs)
        elif noise_type == 'simplex':
            noise_texture = ld.generate_3d_simplex_texture(*self.shape,
                                                           scale=scale,
                                                           **kwargs)
        elif noise_type == 'fbm':
            noise_texture = ld.generate_3d_fbm_texture(*self.shape,
                                                       seed=seed,
                                                       scale=scale, **kwargs)
        elif noise_type == 'filtered_noise':
            noise_texture = ld.generate_3d_filtered_noise_texture(*self.shape,
                                                                  **kwargs)
        return noise_texture*contrast_std*contrast + contrast

    def insert_lesion(self, lesion_type, volume=5, intensity=50,
                      mass_effect: float = 0.5, seed=None,
                      texture_kwargs: dict = {},
                      iph_kwargs: dict = {}):
        """Inserts a lesion of a specified type into the phantom array.

        Args:
            lesion_type (str): Type of the lesion. Options include 'IPH'
                (intraparenchymal), 'EDH' (epidural), and 'SDH' (subdural).
            volume (float, optional): Volume of the lesion in mL.
                Defaults to 5.
            intensity (int, optional): CT number of the lesion in HU.
                Defaults to 50.
            mass_effect (bool | float, optional): Whether to apply mass effect
                processing to displace brain tissue following lesion insertion.
                Defaults to False. If True, a default strength of 0.5 is used.
                If a float is provided, it controls the strength of the effect,
                where 1.0 causes a large degree of warping.
            seed (int, optional): Seed for reproducible lesion insertion.
                Defaults to None.
            iph_kwargs: Additional keyword arguments for the IPH lesion
                insertion function.
            texture_kwargs (dict, optional): Arguments for the noise texture
                generation. If None, default parameters are used.

        Returns:
            LesionPhantom: The updated LesionPhantom object.

        Raises:
            ValueError: If an unknown lesion type is provided, if the volume
            TypeError:  is not a positive integer, or if the intensity is not a
                valid CT number.
            RuntimeError: If the requested volume is too large for the phantom.
        """
        if volume <= 0:
            return self
        self.lesion_type.append(lesion_type)
        if mass_effect is True:
            mass_effect = 1.0
        self.mass_effect = mass_effect
        if lesion_type == 'IPH':
            img_w_lesion, lesion_image, lesion_coords = \
                self.add_round_lesion(volume=volume, intensity=intensity,
                                      mass_effect=mass_effect, seed=seed,
                                      texture_kwargs=texture_kwargs,
                                      **iph_kwargs)
        elif lesion_type in ['EDH', 'SDH']:
            img_w_lesion, lesion_image, lesion_coords = \
                self.add_dural_lesion(volume, lesion_type, intensity,
                                      mass_effect=mass_effect, seed=seed,
                                      texture_kwargs=texture_kwargs)
        else:
            raise ValueError(f'''unknown lesion type passed: {lesion_type}.
                             Currently accepts IPH (intraparenchymal),
                             EDH (epidural), or SDH (subdural).''')
        original = self.get_CT_number_phantom()
        diff_img = abs(img_w_lesion - original)
        img_w_lesion[diff_img > intensity] =\
            original[self.get_warp_inclusion_mask()].mean()
        self._phantom = img_w_lesion
        self._lesion.append(lesion_image)
        self._lesion_coords.append(lesion_coords)
        self.lesion_intensity.append(float(intensity))
        return self

    def apply_transform(self, transform: RandAffine | Affine, seed=None):
        if not self._lesion:
            if seed:
                transform.set_random_state(seed=seed)
            self._phantom = transform(self._phantom)
            return
        self._phantom, self._lesion[0] =\
            transform_image_label_pair(transform,
                                       self._phantom,
                                       self.get_lesion_mask(),
                                       seed=seed)

    def add_round_lesion(self,
                         volume: int = 10,
                         intensity: int = 50,
                         material: str = 'white matter',
                         eccentricity: float = 0.5,
                         mass_effect: float | bool = 0.5,
                         edema: bool | int = False,
                         complexity: int = 3,
                         overlap: float = 0.4,
                         seed: int | None = None,
                         texture_kwargs: dict | None = None) -> tuple:
        """Adds a round lesion to an image in a random location.

        This function inserts a lesion, potentially with complex
        characteristics, into a specified material region of an image.
        It allows for detailed customization of the lesion's shape, intensity,
        and secondary effects like edema or mass effect.

        Args:
            volume (int | list[int]): The volume of the sphere lesion in mL.
                If a list is provided, it will create concentric lesions.
            intensity (int | list[int]): The intensity of the sphere lesion in
                Hounsfield Units (HU). If a list is provided, it will create
                concentric lesions with the corresponding intensities.
            material (str): The material region to insert the lesion into. See
                the `self.materials` attribute for available options.
            eccentricity (float): A value between 0.0 and 1.0 that defines how
                elongated the lesion is. A value of 0 is spherical, while 1.0
                is very oblong.
            mass_effect (bool): If False or 0.0, no mass effect is
                applied. A float between 0.0 and 1.0 controls the strength of
                the effect,  where 1.0 causes a large degree of warping. See
                the `insert_with_mass_effect` method for more details.
            edema (bool | int): Specifies an edema layer to add around the
                lesion. If an integer is provided, it defines the thickness of
                the layer in pixels.
            complexity (int): The number of ellipses used to generate the
                lesion shape. A value of 1 creates a single ellipsoid, while
                higher values result in more complex, overlapping shapes.
            overlap (float): The allowed fractional overlap with the white
                matter mask.
            seed (int, optional): A seed for the random number generator to
                ensure reproducible lesion insertion. Defaults to None.
            texture_kwargs (dict, optional): A dictionary of arguments for the
                noise texture generation. If None, default parameters are used.

        Returns:
            tuple: A tuple containing the following three elements:
                - np.ndarray: The image array with the lesion inserted.
                - np.ndarray: A mask of the generated lesion volume.
                - tuple[int, int, int]: The (z, x, y) coordinates of the
                lesion's center.
        """
        rng = np.random.default_rng(seed)

        voxel_size = np.power(self.dx*self.dy*self.dz, 1/3)
        r = sphere_radius_from_volume(volume) / voxel_size
        img = self.get_CT_number_phantom()
        mask = self.get_material_mask(material).astype(int)

        lesion_vol = np.zeros_like(img)
        valid_points = distance_transform_edt(mask) > (r * overlap)
        r_int = np.ceil(r).astype(int)
        valid_points[:r_int] = False  # ensures lesion is not at the boundary of the phantom
        valid_points[-r_int:] = False
        if not valid_points.any():
            raise RuntimeError(f'Requested volume: {volume} mL too \
large, try smaller volume')
        # lower distance threshold `r` to allow overlap
        z, x, y = np.argwhere(valid_points)[rng.integers(0,
                                            valid_points.sum())]

        lesion_vol = np.full(img.shape, fill_value=-1000)
        transform = RandAffine(prob=1, translate_range=[r, r])
        transform.set_random_state(seed)
        if os.name == 'nt':
            seed = False
            # windows compatibility, monai transform crashes windows kernel
            transform = lambda o: o  # return self

        for _ in range(complexity):
            axes = get_semi_major_axes(eccentricity, seed)
            foci = r * axes
            if complexity > 1:
                correction = np.power(3/(4*np.pi*complexity), 1/3)+overlap
            else:
                correction = 1
            foci = foci*correction
            sphere = ld.elliptical_lesion(img.shape, center=(z, x, y),
                                          radius=foci,
                                          random_rotate=seed)
            sphere = transform(sphere).astype(bool)
            lesion_vol[sphere] = intensity
        lesion_mask = lesion_vol > -1000
        if texture_kwargs:
            lesion_texture = self.get_noise_texture(contrast=intensity,
                                                    seed=seed,
                                                    **texture_kwargs)
            lesion_vol[lesion_mask] = lesion_texture[lesion_mask]
        if edema:
            edema_pixels = 5
            edema_HU = 10
            edema = edema_pixels if edema is True else edema
            edema_mask = binary_erosion(lesion_mask,
                                        np.ones(3*[edema])) ^ lesion_mask
            lesion_vol[edema_mask] = edema_HU
            lesion_mask = lesion_vol > -1000
        lesion_mask = lesion_mask & ~self.get_skull_map()
        img_w_lesion = img.copy()
        img_w_lesion[lesion_mask] = lesion_vol[lesion_mask]

        if mass_effect:
            warped = self.insert_with_mass_effect(img,
                                                  lesion_mask,
                                                  strength=mass_effect)
            warped[lesion_mask] = img_w_lesion[lesion_mask]
            img_w_lesion[lesion_mask.sum(axis=(1, 2)) > 0] =\
                warped[lesion_mask.sum(axis=(1, 2)) > 0]

        return img_w_lesion, lesion_mask, (z, x, y)

    def insert_with_mass_effect(self, img, lesion, strength=0.2):
        if img.ndim == 2:
            img = img[None]
        assert img.ndim == 3

        warped = np.zeros_like(img)
        inclusion_mask = self.get_warp_inclusion_mask()
        for idx in range(lesion.shape[0]):
            if not lesion[idx].any():
                continue
            warped[idx] = ld.warp_slice(axial_slice=img[idx],
                                        object_mask=lesion[idx],
                                        inclusion_mask=inclusion_mask[idx],
                                        strength=strength)
        warped = warped.astype(img.dtype)
        return warped

    def add_dural_lesion(self, volume, lesion_type, intensity,
                         seed=None, mass_effect: float = 0.2,
                         texture_kwargs: dict = {}):
        HU_volume = self.get_CT_number_phantom()
        lesion_vol, HU_volume = self.dural_lesion(
            desired_volume=volume,
            hematoma_type=lesion_type,
            mass_effect=mass_effect,
            seed=seed)
        if not isinstance(HU_volume, np.ndarray):
            HU_volume = HU_volume.numpy()

        img_w_lesion = HU_volume.copy()
        img_w_lesion[lesion_vol] = intensity
        if texture_kwargs:
            lesion_texture = self.get_noise_texture(contrast=intensity,
                                                    seed=seed,
                                                    **texture_kwargs)
            img_w_lesion[lesion_vol] = lesion_texture[lesion_vol]
        z, x, y = center_of_mass(lesion_vol)
        return img_w_lesion, lesion_vol, (int(z), int(x), int(y))

    def dural_lesion(self, desired_volume, hematoma_type, mass_effect=0.2,
                     seed: int | None = None):

        mask = self.get_warp_inclusion_mask()
        random = np.random.default_rng(seed)

        num_slices = ld.coverage_from_volume(volume=desired_volume,
                                             hematoma_type=hematoma_type,
                                             slice_thickness=self.dz)
        ab = (desired_volume*2000)/(num_slices * self.dz)  # using ABC/2
        # formula (although /2000 for mL and mm)
        if hematoma_type == 'EDH':
            desired_distance = np.sqrt(4*ab)  # assume that length of epidural
            #  hemorrhage is about 4 times the width
        elif hematoma_type == 'SDH':
            desired_distance = np.sqrt(11*ab)  # assume that length of subdural
            #  hemorrhage is about 11 times the width

        HU_array = self.get_CT_number_phantom()

        possible_init_slices = np.argwhere(
            self.get_warp_inclusion_mask().sum(axis=(1, 2)) > 0
                ).ravel()[num_slices//2:-num_slices//2]

        init_slice = int(random.choice(possible_init_slices))

        # initialize arrays, maps, and masks
        new_volume = np.copy(HU_array)
        boundary = self.get_dura_map()
        hemorrhage_mask = np.zeros_like(boundary)

        distances = [(desired_distance-5)/self.spacings[2], (desired_distance+5)/self.spacings[2]]

        hemisphere = random.choice(['left', 'right'])  # can either be random or pre-defined
        if hemisphere == 'left':
            boundary[:, :, (int(HU_array.shape[2]/2) - 10):None] = 0.0
        elif hemisphere == 'right':
            boundary[:, :, :(int(HU_array.shape[2]/2) + 10)] = 0.0

        # begin iteration
        iter_flag = True
        slice_counter = slice_idx = 0
        while iter_flag:

            current_vol = ((self.dx * self.dy * self.dz) *
                           hemorrhage_mask.sum())/1000
            if current_vol > desired_volume:
                iter_flag = False

            if slice_counter == 0:  # need to do the slice in the middle of the hemorrhage, same as before

                tol = 2000
                count = 0
                failure_occured = False
                while count < tol:
                    temp_boundary = boundary[init_slice]
                    dura_idx = np.argwhere(temp_boundary == 1.0)

                    try:
                        # choose a random start point, and calculate distance from all available boundary voxels to start point
                        start_point = dura_idx[random.choice(range(len(dura_idx)))]

                        distance_idx = np.zeros(len(dura_idx))
                        for i in range(len(dura_idx)):
                            distance_idx[i] = np.sqrt((start_point[0] - dura_idx[i][0])**2 + (start_point[1] - dura_idx[i][1])**2)

                        # create list of possible end points and choose one at random
                        close_voxel_list = np.where(np.logical_and(distance_idx > distances[0], distance_idx < distances[1]))
                        end_point = dura_idx[random.choice(close_voxel_list[0])]

                        orig_start = new_start = start_point
                        orig_end = new_end = end_point
                    except:
                        count += 1
                        init_slice = int(random.choice(np.linspace(0, int(HU_array.shape[0]/2), int(HU_array.shape[0]/2) + 1)))
                        if count == tol:
                            failure_occured = True
                    else:
                        count = tol
                if failure_occured:
                    raise RuntimeError(f'lesion insertion failed with requested volume: {desired_volume} mL, try a smaller volume')

                # connect the start and end points of the hemorrhage 
                filled_array, boundary_coords, _ =\
                    ld.connect_points(start=orig_start,
                                      end=orig_end,
                                      boundary=temp_boundary,
                                      hematoma_type=hematoma_type)

                if mass_effect:
                    inclusion_mask = binary_erosion(mask[init_slice],
                                                    np.ones((5, 5)))
                    exclusion_mask = self.get_warp_exclusion_mask()[init_slice]
                    object_mask = filled_array & inclusion_mask &\
                        ~exclusion_mask
                    object_mask = filled_array & inclusion_mask
                    warped_slice = ld.warp_slice(axial_slice=HU_array[init_slice],
                                                 object_mask=object_mask,
                                                 inclusion_mask=inclusion_mask,
                                                 strength=mass_effect)
                    warped_slice[self.get_warp_exclusion_mask()[init_slice]] =\
                        HU_array[init_slice][self.get_warp_exclusion_mask()[init_slice]]

                hemorrhage_mask[init_slice, :, :] = filled_array

                slice_counter += 1

            # starting from init_slice, first move down slices 
            # then, move up from init_slice while doing the same
            if slice_counter <= (num_slices-1)/2:  # move down from init_slice
                slice_idx = slice_counter
            elif slice_counter > (num_slices-1)/2:  # start moving up from init_slice
                slice_idx = -1*(slice_counter - int((num_slices-1)/2))

            temp_boundary = boundary[init_slice-slice_idx]
            dura_idx = np.argwhere(temp_boundary == 1.0)

            if len(dura_idx) != 0:  # check that top or bottom of brain wasn't reached

                # find closest boundary point to previous start
                distance_idx = np.zeros((len(dura_idx), 2))

                distance_from_start = np.zeros(len(dura_idx))
                distance_from_end = np.zeros(len(dura_idx))

                if abs(slice_idx) == 1:
                    for i in range(len(dura_idx)):
                        distance_from_start[i] = np.sqrt((orig_start[0] - dura_idx[i][0])**2 + (orig_start[1] - dura_idx[i][1])**2)
                        distance_from_end[i] = np.sqrt((orig_end[0] - dura_idx[i][0])**2 + (orig_end[1] - dura_idx[i][1])**2)
                else:
                    for i in range(len(dura_idx)):
                        distance_from_start[i] = np.sqrt((new_start[0] - dura_idx[i][0])**2 + (new_start[1] - dura_idx[i][1])**2)
                        distance_from_end[i] = np.sqrt((new_end[0] - dura_idx[i][0])**2 + (new_end[1] - dura_idx[i][1])**2)

                new_start = dura_idx[np.argmin(distance_from_start)]
                new_end = dura_idx[np.argmin(distance_from_end)]

                filled_array, boundary_coords, _ =\
                    ld.connect_points(start=new_start,
                                      end=new_end,
                                      boundary=temp_boundary,
                                      hematoma_type=hematoma_type)
                try:
                    new_start = boundary_coords[1:-1][0]
                    new_end = boundary_coords[1:-1][-1]
                except:
                    new_start = dura_idx[np.argmin(distance_from_start)]
                    new_end = dura_idx[np.argmin(distance_from_end)]

                if mass_effect:
                    inclusion_mask = binary_erosion(mask[init_slice-slice_idx],
                                                    np.ones((5, 5)))
                    axial_slice = HU_array[init_slice-slice_idx]
                    exclusion_mask = self.get_warp_exclusion_mask()[init_slice-slice_idx]
                    object_mask = filled_array & inclusion_mask & ~ exclusion_mask
                    warped_slice = ld.warp_slice(axial_slice=axial_slice,
                                                 object_mask=object_mask,
                                                 inclusion_mask=inclusion_mask,
                                                 strength=mass_effect)
                    warped_slice[self.get_warp_exclusion_mask()[init_slice-slice_idx]] =\
                        HU_array[init_slice-slice_idx][
                            self.get_warp_exclusion_mask()[init_slice-slice_idx]
                            ]
                    new_volume[init_slice-slice_idx] = warped_slice

                hemorrhage_mask[init_slice-slice_idx] = filled_array

                slice_counter += 1

                if slice_counter == num_slices:
                    iter_flag = False

            else:
                slice_counter += 1
                if slice_counter == num_slices:
                    iter_flag = False

        return hemorrhage_mask.astype(bool), new_volume
