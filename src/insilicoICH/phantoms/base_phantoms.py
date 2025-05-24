'''
module for working with phantoms
'''

import os

import numpy as np
import skimage as ski
from skimage.morphology import binary_erosion
from monai.transforms import Resize, RandAffine, Affine

from scipy.ndimage import (center_of_mass,
                           distance_transform_edt)

from VITools import Phantom

from ..artifact_generation import transform_image_label_pair

from ..lesion_definition import (elliptical_lesion,
                                 insert_dural,
                                 warp_slice,
                                 get_perimeter)


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


def get_transformation_src_dst(lesion: np.ndarray[bool],
                               strength: int = 0):
    '''
    returns `src` and `dst` arrays to insert `lesion` into an img of the same
    shape. e.g. of the head and applies a mass effect warping of the local
    tissues.

    This function takes a solid lesion and extracts the perimeter as the input
     `dst` to `warp_slice` and applies the perimeter `strength` to determine
    how many erosions make up the `src` input to `warp` slice, where higher
    strength yields a greater degree of warping by creating a greater distance
    between `src` and `dst`. See `warp_slice` for more details.
    :param lesion: a 2D binary image of the lesion, must be same shape as img
    '''
    if (strength < 0) or (strength > 1):
        raise ValueError(f'strength {strength} is not in allowed range [0, 1]')
    footprint = int(strength*np.ceil(distance_transform_edt(lesion).max()))
    dst = get_perimeter(lesion)
    src = get_perimeter(ski.morphology.binary_erosion(lesion,
                                                      np.ones(2*[footprint])))
    return src, dst


def get_mean_age(age_range: str):
    return (float(age_range.split('-')[1])+float(age_range.split('-')[0]))/2


def resize(phantom, shape, **kwargs):
    resize = Resize(max(shape), size_mode='longest', **kwargs)
    resized = resize(phantom[None])[0]
    return resized


class LesionPhantom(Phantom):
    '''
    A Phantom object with methods for inserting lesions.
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
        '''
        Initializes a LesionPhantom object.

        Parameters:
        - img: numpy.ndarray, image data of the phantom.
        :param spacings: tuple, voxel spacings (dz, dx, dy). Default is (1, 1, 1).
        '''
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
        '''
        Resizes the phantom array to the given shape and adjusts the spacings accordingly.

        :param shape: tuple, new shape for the phantom array
        '''
        super().resize(shape, **kwargs)
        self.warp_exclusion_mask = resize(self.warp_exclusion_mask, shape, **kwargs).astype(bool)
        self.warp_inclusion_mask = resize(self.warp_inclusion_mask, shape, **kwargs).astype(bool)

    def get_lesion_volume(self, unit='mL'):
        '''
        Calculates the volume of the lesion in either milliliters (mL) or cubic millimeters (mm3).

        :param unit: str, unit of the lesion volume. Default is 'mL'.
        :return: float, volume of the lesion.
        '''
        vol_mm3 = self.dx * self.dy * self.dz * self.get_lesion_mask().sum()
        if unit == 'mm3':
            return vol_mm3
        if unit == 'mL':
            return vol_mm3 / 1000

    def __repr__(self) -> str:
        '''
        Returns a string representation of the LesionPhantom object.

        :return: str, string representation of the LesionPhantom object.
        '''
        repr = super().__repr__() + f'''
        Number of lesions: {len(self._lesion_coords)}
        Lesion locations [voxel index (z, x, y)]: {self._lesion_coords}
        Mass effect: {self.mass_effect}
        '''
        return repr

    @property
    def spacings(self):
        '''
        Returns the voxel spacings of the phantom.

        :return: tuple, voxel spacings (dz, dx, dy).
        '''
        return self.dz, self.dx, self.dy

    def insert_lesion(self, lesion_type, volume=5, intensity=50,
                      mass_effect=False, seed=None, **kwargs):
        '''
        Inserts a lesion of a specified type into the phantom array.

        :param lesion_type: str, type of the lesion. Options include 'IPH' (intraparenchymal), 'EDH' (epidural), and 'SDH' (subdural).
        :param volume: float, volume of the lesion in mL. Default is 5.
        :param intensity: int, CT number of the lesion in HU. Default is 50.
        :param mass_effect: bool, whether to apply mass effect processing to displace brain tissue following lesion insertion. Default is False.
        :param seed: int, optional seed for reproducible lesion insertion. Default is None.
        :param kwargs: additional keyword arguments to pass to the lesion insertion function.
        :return: self, the updated LesionPhantom object.
        '''
        if volume <= 0:
            return self
        self.lesion_type.append(lesion_type)
        self.mass_effect = mass_effect
        if lesion_type == 'IPH':
            img_w_lesion, lesion_image, lesion_coords = \
                self.add_round_lesion(volume=volume, intensity=intensity,
                                      mass_effect=mass_effect, seed=seed, **kwargs)
        elif lesion_type in ['EDH', 'SDH']:
            img_w_lesion, lesion_image, lesion_coords = \
                self._add_dural_lesion(volume, lesion_type, intensity,
                                       mass_effect=mass_effect, seed=seed)
        else:
            raise ValueError(f'unknown lesion type passed: {lesion_type}. '
                             'Currently accepts IPH (intraparenchymal), EDH (epidural), or SDH (subdural).')
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
                         mass_effect: bool | float = 0.5,
                         edema: bool | int = False,
                         complexity: int = 3,
                         overlap: float = 0.4,
                         seed: int | None = None,
                         **kwargs) -> tuple:
        '''
        adds round lesion to img in random location within mask of size radius
        and intensity level intensity

        See parameter descriptions below for further modifications that can
        be added:

        :param volume: int or list of ints, volume of the sphere lesion in mL,
            if provided a list it will make concentric lesions
        :param intensity: int or list of ints, intensity of the sphere lesion
            in HU, if provided a list it will make concentric lesions of
            intensities
        :param material: which material region to insert lesion into,
            self.materials for options
        :param eccentricity: between 0, 1 defines how elongated the lesions
            are, with 0 being spherical, 1 being very oblong
        :param mass_effect: bool or float between [0, 1], if 0 or False no
            mass effect is applied, a mass effect > 0 but < 1 controls mass
            effect strength where 1 is a large degree of mass effect warping
            and 0.2 is a smaller amount of warping, see
            `insert_with_mass_effect` for more details
        :param edema: bool or int, referring to the number of pixels thick of
            an edema layer to add around the lesion
        :param complexity: int, number of ellipses to aid with
            random jiggle, 1 gives a single ellipsoid, increasing to 2 or 3
            yields overlapping ellipsoids with a more complex shape.
        :param overlap: float, allowed overlap with the white matter mask
        :param seed: optional, defaults to None, set seed for reproducible
            lesion insertion

        :return: img_w_lesion, lesion_vol, (z, x, y)
        '''
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
            seed = False  # windows compatibility, monai transform crashes windows kernel
            transform = lambda o: o  # return self

        for _ in range(complexity):
            axes = get_semi_major_axes(eccentricity, seed)
            foci = r * axes
            if complexity > 1:
                correction = np.power(3/(4*np.pi*complexity), 1/3)+overlap
            else:
                correction = 1
            foci = foci*correction
            sphere = elliptical_lesion(img.shape, center=(z, x, y),
                                       radius=foci,
                                       random_rotate=seed)
            sphere = transform(sphere).astype(bool)
            lesion_vol[sphere] = intensity
        lesion_mask = lesion_vol > -1000

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
            if mass_effect is True:
                mass_effect = 0.5
            warped = self.insert_with_mass_effect(img,
                                                  lesion_mask,
                                                  strength=mass_effect)
            warped[lesion_mask] = img_w_lesion[lesion_mask]
            img_w_lesion[lesion_mask.sum(axis=(1, 2)) > 0] =\
                warped[lesion_mask.sum(axis=(1, 2)) > 0]

        return img_w_lesion, lesion_mask, (z, x, y)

    def insert_with_mass_effect(self, img, lesion, strength=1):
        if img.ndim == 2:
            img = img[None]
        assert img.ndim == 3

        warped = np.zeros_like(img)
        exclusion_mask = self.get_warp_exclusion_mask()
        inclusion_mask = self.get_warp_inclusion_mask()
        for idx in range(lesion.shape[0]):
            if not lesion[idx].any():
                continue
            src_coords, dst_coords = self.get_warp_coordinates(lesion, idx)
            warped[idx] = warp_slice(axial_slice=img[idx],
                                     exclusion_mask=exclusion_mask[idx],
                                     inclusion_mask=inclusion_mask[idx],
                                     src=src_coords, dst=dst_coords,
                                     hematoma_type='IPH')
        return warped

    def get_warp_coordinates(self, lesion, idx):
        # get lesion coordinates
        src, dst = get_transformation_src_dst(lesion[idx])
        warp_dst = np.argwhere(dst)
        warp_src = np.argwhere(src)
        return warp_src, warp_dst

    def _add_dural_lesion(self, volume, lesion_type, intensity,
                          seed=None, mass_effect=True):

        HU_volume = self.get_CT_number_phantom()
        lesion_vol, HU_volume = insert_dural(
            phantom=self,
            desired_volume=volume,
            hematoma_type=lesion_type,
            mass_effect=mass_effect,
            seed=seed)
        if not isinstance(HU_volume, np.ndarray):
            HU_volume = HU_volume.numpy()

        img_w_lesion = HU_volume.copy()
        img_w_lesion[lesion_vol] = intensity
        z, x, y = center_of_mass(lesion_vol)
        return img_w_lesion, lesion_vol, (int(z), int(x), int(y))
