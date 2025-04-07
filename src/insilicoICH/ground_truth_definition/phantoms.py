'''
module for working with phantoms
'''

from pathlib import Path
import os
from warnings import warn
from collections import OrderedDict

import numpy as np
import nibabel as nib
import nrrd
import pandas as pd
import skimage as ski
from dotenv import load_dotenv
from monai.transforms import Resize, RandAffine, Affine, ResizeWithPadOrCrop
from .utils import download_and_extract_archive

from . import dicom_to_voxelized_phantom
from ..artifact_generation import transform_image_label_pair

from ..lesion_definition import (elliptical_lesion,
                                 insert_dural,
                                 warp_slice,
                                 get_perimeter)
from scipy.ndimage import (center_of_mass,
                           distance_transform_edt,
                           binary_erosion)

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


def insert_with_mass_effect(img, lesion, boundary, strength=1):
    if img.ndim == 2:
        img = img[None]
    assert img.ndim == 3

    warped = np.zeros_like(img)
    for idx in range(lesion.shape[0]):
        if not lesion[idx].any():
            continue
        src, dst = get_transformation_src_dst(lesion[idx], strength)
        dst_coords = np.argwhere(dst)
        src_coords = np.argwhere(src)
        warped[idx] = warp_slice(img[idx], boundary[idx],
                                 src_coords, dst_coords, hematoma_type='round')
    return warped


def get_mean_age(age_range: str):
    return (float(age_range.split('-')[1])+float(age_range.split('-')[0]))/2


def resize(phantom, shape):
    resize = Resize(max(shape), size_mode='longest')
    resized = resize(phantom[None])[0]
    return resized


def voxelize_ground_truth(dicom_path: str | Path, phantom_path: str | Path,
                          material_threshold_dict: dict | None = None):
    '''
    Used to convert ground truth image into segmented volumes used by XCIST to
    run simulations

    :param dicom_path: str | Path, path where the DICOM images are located,
        these are typically the output of `convert_to_dicom`
    :param phantom_path: str or Path, where the phantom files are to be
        written
    :param material_threshold_dict: dictionary mapping XCIST materials to
        appropriate lower thresholds in the ground truth image, see the .cfg
        here for examples
        <https://github.com/xcist/phantoms-voxelized/tree/main/DICOM_to_voxelized>
    '''
    nfiles = len(list(Path(dicom_path).rglob('*.dcm')))
    slice_range = list(range(nfiles))
    if not material_threshold_dict:
        material_threshold_dict = dict(zip(
                                        ['ncat_adipose',
                                         'ncat_water',
                                         'ncat_brain',
                                         'ncat_skull'],
                                        [-200, -10, 10, 300]))

    cfg_file_str = f"""
# Path where the DICOM images are located:
phantom.dicom_path = '{dicom_path}'
# Path where the phantom files are to be written
# (the last folder name will be the phantom files' base name):
phantom.phantom_path = '{phantom_path}'
phantom.materials = {list(material_threshold_dict.keys())}
phantom.mu_energy = 60
phantom.thresholds = {list(material_threshold_dict.values())}
phantom.slice_range = [{[slice_range[0], slice_range[-1]]}] # Range of DICOM
# image numbers to include. (first, last slice)
phantom.show_phantom = False  # Flag to turn on/off image display.
phantom.overwrite = True  # Flag to overwrite existing files without warning.
"""

    dicom_to_voxel_cfg = phantom_path / 'dicom_to_voxelized.cfg'

    with open(dicom_to_voxel_cfg, 'w') as f:
        f.write(cfg_file_str)

    dicom_to_voxelized_phantom.run_from_config(dicom_to_voxel_cfg)


def load_phantom(age=38, shape=None, name='default'):
    '''
    Loads appropriate phantom based on age as a keyword

    :param age: patient age in years, MIDA currently hard coded at 38 yrs
    :param shape: shape of that the ground truth phantom will be interpolated
    :param name: patient name to be saved in DICOM header
    :param lesion_type: options include: ['IPH', 'EDH', 'SDH']
    :param radius: lesion radius if sphere is selected
    :param intensity: uniform intensity of the lesion (HU)
    :param add_positioning_augmentation: bool, apply random affine to phantom
    '''
    load_dotenv()
    if 'PHANTOM_DIRECTORY' in os.environ:
        phantom_dir = Path(os.environ['PHANTOM_DIRECTORY'])
    else:
        phantom_dir = Path(__file__).parents[2]
        warn(f'''
The environment variable `PHANTOM_DIRECTORY` has not been set, this is needed
to locate stored base phantom files for the NIHPD and MIDA head phantoms.

If these phantom files cannot be located, NIHPD phantoms will be downloaded to
your working directory: {phantom_dir}

MIDA phantom files need to be downloaded manually and added to this directory,
see `MIDA_Head_Phantom` for details.

Please do one of the following:

1. create a file called `.env` in this project's working directory and add:

`PHANTOM_DIRECTORY=/path/to/phantoms`

or

2. in your terminal `export PHANTOM_DIRECTORY=/path/to_phantoms`
''')

    mida_age = 38
    if age == mida_age:
        phantom = MIDA_Head(phantom_dir / 'MIDA_Head_Phantom', shape=shape)
    else:
        phantom = NIHPD_Head(phantom_dir / 'NIHPD_Head_Phantom',
                             age=age, shape=shape)

    phantom.patient_name = name
    phantom.age = age
    return phantom


class Phantom:
    '''
    A base phantom that accepts any image array and spacings to define its size.

    :param img: numpy.ndarray, 2D or 3D, defining the phantom
    :param spacings: tuple, voxel spacings [mm] (z, x, y). Definition of voxel sizes.
                    Default is 1 mm in each direction.
    :param patient_name: str, patient identifier to be saved in DICOM header. Default is 'default'.
    :param patientid: int, patient identifier to be saved in DICOM header. Default is 0.
    :param age: float, patient age in years to be saved in DICOM header. Default is 0.
    '''
    def __init__(self, img: np.ndarray, spacings: tuple = (1, 1, 1),
                 patient_name: str = 'default', patientid: int = 0, age: float = 0) -> None:
        self._phantom = img
        self.dz, self.dx, self.dy = spacings
        self.nz, self.nx, self.ny = self._phantom.shape
        self.patient_name = patient_name
        self.patientid = patientid
        self.age = age

    def __repr__(self) -> str:
        string_representation = f'''
        Phantom Class: {self.__class__.__name__}
        Age (years): {self.age}
        Shape (voxels): {self.shape}
        Size (mm): {self.size}
        '''
        return string_representation

    def get_CT_number_phantom(self) -> np.ndarray:
        '''Returns the phantom array'''
        return self._phantom

    @property
    def spacings(self) -> tuple:
        '''Returns the voxel spacings (z, x, y)'''
        return self.dz, self.dx, self.dy

    @property
    def shape(self) -> list:
        '''Returns the shape of the phantom array'''
        return list(self._phantom.shape)

    @property
    def size(self) -> np.ndarray:
        '''Returns the size of the phantom array (mm)'''
        return np.array(self.spacings) * self.shape

    def resize(self, shape: tuple) -> None:
        '''
        Resizes the phantom array to the given shape and adjusts the spacings accordingly.

        :param shape: tuple, new shape for the phantom array
        '''
        original_shape = np.array(self.shape)
        self._phantom = resize(self._phantom, shape)
        new_shape = np.array(self._phantom.shape)
        new_spacings = original_shape / new_shape * np.array(self.spacings)
        self.dz, self.dx, self.dy = new_spacings
        self.nz, self.nx, self.ny = shape


class LesionPhantom(Phantom):
    '''
    A Phantom object with methods for inserting lesions.
    '''
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
        self.exclusion_mask = np.zeros(self.shape, dtype=bool)

    def resize(self, shape: tuple) -> None:
        '''
        Resizes the phantom array to the given shape and adjusts the spacings accordingly.

        :param shape: tuple, new shape for the phantom array
        '''
        super().resize(shape)
        self.exclusion_mask = resize(self.exclusion_mask, shape).astype(bool)

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

    def insert_lesion(self, lesion_type, volume=5, intensity=50, mass_effect=False, seed=None, **kwargs):
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
                self.add_round_lesion(volume=volume, intensity=intensity, mass_effect=mass_effect, seed=seed, **kwargs)
        elif lesion_type in ['EDH', 'SDH']:
            img_w_lesion, lesion_image, lesion_coords = \
                self._add_dural_lesion(volume, lesion_type, intensity, mass_effect=mass_effect, seed=seed)
        else:
            raise ValueError(f'unknown lesion type passed: {lesion_type}. '
                             'Currently accepts IPH (intraparenchymal), EDH (epidural), or SDH (subdural).')
        self._phantom = img_w_lesion
        self._lesion.append(lesion_image)
        self._lesion_coords.append(lesion_coords)
        self.lesion_intensity.append(intensity)
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
                         seed: int | None = None) -> tuple:
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
        if not valid_points.any():
            raise RuntimeError(f'Requested volume: {volume} mL too \
large, try smaller volume')
        # lower distance threshold `r` to allow overlap
        z, x, y = np.argwhere(valid_points)[rng.integers(0,
                                            valid_points.sum())]

        lesion_vol = np.full(img.shape, fill_value=-1000)
        transform = RandAffine(prob=1, translate_range=[r, r])
        transform.set_random_state(seed)

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
        lesion_mask = lesion_mask & ~self.exclusion_mask
        img_w_lesion = np.copy(img)
        img_w_lesion[lesion_mask] = lesion_vol[lesion_mask]

        if mass_effect:
            if mass_effect is True:
                mass_effect = 0.5
            warped = insert_with_mass_effect(img,
                                             lesion_mask,
                                             self.exclusion_mask,
                                             strength=mass_effect)
            warped[lesion_mask] = img_w_lesion[lesion_mask]
            img_w_lesion[lesion_mask.sum(axis=(1, 2)) > 0] =\
                warped[lesion_mask.sum(axis=(1, 2)) > 0]

        return img_w_lesion, lesion_mask, (z, x, y)

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


mida_age = 38  # add 38 as the median US adult age to represent MIDA, consider
#  other identifiers when adding more patients


class HeadPhantom(LesionPhantom):
    def __init__(self, phantom_dir, shape=None):
        self.materials = {
            'csf': 10,
            'gray matter': 40,
            'white matter': 30,
            'air': -1000,
            'CSF': 10,
            'skull': 900
            }
        self.patientid = 0
        self._lesion = []
        self._lesion_coords = []
        self.lesion_type = []
        self.lesion_intensity = []  # HU
        self.mass_effect = False
        phantom, spacings = self.load_phantom(Path(phantom_dir))
        super().__init__(phantom, spacings)
        if shape:
            self.resize(shape)
        self.exclusion_mask = self.get_skull_map().astype(bool)

    def load_phantom(self, phantom_dir) -> tuple[np.ndarray, tuple[int, int, int]]:
        'returns phantom and spacings'
        pass

    def get_material_mask(self, material):
        pass

    def get_dura_map(self):
        'used for EDH, SDH lesion insertion'
        pass

    def get_skull_map(self):
        'used for lesion insertion mass effect warping'
        pass

    def get_lesion_mask(self):
        return self._lesion[0]


class MIDA_Head(HeadPhantom):
    def __init__(self, phantom_dir, shape=None):
        if not phantom_dir.exists():
            raise FileNotFoundError(f'''
MIDA head phantom files not found in {phantom_dir}

To use MIDA head phantoms, please download them from:
 <https://cdrh-rst.fda.gov/mida-multimodal-imaging-based-model-human-head-and-neck>

and place in your `PHANTOM_DIRECTORY`, see `load_phantom` for more details
''')
        super().__init__(phantom_dir, shape)
        self.material_lut = self._load_material_LUT()

    def load_phantom(self, phantom_dir):
        'returns phantom and spacings'
        img = nib.load(phantom_dir/'MIDA_v1.nii')
        phantom = np.array(img.get_fdata()).transpose(1, 0, 2)[::-1]
        header = img.header
        dx, dy, dz = header['pixdim'][1:4]
        spacings = dz, dx, dy
        return phantom, spacings

    def _load_material_LUT(self):
        return pd.read_csv(os.path.join(Path(__file__).parent.resolve(),
                                        'MIDA_v1.csv'))

    def get_CT_number_phantom(self):
        if len(self._lesion_coords) > 0:
            return self._phantom
        phantom = self._phantom
        material_lut = self.material_lut
        HU_phantom = np.copy(phantom)
        for _, row in material_lut[~material_lut['HU'].isna()].iterrows():
            if row['HU'] == 8888:
                HU_phantom[phantom == row['MIDA_ID']] = self.materials['white matter']
            elif row['HU'] == 9999:
                HU_phantom[phantom == row['MIDA_ID']] = self.materials['gray matter']
            else:
                HU_phantom[phantom == row['MIDA_ID']] = row['HU']
        return HU_phantom

    def get_material_mask(self, material):
        if material not in self.materials:
            raise ValueError(f'{material} not in {self.materials.keys()}')
        return self.get_CT_number_phantom() == self.materials[material]

    def get_dura_map(self):
        '''obtains dura map using mida atlas index of 1.0'''
        dura_map = np.zeros_like(self._phantom)
        dura_map[np.where(self._phantom == 1.0)] = 1.0
        return dura_map

    def get_skull_map(self):
        '''obtains partial skull map using mida atlas,
         ignoring facial bones (for now)'''
        skull_map = np.zeros_like(self._phantom)
        skull_map[np.where(self._phantom == 52)] = 1.0
        skull_map[np.where(self._phantom == 53)] = 1.0  # skull outer table
        skull_map[np.where(self._phantom == 54)] = 1.0  # skull outer table
        skull_map[np.where(self._phantom == 40)] = 1.0  # skull/facial bone
        skull_map[np.where(self._phantom == 1000)] = 1.0  # other bone voxels
        return skull_map


url = 'https://www.bic.mni.mcgill.ca/~vfonov/nihpd/obj1_analyze.zip'
nihpd_ages = [6.5, 9.0, 10.5, 11.5, 12.0, 15.75]
possible_ages = nihpd_ages + [mida_age]


class NIHPD_Head(HeadPhantom):
    '''
    loads MR brain atlas of mean `age`, downloaded from
    <https://www.bic.mni.mcgill.ca/~vfonov/nihpd/obj1> and saved in `phantom_dir`

    :param phantom_dir: str, directory holding .nii files from
        https://www.bic.mni.mcgill.ca/~vfonov/nihpd/obj1
    :param age: float, mean age of the atlas phantom to load. Note the brain
        atlases are stored as age ranges, thus the mean age is the mid point
        of the range (upper+lower)/2
    :param symmetry: optional, the atlases are provided in their natural
        asymmetric or artificially generated symmetric state, default is
        asymmetric, see article for more details:
    1. Fonov V, Evans AC, Botteron K, Almli CR, McKinstry RC, Collins DL.
        Unbiased average age-appropriate atlases for pediatric studies.
        NeuroImage. 2011;54(1):313-327. doi:10.1016/j.neuroimage.2010.07.033
    '''
    def __init__(self, phantom_dir, age: float, symmetric=False, shape=None):
        phantom_dir = Path(phantom_dir)
        self.age = age
        self.symmetric = symmetric
        if not phantom_dir.exists():
            print(f'''
`PHANTOM_DIRECTORY` {phantom_dir} not found, now downloading NIHPD phantoms
from {url}

If you have already downloaded NIHPD and MIDA head phantoms, please see
`load_phantom` for details on how to add their locations.
''')
            download_and_extract_archive(url, phantom_dir)
        super().__init__(phantom_dir, shape)
        self.materials['skull'] = 900

    def load_phantom(self, phantom_dir, shape=None):
        'sets ._phantom, and .dx, .dy, .dz, .nx, .ny, .nz'
        nii_files = list(phantom_dir.glob('*.nii'))
        age = self.age
        age_ranges = [o.stem.split('_')[2] for o in nii_files]
        ages = {get_mean_age(o): o for o in age_ranges}

        if age not in ages:
            raise ValueError(f'age {age} not in {sorted(ages.keys())}\
from {phantom_dir}')
        age_range = ages[age]
        symmetry = 'sym' if self.symmetric else 'asym'
        nib_img = nib.load(phantom_dir / f'nihpd_{symmetry}_{age_range}_pdw.nii')
        self.csf = nib.load(
            phantom_dir / f'nihpd_{symmetry}_{age_range}_csf.nii'
            ).get_fdata().transpose(2, 1, 0)[:, ::-1, :]
        self.gm = nib.load(
            phantom_dir / f'nihpd_{symmetry}_{age_range}_gm.nii'
            ).get_fdata().transpose(2, 1, 0)[:, ::-1, :]
        self.wm = nib.load(
            phantom_dir / f'nihpd_{symmetry}_{age_range}_wm.nii'
            ).get_fdata().transpose(2, 1, 0)[:, ::-1, :]
        self.mask = nib.load(
            phantom_dir / f'nihpd_{symmetry}_{age_range}_mask.nii'
            ).get_fdata().transpose(2, 1, 0)[:, ::-1, :]
        self.pdw = nib.load(
            phantom_dir / f'nihpd_{symmetry}_{age_range}_pdw.nii'
            ).get_fdata().transpose(2, 1, 0)[:, ::-1, :]

        self.csf = self.csf[::-1]
        self.gm = self.gm[::-1]
        self.wm = self.wm[::-1]
        self.mask = self.mask[::-1]
        self.pdw = self.pdw[::-1]

        skull = (self.mask == 0)*self.pdw / self.pdw.max()
        skull[skull < 0.1] = 0
        skull[skull > 1] = 1
        self.skull = skull
        header = nib_img.header
        self.scalp_dict = self.set_scalp_dict()
        dx, dy, dz = header['pixdim'][1:4]
        self.dx = dx
        phantom = self.get_CT_number_phantom()
        dx, dy, dz = header['pixdim'][1:4]
        spacings = dz, dx, dy
        return phantom, spacings

    def resize(self, shape=None):
        original_shape = self.csf.shape
        self.csf = resize(self.csf, shape).numpy()
        new_shape = self.csf.shape
        self.gm = resize(self.gm, shape).numpy()
        self.wm = resize(self.wm, shape).numpy()
        self.mask = resize(self.mask, shape).numpy()
        self.pdw = resize(self.pdw, shape).numpy()
        self.skull = resize(self.skull, shape).numpy()

        new_spacings = np.array(original_shape) / np.array(new_shape) *\
            [self.dz, self.dx, self.dy]
        self.dz, self.dx, self.dy = new_spacings
        self.nz, self.nx, self.ny = shape

    def set_scalp_dict(self):
        params = OrderedDict()
        params['skin'] = dict(
                thickness=2,
                HU=0)
        params['fat'] = dict(
                thickness=3,
                HU=-30
            )
        params['muscle'] = dict(
                thickness=3,
                HU=20
            )
        return params

    def add_scalp(self, vol):
        """
        adds skin, fat, and muscle layers to the head `vol`
        """
        binary = vol > 0
        erosion = binary.copy()
        params = self.scalp_dict
        for name, param in params.items():
            thickness = param['thickness'] / self.dx
            t = max(int(thickness), 3)  # mm
            struct = np.ones(binary.ndim*[t])
            new_erosion = binary_erosion(erosion, struct)
            mask = new_erosion ^ erosion
            erosion = new_erosion
            vol[mask] = param['HU']
        return vol

    def get_sutures(self, thickness=2, thresh=30):
        """
        returns suture mask to the self skull

        :param thickness: thickness in pixels of the suture
        :returns: boolean suture mask that can be used to set skull suture
            values
        """
        src_dir = Path(__file__).parents[1]
        data = nrrd.read(src_dir / 'annotations/suture/NIHPD_Head_Phantom/labelmap.nrrd')[0].transpose(2, 1, 0)[::-1, ::-1]
        skull = self.get_skull_map().astype(bool)
        dx, dy, dz = np.array(skull.shape) - np.array(data.shape)
        if (dx < 0) | (dy < 0) | (dz < 0):
            resizewithcrop = ResizeWithPadOrCrop(spatial_size=skull.shape)
            data = resizewithcrop(data[None])[0].numpy()
            dx, dy, dz = np.array(skull.shape) - np.array(data.shape)

        dx1 = dx2 = dx//2
        if dx % 2 == 1:
            dx2 += 1
        dy1 = dy2 = dy//2
        if dy % 2 == 1:
            dy2 += 1
        dz1 = dz2 = dz//2
        if dz % 2 == 1:
            dz2 += 1
        data = np.pad(data, ((dx1, dx2), (dy1, dy2), (dz1, dz2))) > 0
        suture_dist = distance_transform_edt(~data)
        sutures = skull & (suture_dist < thresh)
        sutures = ski.morphology.skeletonize(sutures)
        sutures = ski.morphology.dilation(sutures, np.ones(3*[thickness]))
        return sutures

    def get_CT_number_phantom(self, add_scalp=True, add_sutures=True):
        if len(self._lesion_coords) > 0:
            return self._phantom
        phantom = self.csf*self.materials['csf'] + self.gm*self.materials['gray matter'] +\
            self.wm*self.materials['white matter'] + self.skull*self.materials['skull']
        phantom[phantom <= 0] = self.materials['air']
        phantom[self.get_dura_map()] = 50  # HU same as MIDA
        if add_scalp:
            phantom = self.add_scalp(phantom)
        if add_sutures:
            sutures = self.get_sutures()
            phantom[sutures] = 0  # assume water HU
        return phantom

    def get_material_mask(self, material):
        if material not in self.materials:
            raise ValueError(f'{material} not in {self.materials.keys()}')
        if material == 'white matter':
            mask = self.wm > 0.3

        if material == 'gray matter':
            mask = self.gm > 0.3

        if material == 'CSF':
            mask = self.csf > 0.3
        return mask.astype(int)

    def get_dura_map(self):
        '''obtains approximate dura map using inside boundary of brain mask'''
        return ski.segmentation.find_boundaries(self.mask,
                                                mode='inner',
                                                background=0)

    def get_skull_map(self):
        '''obtains rudimentary mask of skull voxels using threshold of
        proton-density weighted image and full mask'''
        skull_map = (self.mask == 0)*self.pdw / self.pdw.max()
        skull_map[skull_map < 0.1] = 0
        skull_map[skull_map > 0] = 1
        return skull_map
