'''
module for working with phantoms
'''

from pathlib import Path
import os
from warnings import warn

import numpy as np
import nibabel as nib
import pandas as pd
import skimage as ski
from dotenv import load_dotenv
from monai.transforms import Resize, RandAffine, Affine
from torchvision.datasets.utils import download_and_extract_archive

from . import dicom_to_voxelized_phantom
from ..artifact_generation import transform_image_label_pair

from ..lesion_definition import (elliptical_lesion,
                                 insert_dural_3D,
                                 warp_slice,
                                 get_perimeter)
from scipy.ndimage import (center_of_mass,
                           distance_transform_edt,
                           binary_dilation)


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
                                 src_coords, dst_coords)
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


def load_phantom(age=38, shape=(480, 480, 350), name='default'):
    '''
    Loads appropriate phantom based on age as a keyword

    :param age: patient age in years, MIDA currently hard coded at 38 yrs
    :param shape: shape of that the ground truth phantom will be interpolated
    :param name: patient name to be saved in DICOM header
    :param lesion_type: options include: ['round', 'epidural', 'subdural']
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
    Base phantom that can accept any img array and spacings which
        specify the size

    :param img: 2D or 3D numpy array defining the phantom
    :param spacings: tuple, voxel spacings [mm] (z, x, y), defining voxel
        sizes, defaults to 1 mm in each direction
    :param patient_name: patient identifier to be saved in DICOM header
    :param patientid: int, patient identifier to be saved in DICOM header
    :param age: float, in years to be saved in DICOM header
    '''
    def __init__(self, img: np.ndarray, spacings: tuple = (1, 1, 1),
                 patient_name='default', patientid=0, age=0) -> None:
        self._phantom = img
        self.dz, self.dx, self.dy = spacings
        self.patient_name = patient_name
        self.patientid = patientid
        self.age = age

    def __repr__(self) -> str:
        repr = f'''
        phantom class: {self.__class__.__name__}
        age [yrs]: {self.age}
        shape [voxels]: {self.shape}
        size [mm]: {self.size}
        '''
        return repr

    def get_CT_number_phantom(self) -> np.ndarray:
        return self._phantom

    @property
    def spacings(self):
        return self.dz, self.dx, self.dy

    @property
    def shape(self):
        return list(self._phantom.shape)

    @property
    def size(self):
        return np.array(self.spacings)*self.shape


class HeadPhantom(Phantom):
    def __init__(self, phantom_dir):
        self.phantom_dir = phantom_dir
        self.materials = {'gray matter': 40,
                          'white matter': 30,
                          'CSF': 10,
                          'skull': 1000}
        self.patientid = 0
        self._lesion = []
        self._lesion_coords = []
        self.lesion_type = []
        self.lesion_intensity = []  # HU
        self.mass_effect = False

    def get_material_mask(self, material):
        pass

    def get_dura_map(self):
        'used for epidural, subdural lesion insertion'
        pass

    def get_skull_map(self):
        'used for lesion insertion mass effect warping'
        pass

    def get_lesion_mask(self):
        return self._lesion[0]

    def __repr__(self) -> str:
        repr = super().__repr__() + f'''
        Number of lesions: {len(self._lesion_coords)}
        Lesion locations [voxel index (z, x, y)]: {self._lesion_coords}
        Mass effect: {self.mass_effect}
        '''
        return repr

    @property
    def spacings(self):
        return self.dz, self.dx, self.dy

    def insert_lesion(self, lesion_type, volume=5, intensity=50,
                      mass_effect=False, seed=None, **kwargs):
        '''
        inserts lesion of `lesion_type` into phantom array

        :param lesion_type: str, options include
            ['round', 'epidural', 'subdural'],
            see associated methods `add_round_lesion`, `_add_dural_lesion`
        :param volume: in mL, volume of the lesion
        :param intensity: lesion CT number in HU
        :param mass_effect: optional, bool whether to apply mass effect
            processing to displace brain tissue following lesion insertion
        :param edema: optional, bool or int. whether to add a ring of low
            contrast, 10 HU, edema around the lesion, currently only
            implemented for sphere
        :param seed: optional, int specify seed for reproducible lesion
            insertion, otherwise random

        return img_w_lesion, lesion_image, lesion_coords
        '''
        if volume <= 0:
            return self
        self.lesion_type.append(lesion_type)
        self.mass_effect = mass_effect
        if lesion_type == 'round':
            img_w_lesion, lesion_image, lesion_coords =\
                self.add_round_lesion(volume=volume,
                                      intensity=intensity,
                                      mass_effect=mass_effect,
                                      seed=seed,
                                      **kwargs)
        elif lesion_type in ['epidural', 'subdural']:
            img_w_lesion, lesion_image, lesion_coords =\
                self._add_dural_lesion(volume, lesion_type, intensity,
                                       mass_effect=mass_effect,
                                       seed=seed,
                                       **kwargs)
        else:
            raise ValueError(f'unknown lesion type passed: {lesion_type}\
                             currently accepts round, epidural, or subdural')
        self._phantom = img_w_lesion
        self._lesion.append(lesion_image)
        self._lesion_coords.append(lesion_coords)
        self.lesion_intensity.append(intensity)
        return self

    def apply_transform(self, transform: RandAffine | Affine, seed=None):
        if not self._lesion:
            self._phantom = transform(self.get_CT_number_phantom())
            return
        self._phantom, self._lesion[0] =\
            transform_image_label_pair(transform,
                                       self.get_CT_number_phantom(),
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
        :param seed: optional, defaults to None, set seed for reproducible
            lesion insertion

        :returns: img_w_lesion, lesion_vol, (z, x, y)
        '''
        rng = np.random.default_rng(seed)

        voxel_size = np.power(self.dx*self.dy*self.dz, 1/3)
        r = sphere_radius_from_volume(volume) / voxel_size
        img = self.get_CT_number_phantom()
        mask = self.get_material_mask(material).astype(int)

        lesion_vol = np.zeros_like(img)
        valid_points = distance_transform_edt(mask) > (r * 0.9)
        if not valid_points.any():
            raise RuntimeError(f'Requested volume: {volume} mL too\
                                large, try smaller volume')
        # lower distance threshold `r` to allow overlap
        z, x, y = np.argwhere(valid_points)[rng.integers(0,
                                            valid_points.sum())]

        lesion_vol = np.full(img.shape, fill_value=-1000)
        transform = RandAffine(prob=1, translate_range=[r, r])
        transform.set_random_state(seed)

        for _ in range(complexity):
            lesion_seed = rng.integers(0, 1e6)
            axes = get_semi_major_axes(eccentricity, lesion_seed)
            foci = r * axes
            sphere = elliptical_lesion(img.shape, center=(z, x, y),
                                       radius=foci,
                                       random_rotate=lesion_seed)
            sphere = transform(sphere).astype(bool)
            lesion_vol[sphere] = intensity
        lesion_mask = lesion_vol > -1000

        if edema:
            edema_pixels = 5
            edema_HU = 10
            edema = edema_pixels if edema is True else edema
            edema_mask = binary_dilation(lesion_mask,
                                         np.ones(3*[edema])) ^ lesion_mask
            lesion_vol[edema_mask] = edema_HU
            lesion_mask = lesion_vol > -1000
        img_w_lesion = img.copy()
        img_w_lesion[lesion_mask] = lesion_vol[lesion_mask]

        if mass_effect:
            if mass_effect is True:
                mass_effect = 0.5
            warped = insert_with_mass_effect(img,
                                             lesion_mask,
                                             self.get_skull_map(),
                                             strength=mass_effect)
            warped[lesion_mask] = img_w_lesion[lesion_mask]
            img_w_lesion[lesion_mask.sum(axis=(1, 2)) > 0] =\
                warped[lesion_mask.sum(axis=(1, 2)) > 0]

        return img_w_lesion, lesion_mask, (z, x, y)

    def _add_dural_lesion(self, volume, lesion_type, intensity,
                          seed=None, mass_effect=True):

        HU_volume = self.get_CT_number_phantom()
        lesion_vol, HU_volume = insert_dural_3D(phantom=self,
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


class MIDA_Head(HeadPhantom):
    def __init__(self, phantom_dir, csf_HU=10, gm_HU=40, wm_HU=30,
                 skull_HU=1000, shape=None):
        if not phantom_dir.exists():
            raise FileNotFoundError(f'''
MIDA head phantom files not found in {phantom_dir}

To use MIDA head phantoms, please download them from:
 <https://cdrh-rst.fda.gov/mida-multimodal-imaging-based-model-human-head-and-neck>

and place in your `PHANTOM_DIRECTORY`, see `load_phantom` for more details
''')
        super().__init__(phantom_dir)
        self.age = 38  # median american age
        self.csf_HU = csf_HU
        self.gm_HU = gm_HU
        self.wm_HU = wm_HU
        self.skull_HU = skull_HU
        self.air_HU = -1000
        self.material_lut = self._load_material_LUT()

        img = nib.load(self.phantom_dir/'MIDA_v1.nii')
        self._phantom = np.array(img.get_fdata()).transpose(1, 0, 2)[::-1]

        header = img.header
        self.dx, self.dy, self.dz = header['pixdim'][1:4]

        original_shape = self._phantom.shape
        if shape:
            self._phantom = resize(self._phantom, shape)
            new_shape = self._phantom.shape
            new_spacings = np.array(original_shape) / np.array(new_shape) *\
                [self.dz, self.dx, self.dy]
            self.dz, self.dx, self.dy = new_spacings
        else:
            shape = original_shape
        self.nz, self.nx, self.ny = shape

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
                HU_phantom[phantom == row['MIDA_ID']] = self.wm_HU
            elif row['HU'] == 9999:
                HU_phantom[phantom == row['MIDA_ID']] = self.gm_HU
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


class NIHPD_Head(HeadPhantom):
    '''
    loads MR brain atlas of mean `age`, downloaded from
    <https://www.bic.mni.mcgill.ca/~vfonov/nihpd/obj1> and saved in `base_dir`

    :param base_dir: str, directory holding .nii files from
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
    def __init__(self, phantom_dir, age: float, symmetric=False, csf_HU=10,
                 gm_HU=40, wm_HU=30, skull_HU=1000, shape=None):
        phantom_dir = Path(phantom_dir)
        if not phantom_dir.exists():
            url = 'https://www.bic.mni.mcgill.ca/~vfonov/nihpd/obj1_analyze.zip'
            print(f'''
`PHANTOM_DIRECTORY` {phantom_dir} not found, now downloading NIHPD phantoms
from {url}

If you have already downloaded NIHPD and MIDA head phantoms, please see
`load_phantom` for details on how to add their locations.
''')
            download_and_extract_archive(url, phantom_dir)
        super().__init__(phantom_dir)
        self.age = age
        self.csf_HU = csf_HU
        self.gm_HU = gm_HU
        self.wm_HU = wm_HU
        self.skull_HU = skull_HU
        self.air_HU = -1000

        nii_files = list(self.phantom_dir.glob('*.nii'))
        age_ranges = [o.stem.split('_')[2] for o in nii_files]
        ages = {get_mean_age(o): o for o in age_ranges}

        if age not in ages:
            raise ValueError(f'age {age} not in {sorted(ages.keys())} from {self.phantom_dir}')
        age_range = ages[age]

        base_dir = self.phantom_dir
        symmetry = 'sym' if symmetric else 'asym'

        nib_img = nib.load(base_dir / f'nihpd_{symmetry}_{age_range}_pdw.nii')
        header = nib_img.header
        self.dx, self.dy, self.dz = header['pixdim'][1:4]

        self.csf = nib.load(
            base_dir / f'nihpd_{symmetry}_{age_range}_csf.nii'
            ).get_fdata().transpose(2, 1, 0)[:, ::-1, :]
        self.gm = nib.load(
            base_dir / f'nihpd_{symmetry}_{age_range}_gm.nii'
            ).get_fdata().transpose(2, 1, 0)[:, ::-1, :]
        self.wm = nib.load(
            base_dir / f'nihpd_{symmetry}_{age_range}_wm.nii'
            ).get_fdata().transpose(2, 1, 0)[:, ::-1, :]
        self.mask = nib.load(
            base_dir / f'nihpd_{symmetry}_{age_range}_mask.nii'
            ).get_fdata().transpose(2, 1, 0)[:, ::-1, :]
        self.pdw = nib.load(
            base_dir / f'nihpd_{symmetry}_{age_range}_pdw.nii'
            ).get_fdata().transpose(2, 1, 0)[:, ::-1, :]

        self.csf = self.csf[::-1]
        self.gm = self.gm[::-1]
        self.wm = self.wm[::-1]
        self.mask = self.mask[::-1]
        self.pdw = self.pdw[::-1]

        original_shape = self.csf.shape
        if shape:
            self.csf = resize(self.csf, shape).numpy()
            new_shape = self.csf.shape
            self.gm = resize(self.gm, shape).numpy()
            self.wm = resize(self.wm, shape).numpy()
            self.mask = resize(self.mask, shape).numpy()
            self.pdw = resize(self.pdw, shape).numpy()

            new_spacings = np.array(original_shape) / np.array(new_shape) *\
                [self.dz, self.dx, self.dy]
            self.dz, self.dx, self.dy = new_spacings
        else:
            shape = original_shape
        self.nz, self.nx, self.ny = shape

        skull = (self.mask == 0)*self.pdw / self.pdw.max()
        skull[skull < 0.1] = 0
        skull[skull > 1] = 1
        self.skull = skull
        self._phantom = self.get_CT_number_phantom()

    def get_CT_number_phantom(self):
        if len(self._lesion_coords) > 0:
            return self._phantom
        phantom = self.csf*self.csf_HU + self.gm*self.gm_HU +\
            self.wm*self.wm_HU + self.skull*self.skull_HU
        phantom[phantom <= 0] = self.air_HU
        phantom[self.get_dura_map()] = 50  # HU same as MIDA
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
