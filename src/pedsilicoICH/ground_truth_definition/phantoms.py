'''
module for working with phantoms
'''

from pathlib import Path
from io import StringIO

import numpy as np
import nibabel as nib
import pandas as pd
import skimage as ski
from monai.transforms import Resize, RandAffine, Affine
from torchvision.datasets.utils import download_and_extract_archive

from . import dicom_to_voxelized_phantom
from ..artifact_generation import transform_image_label_pair

from ..lesion_definition import spherical_lesion, insert_dural_3D
from scipy.ndimage import center_of_mass


def sphere_radius_from_volume(volume):
    return np.power(3/4/np.pi*volume, 1/3)


def get_mean_age(age_range: str):
    return (float(age_range.split('-')[1])+float(age_range.split('-')[0]))/2


def resize(phantom, shape):
    resize = Resize(max(shape), size_mode='longest')
    resized = resize(phantom[None])[0]
    return resized


def voxelize_ground_truth(dicom_path, phantom_path, material_threshold_dict=None):
    """
    Used to convert ground truth image into segmented volumes used by XCIST to run simulations

    Inputs:
    dicom_path    (string)           Path where the DICOM images are located.
    phantom_path  (string)           Path where the phantom files are to be written
    dicom_path [str]: directory containing ground truth dicom images, these are typically the output of `convert_to_dicom`
    material_threshold_dict [dict]: dictionary mapping XCIST materials to appropriate lower thresholds in the ground truth image, see the .cfg here for examples <https://github.com/xcist/phantoms-voxelized/tree/main/DICOM_to_voxelized>
    """
    nfiles = len(list(Path(dicom_path).rglob('*.dcm')))
    slice_range = list(range(nfiles))
    if not material_threshold_dict:
        material_threshold_dict = dict(zip(
                                        ['ICRU_lung_adult_healthy',
                                         'ICRU_adipose_adult2',
                                         'ICRU_liver_adult',
                                         'water',
                                         'ICRU_skeleton_cortical_bone_adult'],
                                        [-1000, -200, 0, 100, 300]))

    cfg_file_str = f"""
# Path where the DICOM images are located:
phantom.dicom_path = '{dicom_path}'
# Path where the phantom files are to be written (the last folder name will be the phantom files' base name):
phantom.phantom_path = '{phantom_path}'
phantom.materials = {list(material_threshold_dict.keys())}
phantom.mu_energy = 60                  # Energy (keV) at which mu is to be calculated for all materials.
phantom.thresholds = {list(material_threshold_dict.values())}	# Lower threshold (HU) for each material.
phantom.slice_range = [{[slice_range[0], slice_range[-1]]}]			  # Range of DICOM image numbers to include. (first, last slice)
phantom.show_phantom = False                # Flag to turn on/off image display.
phantom.overwrite = True                   # Flag to overwrite existing files without warning.
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
    :param lesion_type: options include: ['sphere', 'epidural', 'subdural']
    :param radius: lesion radius if sphere is selected
    :param contrast: uniform contrast of the lesion
    :param add_positioning_augmentation: bool, apply random affine to phantom
    '''
    root_dir = Path(__file__).parents[2]
    nihpd_dir = root_dir.parent / 'NIHPD_Head_Phantom'
    mida_dir = root_dir.parent / 'MIDA_Head_Phantom'

    if not nihpd_dir.exists():
        url = 'https://www.bic.mni.mcgill.ca/~vfonov/nihpd/obj1_analyze.zip'
        download_and_extract_archive(url, nihpd_dir)
    mida_age = 38
    if age == mida_age:
        if not mida_dir.exists():
            Warning(f'MIDA head phantom not found in {mida_dir}, skipping...')
            return None
        phantom = MIDA_Head(mida_dir, shape=shape)
    else:
        phantom = NIHPD_Head(nihpd_dir, age=age, shape=shape)

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
        self._lesion = []
        self._lesion_coords = []
        self.lesion_type = []
        self.lesion_contrast = []

    def get_CT_number_phantom(self) -> np.ndarray:
        return self._phantom

    @property
    def spacings(self):
        return self.dz, self.dx, self.dy


class HeadPhantom(Phantom):
    def __init__(self, phantom_dir):
        self.phantom_dir = phantom_dir
        self.materials = {'gray matter': 45,
                          'white matter': 20,
                          'CSF': 15,
                          'skull': 1000}
        self.patientid = 0
        self._lesion = []
        self._lesion_coords = []
        self.lesion_type = []
        self.lesion_contrast = []

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

    @property
    def spacings(self):
        return self.dz, self.dx, self.dy

    def insert_lesion(self, lesion_type, volume=500, contrast=100,
                      init_slice=None, mass_effect=False, seed=None):
        'return img_w_lesion, lesion_image, lesion_coords'
        self.lesion_type.append(lesion_type)
        if lesion_type == 'sphere':
            img_w_lesion, lesion_image, lesion_coords =\
                self.add_sphere_lesion(volume=volume, contrast=contrast,
                                       mass_effect=mass_effect,
                                       seed=seed)
        elif lesion_type == 'epidural':
            if isinstance(contrast, list):
                contrast = max(contrast)
            img_w_lesion, lesion_image, lesion_coords =\
                self._add_dural_lesion('epidural', contrast,
                                       init_slice, mass_effect=mass_effect,
                                       seed=seed)
        else:
            if isinstance(contrast, list):
                contrast = max(contrast)
            img_w_lesion, lesion_image, lesion_coords =\
                self._add_dural_lesion('subdural', contrast,
                                       init_slice, mass_effect=mass_effect,
                                       seed=seed)

        self._phantom = img_w_lesion
        self._lesion.append(lesion_image)
        self._lesion_coords.append(lesion_coords)
        self.lesion_contrast.append(contrast)
        return self

    def apply_transform(self, transform: RandAffine | Affine, seed=None):
        if not self._lesion:
            self._phantom = transform(self.get_CT_number_phantom())
            return
        self._phantom, self._lesion[0] = transform_image_label_pair(transform,
                                                                    self.get_CT_number_phantom(),
                                                                    self.get_lesion_mask(),
                                                                    seed=seed)

    def add_sphere_lesion(self,
                          volume: list[int] = [200],
                          contrast: list[int] = [-100],
                          material: str = 'white matter',
                          mass_effect: bool = False,  # add mass effect for spherical lesions? These are typically associated with metastases
                          seed: int | None = None,
                          tol: int = 20) -> tuple:
        '''
        adds lesion to img in random location within mask of size radius
        and contrast level contrast

        :param volume: int or list of ints, volume of the sphere lesion,
            if provided a list it will make concentric lesions
        :param contrast: int or list of ints, contrast of the sphere lesion,
            if provided a list it will make concentric lesions of contrasts
        :param material: which material region to insert lesion into,
            self.materials for options
        :param seed: optional, defaults to None, set seed for reproducible lesion insertion

        :returns: img_w_lesion, lesion_vol, (z, x, y)
        '''
        if seed:
            tol = 1
        if not isinstance(volume, list):
            volume = [volume]
        if not isinstance(contrast, list):
            contrast = [contrast]
        radii = [sphere_radius_from_volume(v) for v in volume]
        r = max(radii)

        img = self.get_CT_number_phantom()
        mask = self.get_material_mask(material).astype(int)

        overlap = 0
        counts = 0
        sphere = np.zeros_like(img, dtype=bool)
        while overlap < 0.8:  # can increase threshold to size of lesion
            lesion_vol = np.zeros_like(img)
            rng = np.random.default_rng(seed)
            z, x, y = np.argwhere(mask)[rng.integers(0, mask.sum())]
            if mask[z].sum() < np.pi*r**2:
                continue
            counts += 1
            sphere = spherical_lesion(img, center=(z, x, y),
                                      radius=r).transpose(1, 0, 2)
            overlap = np.sum(mask & sphere)/(np.sum(sphere))
            if counts > tol:
                raise ValueError("Failed to insert lesion into mask")

        lesion_vol = np.zeros_like(img)
        for ri in radii:
            for ci in contrast:
                sphere = spherical_lesion(img, center=(z, x, y),
                                          radius=ri).transpose(1, 0, 2)
                lesion_vol[sphere] += ci
        img_w_lesion = img + lesion_vol
        return img_w_lesion, lesion_vol, (z, x, y)

    def _add_dural_lesion(self, lesion_type, contrast, init_slice=None,
                          seed=None, mass_effect=True):
        rng = np.random.default_rng(seed)
        dura_map = self.get_dura_map()
        volume = self.get_CT_number_phantom()

        init_slice = init_slice or rng.choice(
            np.where(dura_map.mean(axis=(1, 2)) > 0.015)[0])
        lesion_vol, volume = insert_dural_3D(self, init_slice,
                                             lesion_type,
                                             mass_effect=mass_effect,
                                             seed=seed)
        if not isinstance(volume, np.ndarray):
            volume = volume.numpy()

        img_w_lesion = volume.copy()
        img_w_lesion[lesion_vol == 1] = contrast
        z, x, y = center_of_mass(lesion_vol)
        return img_w_lesion, lesion_vol, (int(z), int(x), int(y))


class MIDA_Head(HeadPhantom):
    def __init__(self, phantom_dir, csf_HU=15, gm_HU=45, wm_HU=20,
                 skull_HU=1000, shape=None):
        super().__init__(phantom_dir)
        self.age = 39  # median american age
        self.csf_HU = csf_HU
        self.gm_HU = gm_HU
        self.wm_HU = wm_HU
        self.skull_HU = skull_HU
        self.air_HU = -1000
        self.material_lut = self._load_material_LUT()

        img = nib.load(self.phantom_dir/'MIDA_v1.nii')
        self._phantom = np.array(img.get_fdata()).transpose(1, 0, 2)[::-1]

        original_shape = self._phantom.shape
        if shape:
            self._phantom = resize(self._phantom, shape)
            new_shape = self._phantom.shape
            new_spacings = np.array(original_shape) / np.array(new_shape) * [self.dz, self.dx, self.dy]
            self.dz, self.dx, self.dy = new_spacings
        else:
            shape = original_shape
        self.nz, self.nx, self.ny = shape

    def _load_material_LUT(self):

        with open(self.phantom_dir / 'MIDA_v1.txt', 'rb') as data:
            df = pd.read_csv(StringIO(data.read().decode(errors='replace')), sep='\t', names=['grayscale','c1', 'c2', 'c3', 'material'])
            material_lut = df.iloc[:-8] # read in the material look up table from the top of the txt file
            image_size_info = df.iloc[-8:, :2].set_index('grayscale').T
            self.nx, self.ny, self.nz = int(image_size_info.nx.item()), int(image_size_info.ny.item()), int(image_size_info.nz.item())
            self.dx, self.dy, self.dz = image_size_info.dx.item()*1000, image_size_info.dy.item()*1000, image_size_info.dz.item()*1000

        # BRAIN TISSUE / FLUIDS
        material_lut.loc[material_lut.material == 'Cerebellum Gray Matter', 'grayscale'] = 10
        material_lut.loc[material_lut.material == 'Cerebellum  Gray Matter', 'xcist material'] = 'gray_matter'
        material_lut.loc[material_lut.material == 'Cerebellum  Gray Matter', 'CT Number [HU]'] = self.gm_HU

        material_lut.loc[material_lut.material == 'Brain Gray Matter', 'grayscale'] = 10
        material_lut.loc[material_lut.material == 'Brain Gray Matter', 'xcist material'] = 'gray_matter'
        material_lut.loc[material_lut.material == 'Brain Gray Matter', 'CT Number [HU]'] = self.gm_HU

        material_lut.loc[material_lut.material == 'Thalamus', 'grayscale'] = 116
        material_lut.loc[material_lut.material == 'Thalamus', 'xcist material'] = 'gray_matter'
        material_lut.loc[material_lut.material == 'Thalamus', 'CT Number [HU]'] = self.gm_HU

        material_lut.loc[material_lut.material == 'Cerebellum White Matter', 'grayscale'] = 12
        material_lut.loc[material_lut.material == 'Cerebellum White Matter', 'xcist material'] = 'white_matter'
        material_lut.loc[material_lut.material == 'Cerebellum White Matter', 'CT Number [HU]'] = self.wm_HU

        material_lut.loc[material_lut.material == 'Brain White Matter', 'grayscale'] = 12
        material_lut.loc[material_lut.material == 'Brain White Matter', 'xcist material'] = 'white_matter'
        material_lut.loc[material_lut.material == 'Brain White Matter', 'CT Number [HU]'] = self.wm_HU

        material_lut.loc[material_lut.material == 'CSF Ventricles', 'grayscale'] = 6
        material_lut.loc[material_lut.material == 'CSF Ventricles', 'xcist material'] = 'CSF'
        material_lut.loc[material_lut.material == 'CSF Ventricles', 'CT Number [HU]'] = self.csf_HU

        material_lut.loc[material_lut.material == 'CSF General', 'grayscale'] = 32
        material_lut.loc[material_lut.material == 'CSF General', 'xcist material'] = 'CSF'
        material_lut.loc[material_lut.material == 'CSF General', 'CT Number [HU]'] = self.csf_HU

        # BONE
        material_lut.loc[material_lut.material == 'Skull', 'grayscale'] = 40
        material_lut.loc[material_lut.material == 'Skull', 'xcist material'] = 'ncat_skull'
        material_lut.loc[material_lut.material == 'Skull', 'CT Number [HU]'] = 900

        material_lut.loc[material_lut.material == 'Skull Diplo', 'grayscale'] = 52
        material_lut.loc[material_lut.material == 'Skull Diplo', 'xcist material'] = 'ncat_skull'
        material_lut.loc[material_lut.material == 'Skull Diplo', 'CT Number [HU]'] = 800 # https://en.wikipedia.org/wiki/Diplo%C3%AB

        material_lut.loc[material_lut.material == 'Skull Inner Table', 'grayscale'] = 53
        material_lut.loc[material_lut.material == 'Skull Inner Table', 'xcist material'] = 'ncat_skull'
        material_lut.loc[material_lut.material == 'Skull Inner Table', 'CT Number [HU]'] = 1000

        material_lut.loc[material_lut.material == 'Skull Outer Table', 'grayscale'] = 54
        material_lut.loc[material_lut.material == 'Skull Outer Table', 'xcist material'] = 'ncat_skull'
        material_lut.loc[material_lut.material == 'Skull Outer Table', 'CT Number [HU]'] = 1000

        # OTHER TISSUES
        material_lut.loc[material_lut.material == 'Adipose Tissue', 'grayscale'] = 43
        material_lut.loc[material_lut.material == 'Adipose Tissue', 'CT Number [HU]'] = -120

        material_lut.loc[material_lut.material == 'Epidermis/Dermis', 'grayscale'] = 51
        material_lut.loc[material_lut.material == 'Epidermis/Dermis', 'CT Number [HU]'] = 50

        material_lut.loc[material_lut.material == 'Subcutaneous Adipose Tissue', 'grayscale'] = 62
        material_lut.loc[material_lut.material == 'Subcutaneous Adipose Tissue', 'CT Number [HU]'] = -120

        material_lut.loc[material_lut.material == 'Muscle (General)', 'grayscale'] = 63
        material_lut.loc[material_lut.material == 'Muscle (General)', 'CT Number [HU]'] = 55

        # AIR
        material_lut.loc[material_lut.material == 'Air Internal - Ethmoidal Sinus', 'grayscale'] = 26
        material_lut.loc[material_lut.material == 'Air Internal - Ethmoidal Sinus', 'CT Number [HU]'] = -1000

        material_lut.loc[material_lut.material == 'Air Internal - Frontal Sinus', 'grayscale'] = 27
        material_lut.loc[material_lut.material == 'Air Internal - Frontal Sinus', 'CT Number [HU]'] = -1000

        material_lut.loc[material_lut.material == 'Air Internal - Maxillary Sinus', 'grayscale'] = 28
        material_lut.loc[material_lut.material == 'Air Internal - Maxillary Sinus', 'CT Number [HU]'] = -1000

        material_lut.loc[material_lut.material == 'Air Internal - Sphenoidal Sinus', 'grayscale'] = 29
        material_lut.loc[material_lut.material == 'Air Internal - Sphenoidal Sinus', 'CT Number [HU]'] = -1000
        
        material_lut.loc[material_lut.material == 'Air Internal - Mastoid', 'grayscale'] = 30
        material_lut.loc[material_lut.material == 'Air Internal - Mastoid', 'CT Number [HU]'] = -1000

        material_lut.loc[material_lut.material == 'Air Internal - Nasal/Pharynx', 'grayscale'] = 31
        material_lut.loc[material_lut.material == 'Air Internal - Nasal/Pharynx', 'CT Number [HU]'] = -1000
        
        material_lut.loc[material_lut.material == 'Air Internal - Oral Cavity', 'grayscale'] = 97
        material_lut.loc[material_lut.material == 'Air Internal - Oral Cavity', 'CT Number [HU]'] = -1000


        return material_lut[~material_lut['CT Number [HU]'].isna()]

    def get_CT_number_phantom(self):
        if len(self._lesion_coords) > 0:
            return self._phantom
        phantom = self._phantom
        material_lut = self.material_lut
        phantom[phantom == 50] = -1000  # air
        phantom[phantom == 52] = 800 # the MIDA text file has an unknown character after "Skull Diplo" that makes the above method not work if not removed
        
        for _, row in material_lut[~material_lut['CT Number [HU]'].isna()].iterrows():
            phantom[phantom == row.grayscale] = row['CT Number [HU]']
        return phantom

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
        '''obtains partial skull map using mida atlas, ignoring facial bones (for now)'''
        skull_map = np.zeros_like(self._phantom)
        skull_map[np.where(self._phantom == 53)] = 1.0 # skull outer table
        skull_map[np.where(self._phantom == 40)] = 1.0 # skull/facial bone
        skull_map[np.where(self._phantom == 1000)] = 1.0 # other bone voxels
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
    def __init__(self, phantom_dir, age: float, symmetric=False, csf_HU=15,
                 gm_HU=45, wm_HU=20, skull_HU=1000, shape=None):
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
            raise ValueError(f'age {age} not in {sorted(ages.keys())}')
        age_range = ages[age]

        base_dir = self.phantom_dir
        symmetry = 'sym' if symmetric else 'asym'

        nib_img = nib.load(base_dir / f'nihpd_{symmetry}_{age_range}_pdw.nii')
        header = nib_img.header
        self.dx, self.dy, self.dz = header['pixdim'][1:4]

        self.csf = nib.load(base_dir / f'nihpd_{symmetry}_{age_range}_csf.nii').get_fdata().transpose(2, 1, 0)[:, ::-1, :]
        self.gm = nib.load(base_dir / f'nihpd_{symmetry}_{age_range}_gm.nii').get_fdata().transpose(2, 1, 0)[:, ::-1, :]
        self.wm = nib.load(base_dir / f'nihpd_{symmetry}_{age_range}_wm.nii').get_fdata().transpose(2, 1, 0)[:, ::-1, :]
        self.mask = nib.load(base_dir / f'nihpd_{symmetry}_{age_range}_mask.nii').get_fdata().transpose(2, 1, 0)[:, ::-1, :]
        self.pdw = nib.load(base_dir / f'nihpd_{symmetry}_{age_range}_pdw.nii').get_fdata().transpose(2, 1, 0)[:, ::-1, :]

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

            new_spacings = np.array(original_shape) / np.array(new_shape) * [self.dz, self.dx, self.dy]
            self.dz, self.dx, self.dy = new_spacings
        else:
            shape = original_shape
        self.nz, self.nx, self.ny = shape

        skull = (self.mask == 0)*self.pdw / self.pdw.max()
        skull[skull < 0.1] = 0
        self.skull = skull

    def get_CT_number_phantom(self):
        if len(self._lesion_coords) > 0:
            return self._phantom
        phantom = self.csf*self.csf_HU + self.gm*self.gm_HU + self.wm*self.wm_HU + self.skull*self.skull_HU
        phantom[phantom <= 0] = self.air_HU
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
        '''obtains rudimentary mask of skull voxels using threshold of proton-density weighted image and full mask'''
        skull_map = (self.mask == 0)*self.pdw / self.pdw.max()
        skull_map[skull_map < 0.1] = 0
        skull_map[skull_map > 0] = 1
        return skull_map
