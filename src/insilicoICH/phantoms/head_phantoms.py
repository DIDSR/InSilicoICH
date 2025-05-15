'''
module for working with phantoms
'''

from pathlib import Path
import os

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import pandas as pd
import skimage as ski
from skimage.morphology import (binary_closing,
                                remove_small_holes,
                                binary_dilation,
                                binary_erosion)
from monai.transforms import ResizeWithPadOrCrop

from scipy.ndimage import distance_transform_edt

from .base_phantoms import LesionPhantom, resize, get_mean_age
from .utils import download_and_extract_archive
from ..hooks import hookimpl


class HeadPhantom(LesionPhantom):
    def __init__(self, phantom_dir, shape=None, **kwargs):
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
        super().__init__(phantom, spacings, **kwargs)
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
    age = 38  # add 38 as the median US adult age to represent MIDA, consider
#  other identifiers when adding more patients

    def __init__(self, phantom_dir, shape=None):
        if not phantom_dir.exists():
            raise FileNotFoundError(f'''
MIDA head phantom files not found in {phantom_dir}

To use MIDA head phantoms, please download them from:
 <https://cdrh-rst.fda.gov/mida-multimodal-imaging-based-model-human-head-and-neck>

and place in your `PHANTOM_DIRECTORY`, see `load_phantom` for more details
''')
        super().__init__(phantom_dir, shape)
        self.patient_name = 'Adult MIDA Head'
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
        return skull_map.astype(bool)


def minmax_scale(x, feature_range=(0, 1)):
    'adapted from sklearn https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html'
    x_std = (x - x.min())/(x.max() - x.min())
    return x_std*(max(feature_range) - min(feature_range)) + min(feature_range)


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
    ages = [6.5, 9.0, 10.5, 11.5, 12.0, 15.75]
    relative_head_size = dict(zip(ages, [0.8, 0.82, 0.85, 0.87, 0.9, 0.95]))
    url = 'https://www.bic.mni.mcgill.ca/~vfonov/nihpd/obj1_analyze.zip'

    def __init__(self, phantom_dir, age: float, symmetric=False, shape=None,
                 skull_seg_method='otsu', add_sutures=True):
        phantom_dir = Path(phantom_dir)
        self.age = age
        self.patient_name = f'{age} yr NIHPD Head'
        self.symmetric = symmetric
        self.skull_seg_method = skull_seg_method
        self.add_sutures = add_sutures
        if not phantom_dir.exists():
            print(f'''
`PHANTOM_DIRECTORY` {phantom_dir} not found, now downloading NIHPD phantoms
from {NIHPD_Head.url}

If you have already downloaded NIHPD and MIDA head phantoms, please see
`load_phantom` for details on how to add their locations.
''')
            download_and_extract_archive(NIHPD_Head.url, phantom_dir)
        super().__init__(phantom_dir, shape, age=age)

    def load_phantom(self, phantom_dir):
        'sets ._phantom, and .dx, .dy, .dz, .nx, .ny, .nz'
        nii_files = list(phantom_dir.glob('*.nii'))
        age = self.age
        age_ranges = [o.stem.split('_')[2] for o in nii_files]
        ages = {get_mean_age(o): o for o in age_ranges}

        if age not in ages:
            raise ValueError(f'age {age} not in {sorted(ages.keys())}\
from {phantom_dir}')
        age_range = ages[age]

        base_dir = phantom_dir
        symmetry = 'sym' if self.symmetric else 'asym'

        nib_img = nib.load(base_dir / f'nihpd_{symmetry}_{age_range}_pdw.nii')
        header = nib_img.header
        self.dx, self.dy, self.dz = header['pixdim'][1:4]
        self.dx, self.dy, self.dz = list(map(lambda o: o*self.relative_head_size[age], (self.dx, self.dy, self.dz)))
        self.t1w = nib.load(
            base_dir / f'nihpd_{symmetry}_{age_range}_t1w.nii'
            ).get_fdata().transpose(2, 1, 0)[:, ::-1, :]
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

        try:
            src_dir = Path(__file__).parents[0]
            self.pseudoct = nib.load(
                src_dir / 'NIHPD_pseudoCT/nihpd_asym_04.5-08.5_pct.nii'
                ).get_fdata().transpose(2, 1, 0)[:, ::-1, :]
            self.pseudoct = self.pseudoct[::-1]
        except FileNotFoundError:
            # if pseudoct can't be loaded, default to
            # otsu skull segmentation method
            self.skull_seg_method = 'otsu'
            print('pseudo-CT images not found; defaulting to otsu segmentation method')

        self.csf = self.csf[::-1]
        self.gm = self.gm[::-1]
        self.wm = self.wm[::-1]
        self.mask = self.mask[::-1].astype(bool)
        self.pdw = self.pdw[::-1]
        self.t1w = self.t1w[::-1]

        self.nz, self.nx, self.ny = self.csf.shape

        self.head_mask = self.get_head_mask()
        self.skull = self.get_skull_map()
        self._phantom = self.get_CT_number_phantom()

        spacings = self.dz, self.dx, self.dy
        return self._phantom, spacings

    def resize(self, shape=None):
        original_shape = self.csf.shape

        # resize original images
        self.csf = resize(self.csf, shape).numpy()
        self.gm = resize(self.gm, shape).numpy()
        self.wm = resize(self.wm, shape).numpy()
        self.mask = resize(self.mask, shape).numpy()
        self.pdw = resize(self.pdw, shape).numpy()
        self.t1w = resize(self.t1w, shape).numpy()

        # resize additional
        self.head_mask = resize(self.head_mask, shape).numpy().astype(bool)
        self.skull = resize(self.skull, shape).numpy()
        new_shape = self.csf.shape

        new_spacings = np.array(original_shape) / np.array(new_shape) *\
            [self.dz, self.dx, self.dy]
        self.dz, self.dx, self.dy = new_spacings
        self.nz, self.nx, self.ny = shape

    def get_sutures(self, thickness=2, thresh=30):
        """
        returns suture mask to the self skull

        :param thickness: thickness in pixels of the suture
        :returns: boolean suture mask that can be used to set skull suture
            values
        """
        src_dir = Path(__file__).parents[1]
        fname = src_dir / 'annotations/suture/NIHPD_Head_Phantom/labelmap.nrrd'
        data = sitk.GetArrayFromImage(sitk.ReadImage(fname)).transpose(2, 1, 0)[::-1, ::-1]
        skull = self.get_skull_map()
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

    def assign_HUs(self, feature_range=(-100, 100)):
        phantom = self.csf*self.materials['CSF'] +\
                    self.gm*self.materials['gray matter'] +\
                    self.wm*self.materials['white matter'] +\
                    self.skull*self.materials['skull']
        # fills remaining tissues with scaled pdw to approximate
        phantom[phantom < self.materials['CSF']] = minmax_scale(
            self.pdw[phantom < self.materials['CSF']], feature_range)
        skin = binary_erosion(self.head_mask, np.ones(3*[3])) ^ binary_dilation(self.head_mask, np.ones(3*[3]))
        phantom[skin] = minmax_scale(self.pdw[skin], feature_range)
        phantom[self.head_mask & (phantom < 0)] = minmax_scale(
            self.pdw[self.head_mask & (phantom < 0)], feature_range)
        sinous = (phantom < 0) & self.head_mask
        phantom[sinous] = self.materials['CSF']  # approximates blood
        if self.skull_seg_method == 'pseudoct':
            phantom[self.skull] = self.pseudoct[self.skull]
        if self.add_sutures:
            sutures = self.get_sutures()
            phantom[sutures] = 0  # assume water HU
        phantom[phantom < 0] = self.materials['air']

        # # TODO: dura map currently overlaps with new skull methods, need fix
        # phantom[self.get_dura_map()] = 50  # HU same as MIDA
        return phantom

    def get_CT_number_phantom(self):
        if len(self._lesion_coords) > 0:
            return self._phantom
        phantom = self.assign_HUs()

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

    def get_head_mask(self):
        '''obtains mask of head voxels'''
        thresh = 35.61893018554474  # ski.filters.threshold_otsu(vol)
        head_mask = self.pdw > thresh
        head_mask = binary_closing(head_mask, np.ones(3*[7]))
        head_mask = remove_small_holes(head_mask, area_threshold=1e5)
        return head_mask

    def get_skull_map(self):
        '''obtains mask of skull voxels'''
        if self.skull_seg_method == 'pseudoct':
            print('using pseudoct method')
            # currently, pesudoCT images are generated outside of the repository, TODO: integrate code
            threshold = 300  # units: HU
            skull = np.where(self.pseudoct > threshold, 1, 0)

        elif self.skull_seg_method == 'otsu':
            vol = self.t1w
            thresh = 35.61893018554474  # precalculated by performing otsu across all nihpd
            skull = vol < thresh
            skull = skull & ~binary_erosion(self.mask, np.ones(3*[3]))
            skull = skull & self.head_mask
            skull = skull*(-1*(1-self.mask))
        elif self.skull_seg_method == 'old':  # old method for posterity (gives VERY thick skull...)
            skull = (self.mask == 0)*self.pdw / self.pdw.max()
            skull[skull < 0.1] = 0
            skull[skull > 0] = 1
        return skull.astype(bool)


possible_ages = NIHPD_Head.ages + [MIDA_Head.age]


@hookimpl
def register_phantom_types():
    return [MIDA_Head, NIHPD_Head]
