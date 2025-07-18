'''
module for working with phantoms
'''

from pathlib import Path
import os
from functools import partial

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import pandas as pd
import skimage as ski
from dotenv import load_dotenv
from warnings import warn

from skimage.morphology import (binary_closing,
                                remove_small_holes,
                                binary_dilation,
                                binary_erosion)
from monai.transforms import ResizeWithPadOrCrop

from scipy.ndimage import distance_transform_edt

from .base_phantoms import LesionPhantom, resize, get_mean_age, get_transformation_src_dst
from .utils import download_and_extract_archive
from VITools.hooks import hookimpl
from ..annotations.skull.NIHPD_Head_Phantom.skull_fracture import SkullFracture
import random
from ..annotations.skull.NIHPD_Head_Phantom.fracture_projector import SkullFractureProjector


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

    def get_warp_exclusion_mask(self):
        return self.get_skull_map().astype(bool)

    def get_warp_inclusion_mask(self):
        return self.get_CT_number_phantom() > -900
    
    def save_volume_nifti(self, volume: np.ndarray, affine: np.ndarray, path_save: str):
        """
        Saves volume as nifti with the given geometry as an affine matrix
        """
        nifti_img = nib.Nifti1Image(volume, affine)
        nib.save(nifti_img, path_save)

    def get_nifti_info(self, nifti_path):
        """
        Returns the nifti geometry.
        """
        img = nib.load(nifti_path)
        array = img.get_fdata()
        shape = img.shape  # (z, y, x) or (x, y, z), depending on orientation
        affine = img.affine  # 4x4 affine transformation matrix
        spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))  # voxel spacing
        origin = affine[:3, 3]  # origin (translation component)

        return shape, origin, spacing, affine, array

class MIDA_Head(HeadPhantom):
    name = 'MIDA Head'
    ages = [38.0]  # add 38 as the median US adult age to represent MIDA, consider
#  other identifiers when adding more patients

    def __init__(self, phantom_dir, shape=None, age=None):
        self.age = age or MIDA_Head.ages[0]
        if not phantom_dir.exists():
            raise FileNotFoundError(f'''
MIDA head phantom files not found in {phantom_dir}

To use MIDA head phantoms, please download them from:
 <https://cdrh-rst.fda.gov/mida-multimodal-imaging-based-model-human-head-and-neck>

and place in your `PHANTOM_DIRECTORY`, see `load_phantom` for more details
''')
        super().__init__(phantom_dir, shape, age=self.age)
        self.patient_name = f'{age} yr {MIDA_Head.name}'
        self.material_lut = self._load_material_LUT()

    def load_phantom(self, phantom_dir):
        'returns phantom and spacings'
        img = nib.load(phantom_dir/'MIDA_v1.nii')
        phantom = np.array(img.get_fdata()).transpose(1, 0, 2)[::-1]
        header = img.header
        dx, dy, dz = header['pixdim'][1:4]
        spacings = dz, dx, dy
        return phantom, spacings

    def get_warp_exclusion_mask(self):
        skull = self.get_skull_map()
        mask = np.zeros_like(skull, dtype=bool)
        for idx in range(self.shape[0]):
            skull_slice = skull[idx]
            flood_mask = ski.segmentation.flood(skull_slice, seed_point=(0, 0))
            skull_slice[flood_mask] = True
            mask[idx] = skull_slice
        return mask

    def get_warp_inclusion_mask(self):
        return np.where(self.warp_exclusion_mask, False, True)

    def get_warp_coordinates(self, lesion, idx, strength=1):
        # get lesion coordinates
        src, dst = get_transformation_src_dst(lesion[idx], strength=strength)
        src = np.argwhere(src)
        dst = np.argwhere(dst)
        skull_slice = self.warp_exclusion_mask[idx]
        # using the entire inner boundary of the skull mask seems to work great as anchor points
        skull_boundary = ski.segmentation.find_boundaries(skull_slice, mode='inner', background=0)
        skull_sample = np.argwhere(skull_boundary)

        warp_src = warp_dst = skull_sample  # initialize warp_src and warp_dst with the skull boundary voxels

        src_subset = src[np.round(np.linspace(0, len(src)-1, 5)).astype(int)] # subsample points from the src points
        dst_subset = dst[np.round(np.linspace(0, len(dst)-1, 5)).astype(int)] # subsample points from the dst points

        warp_src = np.insert(warp_src, 0, src_subset, axis=0) # insert src subset into main warp list
        warp_dst = np.insert(warp_dst, 0, dst_subset, axis=0) # insert dst subset into main warp list

        # insert the four corner coordinates for added warp stability
        warp_src = np.insert(warp_src, 0, [[0, 0],
                                           [0, lesion.shape[1]],
                                           [lesion.shape[0], 0],
                                           [lesion.shape[0], lesion.shape[1]]
                                           ], axis=0)
        warp_dst = np.insert(warp_dst, 0, [[0, 0],
                                           [0, lesion.shape[1]],
                                           [lesion.shape[0], 0],
                                           [lesion.shape[0], lesion.shape[1]]
                                           ], axis=0)
        return warp_src, warp_dst

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
    name = 'NIHPD Head'
    relative_head_size = dict(zip(ages, [0.8, 0.82, 0.85, 0.87, 0.9, 0.95]))
    url = 'https://www.bic.mni.mcgill.ca/~vfonov/nihpd/obj1_analyze.zip'

    def __init__(self, phantom_dir, age: float, symmetric=False, shape=None,
                 skull_seg_method='otsu', add_sutures=False, add_fractures=True):
        phantom_dir = Path(phantom_dir)
        self.age = age
        self.patient_name = f'{age} yr {NIHPD_Head.name}'
        self.symmetric = symmetric
        self.skull_seg_method = skull_seg_method
        self.add_sutures = add_sutures
        self.add_fractures = add_fractures
        self.threshold_degree_phi = 100
        self.frectures_seg = None
        if not phantom_dir.exists():
            print(f'''
`PHANTOM_DIRECTORY` {phantom_dir} not found, now downloading NIHPD phantoms
from {NIHPD_Head.url}

If you have already downloaded NIHPD and MIDA head phantoms, please see
`load_phantom` for details on how to add their locations.
''')
            download_and_extract_archive(NIHPD_Head.url, phantom_dir)
        self.dict_skull_paths = {
            "path_mesh_brainmask": os.path.join(
                Path(__file__).parents[1],
                "annotations/skull/NIHPD_Head_Phantom/assets",
                "mesh_brain.vtk",
            ),
            "path_mask_brain": phantom_dir / "nihpd_asym_04.5-08.5_mask.nii",
            "path_file_config": os.path.join(
                Path(__file__).parents[1],
                "annotations/skull/NIHPD_Head_Phantom/assets",
                "config.toml",
            ),
            "path_skull_mesh": os.path.join(
                Path(__file__).parents[1],
                "annotations/skull/NIHPD_Head_Phantom/assets",
                "skull_mesh.vtk",
            )
        }
        super().__init__(phantom_dir, shape, age=age, patient_name=self.patient_name)

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

    def get_warp_coordinates(self, lesion, idx, strength=1):
        # get lesion coordinates
        src, dst = get_transformation_src_dst(lesion[idx], strength=strength)
        src = np.argwhere(src)
        dst = np.argwhere(dst)
        # use brain mask to define anchor points
        skull_boundary = self.warp_exclusion_mask[idx]
        skull_sample = np.argwhere(skull_boundary)

        warp_src = warp_dst = skull_sample  # initialize warp_src and warp_dst with the skull boundary voxels

        src_subset = src[np.round(np.linspace(0, len(src)-1, 5)).astype(int)] # subsample points from the src points
        dst_subset = dst[np.round(np.linspace(0, len(dst)-1, 5)).astype(int)] # subsample points from the dst points

        warp_src = np.insert(warp_src, 0, src_subset, axis=0) # insert src subset into main warp list
        warp_dst = np.insert(warp_dst, 0, dst_subset, axis=0) # insert dst subset into main warp list

        # insert the four corner coordinates for added warp stability
        warp_src = np.insert(warp_src, 0, [[0, 0],
                                           [0, lesion.shape[1]],
                                           [lesion.shape[0], 0],
                                           [lesion.shape[0], lesion.shape[1]]
                                           ], axis=0)
        warp_dst = np.insert(warp_dst, 0, [[0, 0],
                                           [0, lesion.shape[1]],
                                           [lesion.shape[0], 0],
                                           [lesion.shape[0], lesion.shape[1]]
                                           ], axis=0)
        return warp_src, warp_dst

    def get_warp_exclusion_mask(self):
        '''
        approximates the skull as the outer boundary of the brain mask
        '''
        return ski.segmentation.find_boundaries(self.mask.astype(bool),
                                                mode='outer', background=0)

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

    def fetch_fractures_seg(self, length, phi_degree, theta_degree):
        """
        Fetch the skull fracture with given parameters.
        """
        assert phi_degree <= self.threshold_degree_phi and phi_degree > 0, "requirement 0 < phi_degree < 100 is not met"

        skull = self.get_skull_map()
        
        skull_int = skull.astype(np.int32).transpose(2, 1, 0)[:, ::-1, :]
        projector = SkullFractureProjector(skull_mask=skull_int)

        # Perform ray casting projection
        # Note: centroid is considered as the center of the 3D array
        fractures_proj = projector.centroid_ray_casting_random_walk(length=length, phi_degree=phi_degree, theta_degree=theta_degree)

        skull_int = skull_int.transpose(2, 1, 0)[:, ::-1, :]
        fractures_proj = fractures_proj.transpose(2, 1, 0)[:, ::-1, :]
        self.fracture_seg = fractures_proj
    
        return fractures_proj
    
    def get_fractures(self, length: int = random.randint(50, 200), phi_degree: float = random.uniform(0, 60), theta_degree: float = random.uniform(0, 360)):
        """
        returns fracture mask to the self skull

        :param thickness: thickness in pixels of the fracture
        :param thresh: distance threshold for fracture mask, smaller values means closer to the skull surface
        :returns: boolean fracture mask that can be used to set skull fracture
            values
        """
        fractures_proj = self.fetch_fractures_seg(length=length, phi_degree=phi_degree, theta_degree=theta_degree)
        fractures = fractures_proj.astype(bool)

        return fractures

    def get_fracture_seg_slice_labels(self):
        """
        Returns the binary array with 1 where fracture present in slice, and 0 where not present.
        """
        assert self.fracture_seg is not None, "self.fracture_seg is not available, refer get_fractures()"
        binary_mask = np.any(self.fracture_seg > 0, axis=(1, 2)).astype(np.uint8)

        return binary_mask

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
        if self.add_fractures:
            fractures = self.get_fractures(phi_degree=5)
            phantom[fractures] = 0  # assume water HU
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


class UNC_Head(NIHPD_Head):
    '''
    loads MR brain atlas of mean `age`, downloaded from
    <https://www.nitrc.org/projects/pediatricatlas> and saved in `phantom_dir`

    :param phantom_dir: str, directory holding .nii files from
        https://www.nitrc.org/projects/pediatricatlas
    :param age: float, mean age of the atlas phantom to load. Options
        include neonate (0), 1-, and 2-year-old
    1. Feng Shi, Pew-Thian Yap, Guorong Wu, Hongjun Jia, John H. Gilmore, Weili Lin, Dinggang Shen,
        "Infant Brain Atlases from Neonates to 1- and 2-year-olds", PLoS ONE, 6(4): e18746, 2011
    '''
    ages = [0.0, 1.0, 2.0]
    name = 'UNC Head'
    url = 'https://www.nitrc.org/frs/download.php/14897/UNCInfant012Atlases-2022-10-21.zip'

    def __init__(self, phantom_dir, age: float, symmetric=False, shape=None,
                 skull_seg_method='otsu'):
        phantom_dir = Path(phantom_dir)
        self.age = age
        self.skull_seg_method = skull_seg_method
        if not phantom_dir.exists():
            print(f'''
`PHANTOM_DIRECTORY` {phantom_dir} not found, now downloading UNC phantoms
from {UNC_Head.url}

If you have already downloaded NIHPD and MIDA head phantoms, please see
`load_phantom` for details on how to add their locations.
''')
            download_and_extract_archive(UNC_Head.url, phantom_dir, remove_finished=True)
        super().__init__(phantom_dir, shape=shape, age=age)
        self.patient_name = f'{age} yr {UNC_Head.name}'

        # define material HU; 0-2 yr old based on cases in
        # https://physionet.org/content/ct-ich/1.3.1/ and 
        # https://pubmed.ncbi.nlm.nih.gov/3652069/
        if self.age == 0.0:
            self.materials = {
                'csf': 10,
                'gray matter': 24,
                'white matter': 14,
                'air': -1000,
                'CSF': 10,
                'skull': 400
                }
        if self.age == 1.0:
            self.materials = {
                'csf': 10,
                'gray matter': 35,
                'white matter': 27,
                'air': -1000,
                'CSF': 10,
                'skull': 500
                }
        if self.age == 2.0:
            self.materials = {
                'csf': 10,
                'gray matter': 35,
                'white matter': 27,
                'air': -1000,
                'CSF': 10,
                'skull': 700
                }

    def load_phantom(self, phantom_dir):
        'sets ._phantom, and .dx, .dy, .dz, .nx, .ny, .nz'
        age = self.age

        base_dir = phantom_dir / 'UNCInfant012Atlases-2022-10-21'

        if age == 0.0:
            age_string = 'neo'
        elif age == 1.0:
            age_string = '1yr'
        elif age == 2.0:
            age_string = '2yr'

        nib_img = nib.load(base_dir / f'infant-{age_string}.nii.gz')
        header = nib_img.header
        self.dx, self.dy, self.dz = header['pixdim'][1:4]

        self.intensity = nib.load(base_dir / f'infant-{age_string}-withSkull.nii.gz').get_fdata().transpose(2, 1, 0)[:, ::-1, :]
        self.segmentation = nib.load(base_dir / f'infant-{age_string}-seg.nii.gz').get_fdata().transpose(2, 1, 0)[:, ::-1, :]

        # extract segmentations from seg image
        self.gm = np.where(self.segmentation == 150, 1, 0)
        self.wm = np.where(self.segmentation == 250, 1, 0)
        self.csf = np.where(self.segmentation == 10, 1, 0)
        self.mask = np.where(self.segmentation > 0, 1, 0)

        if self.skull_seg_method == 'pseudoCT':
            try:
                src_dir = Path(__file__).parents[0]
                self.pseudoct = nib.load(
                    src_dir / f'UNC_pseudoCT/UNC_{age_string}_pct.nii'
                    ).get_fdata().transpose(2, 1, 0)[:, ::-1, :]
                self.pseudoct = self.pseudoct[::-1]
            except FileNotFoundError:
                # if pseudoct can't be loaded, default to
                # otsu skull segmentation method
                self.skull_seg_method = 'otsu'
                print('pseudo-CT images not found; defaulting to otsu segmentation method')

        self.gm = self.gm[::-1]
        self.wm = self.wm[::-1]
        self.csf = self.csf[::-1]
        self.mask = self.mask[::-1].astype(bool)
        self.intensity = self.intensity[::-1]

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
        self.intensity = resize(self.t1w, shape).numpy()

        # resize additional
        self.head_mask = resize(self.head_mask, shape).numpy().astype(bool)
        self.skull = resize(self.skull, shape).numpy()

        new_shape = self.csf.shape

        new_spacings = np.array(original_shape) / np.array(new_shape) *\
            [self.dz, self.dx, self.dy]
        self.dz, self.dx, self.dy = new_spacings
        self.nz, self.nx, self.ny = shape

    def assign_HUs(self, feature_range=(-100, 100)):
        phantom = self.csf*self.materials['CSF'] +\
                    self.gm*self.materials['gray matter'] +\
                    self.wm*self.materials['white matter'] +\
                    self.skull*self.materials['skull']
        # fills remaining tissues with scaled intensity to approximate
        phantom[phantom < self.materials['CSF']] = minmax_scale(
            self.intensity[phantom < self.materials['CSF']], feature_range)
        skin = binary_erosion(self.head_mask, np.ones(3*[3])) ^ binary_dilation(self.head_mask, np.ones(3*[3]))
        phantom[skin] = minmax_scale(self.intensity[skin], feature_range)     
        phantom[self.head_mask & (phantom < 0)] = minmax_scale(
            self.intensity[self.head_mask & (phantom < 0)], feature_range)
        sinous = (phantom < 0) & self.head_mask
        phantom[sinous] = self.materials['CSF']  # approximates blood
        if self.skull_seg_method == 'pseudoct':
            phantom[self.skull] = self.pseudoct[self.skull]
        phantom[phantom < 0] = self.materials['air']

        # # TODO: dura map currently overlaps with new skull methods, need fix
        # phantom[self.get_dura_map()] = 50  # HU same as MIDA
        return phantom

    def get_CT_number_phantom(self, add_sutures=False):
        if len(self._lesion_coords) > 0:
            return self._phantom
        phantom = self.assign_HUs()
        if add_sutures:
            sutures = self.get_sutures()
            phantom[sutures] = 0  # assume water HU
        return phantom

    def get_head_mask(self):
        '''obtains mask of head voxels'''
        thresh = 50  # threshold manually determined; ski.filters.threshold_otsu too high 
        head_mask = self.intensity > thresh
        head_mask = binary_closing(head_mask, np.ones(3*[7]))
        head_mask = remove_small_holes(head_mask, area_threshold=1e5)
        return head_mask

    def get_skull_map(self):
        '''obtains mask of skull voxels'''
        if self.skull_seg_method == 'pseudoct':
            # currently, pesudoCT images are generated outside of the repository, TODO: integrate code
            threshold = 300  # units: HU
            skull = np.where(self.pseudoct > threshold, 1, 0)

        elif self.skull_seg_method == 'otsu':
            vol = self.intensity
            #thresh = ski.filters.threshold_otsu(self.intensity) # TODO: SEPARATE THRESHOLD FOR AGE 0, Otsu not working well
            if self.age == 0.0:
                thresh = 100
            elif self.age == 1.0:
                thresh = 160
            elif self.age == 2.0:
                thresh = 150
            skull = vol < thresh
            skull = skull & ~self.mask
            skull = skull & binary_erosion(binary_erosion(self.head_mask, np.ones(3*[3])), np.ones(3*[3]))
            skull = skull*(-1*(1-self.mask))
        elif self.skull_seg_method == 'old': # old method for posterity (gives VERY thick skull...)
            skull = (self.mask == 0)*self.intensity / self.intensity.max()
            skull[skull < 0.1] = 0
            skull[skull > 0] = 1
        return skull.astype(bool)


possible_ages = UNC_Head.ages + NIHPD_Head.ages + MIDA_Head.ages


@hookimpl
def register_phantom_types():
    head_phantoms = {}
    for head_phantom, sub_dir in zip([UNC_Head, NIHPD_Head, MIDA_Head],
                                     ['UNC_Head_phantom', 'NIHPD_Head_Phantom', 'MIDA_Head_Phantom']):
        head_phantoms.update({f"{o} yr {head_phantom.name}": partial(head_phantom, age=o, phantom_dir=phantom_dir / sub_dir)
                              for o in head_phantom.ages})
    return head_phantoms
