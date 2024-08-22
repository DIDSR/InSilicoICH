'''
module for working with phantoms
'''

from pathlib import Path
from io import StringIO

import numpy as np
import nibabel as nib
import pandas as pd

from . import dicom_to_voxelized_phantom


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
        ['ICRU_lung_adult_healthy', 'ICRU_adipose_adult2', 'ICRU_liver_adult', 'water', 'ICRU_skeleton_cortical_bone_adult'],
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


class Phantom:
    def __init__(self, phantom_dir):
        self.phantom_dir = phantom_dir
        self.materials = {'gray matter': 45,
                          'white matter': 20,
                          'CSF': 15,
                          'skull': 1000}

    def get_CT_number_phantom(self):
        pass

    def get_material_mask(self, material):
        pass


class MIDA_Head(Phantom):
    def __init__(self, phantom_dir, csf_HU=15, gm_HU=45, wm_HU=20, skull_HU=1000):
        super().__init__(phantom_dir)

        self.age = 39  # median american age
        self.csf_HU = csf_HU
        self.gm_HU = gm_HU
        self.wm_HU = wm_HU
        self.skull_HU = skull_HU
        self.air_HU = -1000
        self.material_lut = self._load_material_LUT()

    def _load_material_LUT(self):

        with open(self.phantom_dir / 'MIDA_v1.txt', 'rb') as data:
            df = pd.read_csv(StringIO(data.read().decode(errors='replace')), sep='\t', names=['grayscale','c1', 'c2', 'c3', 'material'])
            material_lut = df.iloc[:-8] # read in the material look up table from the top of the txt file
            image_size_info = df.iloc[-8:, :2].set_index('grayscale').T
            self.nx, self.ny, self.nz = int(image_size_info.nx.item()), int(image_size_info.ny.item()), int(image_size_info.nz.item())
            self.dx, self.dy, self.dz = image_size_info.dx.item()*1000, image_size_info.dy.item()*1000, image_size_info.dz.item()*1000

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

        material_lut.loc[material_lut.material == 'CSF General', 'grayscale'] = 32
        material_lut.loc[material_lut.material == 'CSF General', 'xcist material'] = 'CSF'
        material_lut.loc[material_lut.material == 'CSF General', 'grayscale'] = 32
        material_lut.loc[material_lut.material == 'CSF General', 'CT Number [HU]'] = self.csf_HU
        material_lut.loc[material_lut.material == 'CSF Ventricles', 'xcist material'] = 'CSF'
        material_lut.loc[material_lut.material == 'CSF Ventricles', 'CT Number [HU]'] = self.csf_HU

        material_lut.loc[material_lut.material == 'Skull', 'grayscale'] = 62
        material_lut.loc[material_lut.material == 'Skull', 'xcist material'] = 'ncat_skull'
        material_lut.loc[material_lut.material == 'Skull', 'CT Number [HU]'] = 900

        material_lut.loc[material_lut.material == 'Skull Diplo�', 'grayscale'] = 52
        material_lut.loc[material_lut.material == 'Skull Diplo�', 'xcist material'] = 'ncat_skull'
        material_lut.loc[material_lut.material == 'Skull Diplo�', 'CT Number [HU]'] = 800 # https://en.wikipedia.org/wiki/Diplo%C3%AB

        material_lut.loc[material_lut.material == 'Skull Inner Table', 'grayscale'] = 52
        material_lut.loc[material_lut.material == 'Skull Inner Table', 'xcist material'] = 'ncat_skull'
        material_lut.loc[material_lut.material == 'Skull Inner Table', 'CT Number [HU]'] = 1000

        material_lut.loc[material_lut.material == 'Skull Outer Table', 'grayscale'] = 54
        material_lut.loc[material_lut.material == 'Skull Outer Table', 'xcist material'] = 'ncat_skull'
        material_lut.loc[material_lut.material == 'Skull Outer Table', 'CT Number [HU]'] = 1000

        material_lut.loc[material_lut.material == 'Adipose Tissue', 'grayscale'] = 62
        material_lut.loc[material_lut.material == 'Adipose Tissue', 'CT Number [HU]'] = -120

        material_lut.loc[material_lut.material == 'Muscle (General)', 'grayscale'] = 63
        material_lut.loc[material_lut.material == 'Muscle (General)', 'CT Number [HU]'] = 55

        return material_lut[~material_lut['CT Number [HU]'].isna()]

    def get_CT_number_phantom(self):

        material_lut = self.material_lut
        img = nib.load(self.phantom_dir/'MIDA_v1.nii')
        phantom = np.array(img.get_fdata()).transpose(1, 0, 2)
        phantom[phantom == 50] = -1000  # air
        for _, row in material_lut[~material_lut['CT Number [HU]'].isna()].iterrows():
            phantom[phantom == row.grayscale] = row['CT Number [HU]']
        return phantom

    def get_material_mask(self, material):
        if material not in self.materials:
            raise ValueError(f'{material} not in {self.materials.keys()}')
        return self.get_CT_number_phantom() == self.materials[material]


def get_mean_age(age_range: str):
    return (float(age_range.split('-')[1])+float(age_range.split('-')[0]))/2


class NIHPD_Head(Phantom):
    '''
    loads MR brain atlas of mean `age`, downloaded from <https://www.bic.mni.mcgill.ca/~vfonov/nihpd/obj1> and saved in `base_dir`

    :param base_dir: str, directory holding .nii files from https://www.bic.mni.mcgill.ca/~vfonov/nihpd/obj1
    :param age: float, mean age of the atlas phantom to load. Note the brain atlases are stored as age ranges, thus the mean age is the mid point of the range (upper+lower)/2
    :param symmetry: optional, the atlases are provided in their natural asymmetric or artificially generated symmetric state, default is asymmetric, see article for more details: 
    1. Fonov V, Evans AC, Botteron K, Almli CR, McKinstry RC, Collins DL. Unbiased average age-appropriate atlases for pediatric studies. NeuroImage. 2011;54(1):313-327. doi:10.1016/j.neuroimage.2010.07.033
    '''
    def __init__(self, phantom_dir, age: float, symmetric=False, csf_HU = 15, gm_HU = 45, wm_HU = 20, skull_HU = 1000):
        super().__init__(phantom_dir)
        self.age = age
        self.csf_HU = csf_HU
        self.gm_HU = gm_HU
        self.wm_HU = wm_HU
        self.skull_HU = skull_HU
        self.air_HU = -1000

        nii_files = list(self.phantom_dir.glob('*.nii'))
        age_ranges = [o.stem.split('_')[2] for o in nii_files]
        ages = {get_mean_age(o) : o for o in age_ranges}

        if age not in ages:
            raise ValueError(f'age {age} not in {sorted(ages.keys())}')
        age_range = ages[age]

        base_dir = self.phantom_dir
        symmetry = 'sym' if symmetric else 'asym'

        nib_img = nib.load(base_dir / f'nihpd_{symmetry}_{age_range}_pdw.nii')
        header = nib_img.header
        self.dx, self.dy, self.dz = header['pixdim'][1:4]
        
        self.csf = nib.load(base_dir / f'nihpd_{symmetry}_{age_range}_csf.nii').get_fdata().transpose(2, 1, 0)[:,::-1,:]
        self.gm = nib.load(base_dir / f'nihpd_{symmetry}_{age_range}_gm.nii').get_fdata().transpose(2, 1, 0)[:,::-1,:]
        self.wm = nib.load(base_dir / f'nihpd_{symmetry}_{age_range}_wm.nii').get_fdata().transpose(2, 1, 0)[:,::-1,:]
        self.mask = nib.load(base_dir / f'nihpd_{symmetry}_{age_range}_mask.nii').get_fdata().transpose(2, 1, 0)[:,::-1,:]
        self.pdw = nib.load(base_dir / f'nihpd_{symmetry}_{age_range}_pdw.nii').get_fdata().transpose(2, 1, 0)[:,::-1,:]

        skull = (self.mask == 0)*self.pdw / self.pdw.max()
        skull[skull < 0.1] = 0
        self.skull = skull

    def get_CT_number_phantom(self):
        
        phantom = self.csf*self.csf_HU + self.gm*self.gm_HU + self.wm*self.wm_HU + self.skull*self.skull_HU
        phantom[phantom==0] = self.air_HU
        return phantom

    def get_material_mask(self, material):            
        if material == 'white matter':
            return self.wm > 0.3

        if material == 'gray matter':
            return self.gm > 0.3

        if material == 'CSF':
            return self.csf > 0.3
            
        raise ValueError(f'{material} not in {self.materials.keys()}')