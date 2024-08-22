# %%
from pathlib import Path
from io import StringIO

import pandas as pd
import nibabel as nib
import numpy as np
from monai.transforms import RandAffine

from pedsilicoICH.image_acquisition import CTobj
from pedsilicoICH.lesion_insertion import add_random_sphere_lesion
from pedsilicoICH.artifact_generation import transform_image_label_pair

# %% [markdown]
# Define Ground Truth Head
phantom_dir = Path('MIDA Head Phantom')
img = nib.load(phantom_dir/'MIDA_v1.nii')
phantom = np.array(img.get_fdata()).transpose(1, 0, 2)

with open(phantom_dir / 'MIDA_v1.txt', 'rb') as data:
    df = pd.read_csv(StringIO(data.read().decode(errors='replace')), sep='\t',
                     names=['grayscale', 'c1', 'c2', 'c3', 'material'])
    material_lut = df.iloc[:-8]
    temp = df.iloc[-8:, :2].set_index('grayscale').T
    nx, ny, nz, dx, dy, dz = temp.nx.item(), temp.ny.item(), temp.nz.item(), temp.dx.item()*1000, temp.dy.item()*1000, temp.dz.item()*1000

material_lut = pd.read_csv('notebooks/material_lut.csv')

phantom[phantom == 50] = -1000  # air
for idx, row in material_lut[~material_lut['CT Number [HU]'].isna()].iterrows():
    phantom[phantom == row.grayscale] = row['CT Number [HU]']

# %% [markdown]
# Define simulation parameters
base_dir = Path('/gpfs_projects/brandon.nelson/pedsilicoICH/MIDA_sphere_ICH_position_augmentation') # output directory to save simulation results
# consider turning this script into a command line function, similar to https://github.com/bnel1201/Virtual-Patient-CT-Simulations/blob/PedSilicoICH-Pilot/run_xcat.py
# it makes the files more annoying to develop, but avoids having different developers have personal copies just to update output directories

desired_cases = 100

add_positioning_augmentation = True  # whether to apply random rotation, translation, and resizing of the ground truth phantom head, see line 78 for default parameters

# simple sphere lesion settings
min_radius, max_radius = 2, 20
min_contrast, max_contrast = 20, 200
material = 'white_matter'  # brain region where lesion will be inserted options based on `material_lut` materials

# scan acquisition settings
startZ = -95
endZ = 75
views = 1000
fov = 250
mA = 200
kVp = 120
# End of user defined settings

# define empty list of metadata columns to store in the resulting dataframe
names = []
files = []
masks = []
contrast_list = []
radius_list = []
center_x_list = []
center_y_list = []
center_z_list = []

case_count = 0
while case_count < desired_cases:
    print(f'Case count number: {case_count}')
    radius = np.random.randint(min_radius, max_radius)
    contrast = np.random.randint(min_contrast, max_contrast)

    ground_truth_image = phantom
    try:
        brain_mask = ground_truth_image == material_lut.loc[material_lut['xcist material']==material]['CT Number [HU]'].iloc[1]   
        img_w_lesion, lesion_image, lesion_coords = add_random_sphere_lesion(ground_truth_image, brain_mask, radius=radius, contrast=contrast)
    except:
        print('Failed to insert lesion, continuing...')
        continue

    if add_positioning_augmentation:
        transform = RandAffine(prob=0.5,
                               rotate_range=[np.pi/4, np.pi/20, np.pi/20],
                               translate_range=[10, 10, 10],
                               scale_range=[0.1, 0.1, 0.1],
                               padding_mode="border")
        img_w_lesion, lesion_image = transform_image_label_pair(transform,
                                                                img_w_lesion,
                                                                lesion_image,
                                                                seed=42)

    patient_name = f'case_{case_count:03d}'
    output_dir = base_dir / patient_name
    output_dir.mkdir(exist_ok=True, parents=True)
    ct = CTobj(img_w_lesion, spacings=(dz, dx, dy), patientname=patient_name,
               studyname='full volume long scan', output_dir=output_dir)
    ct.run_scan(startZ=startZ, endZ=endZ, views=views, mA=mA, kVp=kVp)
    ct.run_recon(fov=fov)
    dicom_path = output_dir / 'dicoms'
    dcm_files = ct.write_to_dicom(dicom_path / f'{patient_name}.dcm')

    lesion_only = ct
    lesion_only.recon = ct.get_lesion_mask(lesion_image, startZ=startZ, endZ=endZ)

    dicom_path = output_dir / 'lesion_masks'
    mask_files = lesion_only.write_to_dicom(dicom_path / f'{patient_name}_mask.dcm')

    for f, m in zip(dcm_files, mask_files):
        names.append(patient_name)
        files.append(f)
        masks.append(m)
        contrast_list.append(contrast)
        radius_list.append(radius)
        center_x_list.append(lesion_coords[0])
        center_y_list.append(lesion_coords[1])
        center_z_list.append(lesion_coords[2])
    case_count += 1
    metadata = pd.DataFrame({'name': names,
                             'contrast': contrast_list,
                             'radius': radius_list,
                             'center x': center_x_list,
                             'center y': center_y_list,
                             'center z': center_z_list,
                             'image file': files,
                             'mask file': masks})
    metadata.to_csv(base_dir / 'metadata.csv', index=False)
