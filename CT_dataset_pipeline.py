# %%
from pathlib import Path
from io import StringIO

import pandas as pd
import nibabel as nib
import numpy as np

from pedsilicoICH.image_acquisition import CTobj
from pedsilicoICH.lesion_insertion import add_random_sphere_lesion

# %% [markdown]
# Define Ground Truth Head
phantom_dir = Path('MIDA Head Phantom')
img = nib.load(phantom_dir/'MIDA_v1.nii')
phantom = np.array(img.get_fdata()).transpose(1, 0, 2)

with open(phantom_dir / 'MIDA_v1.txt', 'rb') as data:
    df = pd.read_csv(StringIO(data.read().decode(errors='replace')), sep='\t', names=['grayscale','c1', 'c2', 'c3', 'material'])
    material_lut = df.iloc[:-8]
    temp = df.iloc[-8:, :2].set_index('grayscale').T
    nx, ny, nz, dx, dy, dz = temp.nx.item(), temp.ny.item(), temp.nz.item(), temp.dx.item()*1000, temp.dy.item()*1000, temp.dz.item()*1000

material_lut = pd.read_csv('notebooks/material_lut.csv')

phantom[phantom == 50] = -1000  # air
for idx, row in material_lut[~material_lut['CT Number [HU]'].isna()].iterrows():
    phantom[phantom == row.grayscale] = row['CT Number [HU]']

base_dir = Path('/gpfs_projects/brandon.nelson/pedsilicoICH/MIDA_analytical_sphere_ICH')

desired_cases = 100
case_count = 0

min_radius, max_radius = 2, 20
min_contrast, max_contrast = 20, 200
startZ = -95
endZ = 75
views = 1000
fov = 250
mA = 200
kVp = 120

names = []
files = []
masks = []
contrast_list = []
radius_list = []
center_x_list = []
center_y_list = []
center_z_list = []

material = 'white_matter'

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

    patient_name = f'case_{case_count:03d}'
    output_dir = base_dir / patient_name
    output_dir.mkdir(exist_ok=True, parents=True)
    ct = CTobj(img_w_lesion, spacings=(dz, dx, dy), patientname=patient_name,
               studyname='full volume long scan', output_dir=output_dir)
    ct.run_scan(startZ=startZ, endZ=endZ, views=views, mA=mA, kVp=kVp)
    ct.run_recon(fov=fov)
    dicom_path = output_dir / 'dicoms'
    dcm_files = ct.write_to_dicom(dicom_path / f'{patient_name}.dcm')
    # figure out how to save out ground truth lesion segmentation, currently only save coordinates

    lesion_only = CTobj(np.where(lesion_image > 0, 0, - 1000), spacings=(dz, dx, dy),
                        patientname='lesion only', output_dir=output_dir / 'lesion only',
                        materials={'ICRU_lung_adult_healthy':-1000, 'water': 0})
    lesion_only.xcist.cfg.physics.energyCount = 2
    lesion_only.xcist.cfg.physics.monochromatic = 0
    lesion_only.xcist.cfg.physics.enableElectronicNoise = 0
    lesion_only.xcist.cfg.physics.enableQuantumNoise = 0
    lesion_only.run_scan(mA=500, startZ=startZ, endZ=endZ, views=100)
    lesion_only.run_recon(fov=fov)
    lesion_only.recon = lesion_only.recon > - 950

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
