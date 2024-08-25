# %% [markdown]
# # CT_dataset_pipeline
#
# runs CT simulations given inputs of a voxelized
# head phantom with ICH lesion insertion and scan acquisition parameters.
#
# Outputs:
#
# 1. a directory structure with dicom images for each simulation run and
# 2. a CSV file containing relevant simulation metadata. This metadata
# includes:
#  - a summary of simulation inputs such as patient identifiers (name, age),
#    as well as lesion insertion parameters (radius and center coordinates
#    in case of a spherical ICH)
#  - a summary of simulation outputs such as saved file locations of the dicom
#    images "image file" as well as lesion segmentation masks 'mask file'
#    lesion volume [mL] is also provided based on the measured lesion volume
#    following CT simulation - this should be similar to volume predicted by
#    the input radius
# %%
from pathlib import Path
from random import choice

import pandas as pd
import numpy as np
from monai.transforms import RandAffine

from pedsilicoICH.ground_truth_definition.phantoms import NIHPD_Head, MIDA_Head
from pedsilicoICH.image_acquisition import CTobj
from pedsilicoICH.lesion_insertion import add_random_sphere_lesion
from pedsilicoICH.artifact_generation import transform_image_label_pair

# %% [markdown]
# Define Ground Truth Head

nihpd_dir = Path('/gpfs_projects/brandon.nelson/pedsilicoICH/brain_atlases/obj1_analyze/')
MIDA_dir = Path('MIDA Head Phantom')

nihpd_ages = [6.5, 9.0, 10.5, 11.5, 12.0, 15.75]
mida_age = 38  # add 38 as the median US adult age to represent MIDA, consider other identifiers when adding more patients
possible_ages = nihpd_ages + [mida_age]

# %% [markdown]
# Define simulation parameters
output_directory = Path('/gpfs_projects/brandon.nelson/pedsilicoICH/mixed_datasets_sphere_ICH') # output directory to save simulation results
# consider turning this script into a command line function, similar to https://github.com/bnel1201/Virtual-Patient-CT-Simulations/blob/PedSilicoICH-Pilot/run_xcat.py
# it makes the files more annoying to develop, but avoids having different developers have personal copies just to update output directories

desired_cases = 100

add_positioning_augmentation = True  # whether to apply random rotation, translation, and resizing of the ground truth phantom head, see line 78 for default parameters

# simple sphere lesion settings
min_radius, max_radius = 2, 20
min_contrast, max_contrast = 20, 200
material = 'white matter'  # brain region where lesion will be inserted options based on `material_lut` materials

# scan acquisition settings
dynamic_scan_range = True  # used to determine z range covering the z extent of the head, can handle different sized heads e.g. peds vs adults
if not dynamic_scan_range:
    startZ = None
    endZ = None  # if none, will default to the full possible scan range units in mm, centered at 0, see ct.scout_view() for examples
views = 1000
fov = 250
mA = 200
kVp = 120
# End of user defined settings

# define empty list of metadata columns to store in the resulting dataframe
ages = []
names = []
files = []
masks = []
contrast_list = []
radius_list = []
center_x_list = []
center_y_list = []
center_z_list = []
lesion_volume_list = []

mida_shape = MIDA_Head(MIDA_dir).get_CT_number_phantom().shape  # for consistent phantom sizes

case_count = 0
while case_count < desired_cases:
    print(f'Case count number: {case_count}')

    age = choice(possible_ages)
    if age == mida_age:
        phantom = MIDA_Head(MIDA_dir)
    else:
        phantom = NIHPD_Head(nihpd_dir, age=age, shape=mida_shape)
    ground_truth_image = phantom.get_CT_number_phantom()

    radius = np.random.randint(min_radius, max_radius)
    contrast = np.random.randint(min_contrast, max_contrast)
    try:
        brain_mask = phantom.get_material_mask(material)
        img_w_lesion, lesion_image, lesion_coords = add_random_sphere_lesion(ground_truth_image,
                                                                             brain_mask,
                                                                             radius=radius,
                                                                             contrast=contrast)
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
                                                                lesion_image)

    patient_name = f'case_{case_count:03d}'
    output_dir = output_directory / patient_name
    output_dir.mkdir(exist_ok=True, parents=True)
    ct = CTobj(img_w_lesion, spacings=(phantom.dz, phantom.dx, phantom.dy),
               patientname=patient_name,
               age=age,
               studyname='full volume long scan',
               output_dir=output_dir)

    if dynamic_scan_range:
        startZ, endZ = ct.recommend_scan_range()

    ct.run_scan(startZ=startZ, endZ=endZ, views=views, mA=mA, kVp=kVp)
    ct.run_recon(fov=fov)
    dicom_path = output_dir / 'dicoms'
    dcm_files = ct.write_to_dicom(dicom_path / f'{patient_name}.dcm')

    lesion_only = ct
    mask = ct.get_lesion_mask(lesion_image,
                              startZ=startZ, endZ=endZ)
    vol_ml = np.prod(lesion_only.spacings) * mask.sum() / 1000

    lesion_only.recon = mask
    dicom_path = output_dir / 'lesion_masks'
    mask_files = lesion_only.write_to_dicom(dicom_path / f'{patient_name}_mask.dcm')

    for f, m in zip(dcm_files, mask_files):
        names.append(patient_name)
        ages.append(age)
        files.append(f)
        masks.append(m)
        contrast_list.append(contrast)
        radius_list.append(radius)
        center_x_list.append(lesion_coords[0])
        center_y_list.append(lesion_coords[1])
        center_z_list.append(lesion_coords[2])
        lesion_volume_list.append(vol_ml)

    case_count += 1
    metadata = pd.DataFrame({'name': names,
                             'age': ages,
                             'contrast': contrast_list,
                             'radius': radius_list,
                             'center x': center_x_list,
                             'center y': center_y_list,
                             'center z': center_z_list,
                             'lesion volume [mL]': lesion_volume_list,
                             'image file': files,
                             'mask file': masks})
    metadata.to_csv(output_directory / 'metadata.csv', index=False)
