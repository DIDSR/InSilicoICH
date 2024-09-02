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

from pathlib import Path
from argparse import ArgumentParser
from random import shuffle
import os

import pandas as pd
import numpy as np
from scipy.ndimage import center_of_mass
import pydicom

from pedsilicoICH.pipeline import ct_simulation
from pedsilicoICH.image_acquisition import read_dicom


def load_vol(file_list):
    return np.stack(list(map(read_dicom, file_list)))


output_directory = Path('/gpfs_projects/brandon.nelson/pedsilicoICH/parallel') # output directory to save simulation results
desired_cases = 100
views = 1000

def main(patientid, age, kVp, mA, contrast, radius, lesion_type, views=1000, zspan='dynamic'):
    print(patientid)
    patient_name = f'case_{patientid:03}'
    dcm_files, mask_files = ct_simulation(output_directory=output_directory,
                                          patient_name=patient_name,
                                          age=age,
                                          kVp=kVp,
                                          mA=mA,
                                          contrast=contrast,
                                          radius=radius,
                                          lesion_type=lesion_type,
                                          views=views,
                                          zspan=zspan)

    mask = load_vol(mask_files)
    dcm = pydicom.read_file(mask_files[0])
    spacings = dcm.PixelSpacing

    vol_ml = np.prod(spacings) * mask.sum() / 1000
    z, x, y = center_of_mass(mask)
    # define empty list of metadata columns to store in the resulting dataframe
    ages = []
    names = []
    files = []
    masks = []
    contrast_list = []
    radius_list = []
    lesion_type_list = []
    center_x_list = []
    center_y_list = []
    center_z_list = []
    lesion_volume_list = []

    for f, m in zip(dcm_files, mask_files):
        names.append(patient_name)
        ages.append(age)
        files.append(f)
        masks.append(m)
        contrast_list.append(contrast)
        radius_list.append(radius)
        lesion_type_list.append(lesion_type)
        center_x_list.append(x)
        center_y_list.append(y)
        center_z_list.append(z)
        lesion_volume_list.append(vol_ml)

    metadata = pd.DataFrame({'name': names,
                             'age': ages,
                             'contrast': contrast_list,
                             'radius': radius_list,
                             'center x': center_x_list,
                             'center y': center_y_list,
                             'center z': center_z_list,
                             'lesion type': lesion_type_list,
                             'lesion volume [mL]': lesion_volume_list,
                             'image file': files,
                             'mask file': masks})
    return metadata

# %%
if __name__ == "__main__":
    parser = ArgumentParser(description='Runs XCIST CT simulations of ICH models')
    parser.add_argument('--output_directory', type=str, default="", help='output directory to save simulation results')
    parser.add_argument('--views', type=int, default=1000, help='number of angular CT views per rotation')
    parser.add_argument('--desired_cases', type=int, default=1000, help='number of simulations to run')
    args = parser.parse_args()

    desired_cases = args.desired_cases
    # </https://www.aapm.org/pubs/CTProtocols/documents/PediatricRoutineHeadCT.pdf>
    # find parameter
# %%
    nihpd_ages = [6.5, 9.0, 10.5, 11.5, 12.0, 15.75]
    mida_age = 38  # add 38 as the median US adult age to represent MIDA, consider other identifiers when adding more patients
    possible_ages = nihpd_ages + [mida_age]
    kVp_list = [110, 120, 130]
    mA_list = list(range(50, 400, 50))
    lesion_types = ['sphere', 'epidural', 'subdural']
    min_radius, max_radius = 2, 20  # applied only to spheres, TODO turn to volume then can be more general to allow types
    min_contrast, max_contrast = 20, 200
    contrast_list = np.arange(20, 200)
    radii_list = np.arange(min_radius, max_radius)
    simulations_list = list(range(1)) # This can be increased to enable multiple scans (different noise realizations of the same slice and settings)
    l_parameter_comb = []
    for age_id in possible_ages:
        for kVp_id in kVp_list:
            for mA_id in mA_list:
                for contrast_id in contrast_list:
                    for radii_id in radii_list:
                        for lesion_id in lesion_types:
                            for simulation_id in simulations_list:
                                l_parameter_comb.append([age_id,
                                                         kVp_id,
                                                         mA_id,
                                                         contrast_id,
                                                         radii_id,
                                                         lesion_id,
                                                         simulation_id])
    shuffle(l_parameter_comb)
    l_parameter_comb = l_parameter_comb[:desired_cases]

    try:
        SGE_TASK_ID = int(os.environ['SGE_TASK_ID'])-1 # since tasks start from 1
        age, kVp, mA, contrast, radius, lesion_type, sim_id = l_parameter_comb[SGE_TASK_ID]
        metadata = main(patientid=SGE_TASK_ID, age=age,
                        kVp=kVp, mA=mA, contrast=contrast,
                        radius=radius,
                        lesion_type=lesion_type,
                        views=views, zspan=zspan)
        metadata.to_csv(output_directory / f'metadata_{SGE_TASK_ID}.csv', index=False)
    except:
        print('SGE_TASK_ID not set, running in serial')
        n_params = len(l_parameter_comb)
        for SGE_TASK_ID in range(n_params):
            print(f'{SGE_TASK_ID}/{n_params}')
            age, kVp, mA, contrast, radius, lesion_type, sim_id = l_parameter_comb[SGE_TASK_ID]
            metadata = main(patientid=SGE_TASK_ID, age=age,
                            kVp=kVp, mA=mA, contrast=contrast,
                            radius=radius,
                            lesion_type=lesion_type,
                            views=views, zspan=zspan)
            metadata.to_csv(output_directory / f'metadata_{SGE_TASK_ID}.csv', index=False)

# %%
