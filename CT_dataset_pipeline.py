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

import numpy as np

from pedsilicoICH.pipeline import run_study


if __name__ == "__main__":
    parser = ArgumentParser(description='Runs XCIST CT simulations of ICH models')
    parser.add_argument('--output_directory', type=str, default="",
                        help='output directory to save simulation results')
    parser.add_argument('--views', type=int, default=1000,
                        help='number of angular CT views per rotation')
    parser.add_argument('--desired_cases', type=int, default=1000,
                        help='number of simulations to run')    
    parser.add_argument('--zspan', nargs='+', default='dynamic',
                        help='z range of scans [mm], defaults to dynamic')
    args = parser.parse_args()

    output_directory = Path(args.output_directory)
    desired_cases = args.desired_cases
    zspan = args.zspan
    views = args.views
    # </https://www.aapm.org/pubs/CTProtocols/documents/PediatricRoutineHeadCT.pdf>
    # find parameter
# %%
    recon_kernel = 'soft'  # options include ['standard', 'soft', 'bone', 'R-L', 'S-L']
    slice_thickness = 5 # in mm
    nihpd_ages = [6.5, 9.0, 10.5, 11.5, 12.0, 15.75]
    mida_age = 38  # median US adult age to represent MIDA
    possible_ages = nihpd_ages + [mida_age]
    kVp_list = [120]
    mA_list = list(range(300, 400, 50))
    lesion_types = [None, 'sphere', 'epidural', 'subdural']
    min_vol, max_vol = 34, 34000  # applied only to spheres [units of voxels, TODO convert to mL or mm^3]
    min_contrast, max_contrast = 20, 200
    contrast_list = np.arange(20, 200)
    volume_list = np.linspace(min_vol, max_vol, 20)
    mass_effect = [True, False]
    simulations_list = list(range(1))  # increased for multiple scans
    l_parameter_comb = []
    for age_id in possible_ages:
        for kVp_id in kVp_list:
            for mA_id in mA_list:
                for contrast_id in contrast_list:
                    for vol_id in volume_list:
                        for lesion_id in lesion_types:
                            for mass_effect_id in mass_effect:
                                for simulation_id in simulations_list:
                                    l_parameter_comb.append([age_id,
                                                            kVp_id,
                                                            mA_id,
                                                            contrast_id,
                                                            vol_id,
                                                            lesion_id,
                                                            mass_effect_id,
                                                            simulation_id])
    shuffle(l_parameter_comb)
    l_parameter_comb = l_parameter_comb[:desired_cases]
    n_params = len(l_parameter_comb)

    try:
        patientids = [int(os.environ['SLURM_ARRAY_TASK_ID']) - 1]
    except:
        print('SLURM_ARRAY_TASK_ID not set, running in serial')
        patientids = list(range(n_params))

    for patientid in patientids:
        print(f'{patientid}/{n_params}')
        age, kVp, mA, contrast, volume, lesion_type, mass_effect, sim_id =\
            l_parameter_comb[patientid]
        print(f'{age} years, {lesion_type}, {contrast} HU')
        patient_name = f'case_{patientid:03}'
        study = run_study(output_directory,
                          patient_name, age=age,
                          kVp=kVp, mA=mA, contrast=contrast,
                          volume=volume,
                          lesion_type=lesion_type,
                          mass_effect=mass_effect,
                          views=views, zspan=zspan,
                          kernel=recon_kernel, slice_thickness=slice_thickness)
        study.metadata.to_csv(output_directory / patient_name /
                              f'metadata_{patientid}.csv',
                              index=False)
