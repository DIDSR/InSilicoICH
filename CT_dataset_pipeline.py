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
import os

import numpy as np
import pandas as pd

from pedsilicoICH.pipeline import run_study


if __name__ == "__main__":
    parser = ArgumentParser(
        description='Runs XCIST CT simulations of ICH models')
    parser.add_argument('--output_directory', type=str, default="",
                        help='output directory to save simulation results')
    parser.add_argument('--views', type=int, default=1000,
                        help='number of angular CT views per rotation')
    parser.add_argument('--desired_cases', type=int, default=1000,
                        help='number of simulations to run')
    parser.add_argument('--zspan', nargs='+', default='dynamic',
                        help='z range of scans [mm], defaults to dynamic')
    parser.add_argument('--seed', type=int, help='seed to reproduce a dataset')
    args = parser.parse_args()

    zspan = args.zspan
    if isinstance(zspan, list):
        zspan = list(map(int, zspan))
    output_directory = Path(args.output_directory)
    # </https://www.aapm.org/pubs/CTProtocols/documents/PediatricRoutineHeadCT.pdf>
    # find parameter
# %%
    # load volume and HU distributions
    try:
        print(os.getcwd())
        df_volume = pd.read_csv(
            'src/pedsilicoICH/distributions/BHSD_volume_distributions.csv')
        df_HU = pd.read_csv(
            'src/pedsilicoICH/distributions/BHSD_HU_distributions.csv')
        print('Successfully loaded volume and HU distributions')
    except FileNotFoundError:
        min_vol, max_vol = 1, 60
        volume_list = np.linspace(min_vol, max_vol, 20)
        min_intensity, max_intensity = 20, 200
        intensity_list = np.arange(20, 200)

    recon_kernel = 'soft'
    # options include ['standard', 'soft', 'bone', 'R-L', 'S-L']
    slice_thickness = 5  # in mm
    nihpd_ages = [6.5, 9.0, 10.5, 11.5, 12.0, 15.75]
    mida_age = 38  # median US adult age to represent MIDA
    possible_ages = nihpd_ages + [mida_age]
    kVp_list = [120]
    mA_list = list(range(300, 400, 50))
    lesion_types = [None, 'round', 'epidural', 'subdural']
    mass_effect = np.linspace(0, 1, 10)
    random = np.random.default_rng(args.seed)
    seed = random.randint(0, 1e6, size=1)[0]
    l_parameter_comb = []

    for case_idx in range(args.desired_cases):
        lesion_id = random.choice(lesion_types)  # select a random lesion type
        if lesion_id is None:
            vol = 0
            intensity = 0
        elif lesion_id == 'epidural':
            vol = random.choices(df_volume['EDH_volume'],
                                 weights=df_volume['EDH_weight'])[0]
            intensity = random.choices(df_HU['EDH_HU'],
                                       weights=df_HU['EDH_weight'])[0]
        elif lesion_id == 'subdural':
            vol = random.choices(df_volume['SDH_volume'],
                                 weights=df_volume['SDH_weight'])[0]
            intensity = random.choices(df_HU['SDH_HU'],
                                       weights=df_HU['SDH_weight'])[0]
        elif lesion_id == 'round':
            vol = random.choices(df_volume['IPH_volume'],
                                 weights=df_volume['IPH_weight'])[0]
            intensity = random.choices(df_HU['IPH_HU'],
                                       weights=df_HU['IPH_weight'])[0]

        l_parameter_comb.append([
            random.choice(possible_ages),  # age
            random.choice(kVp_list),  # kVp
            random.choice(mA_list),  # mA
            intensity,
            vol,
            lesion_id,
            random.choice(mass_effect),
        ])

    n_params = len(l_parameter_comb)

    try:
        patientids = [int(os.environ['SLURM_ARRAY_TASK_ID']) - 1]
    except KeyError:
        print('SLURM_ARRAY_TASK_ID not set, running in serial')
        patientids = list(range(n_params))

    for patientid in patientids:
        print(f'{patientid+1}/{n_params}')
        age, kVp, mA, intensity, volume, lesion_type, mass_effect =\
            l_parameter_comb[patientid]
        print(f'{age} years, {lesion_type}, {volume} volume, {intensity} HU')
        patient_name = f'case_{patientid:03}'
        study = run_study(output_directory,
                          patient_name, age=age,
                          kVp=kVp, mA=mA, intensity=intensity,
                          volume=volume,
                          lesion_type=lesion_type,
                          mass_effect=mass_effect,
                          views=args.views, zspan=args.zspan,
                          kernel=recon_kernel,
                          slice_thickness=slice_thickness,
                          seed=seed)
        study.metadata.to_csv(output_directory / patient_name /
                              f'metadata_{patientid}.csv',
                              index=False)
