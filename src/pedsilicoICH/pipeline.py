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

from argparse import ArgumentParser
import os
from pathlib import Path

import pandas as pd

from pedsilicoICH.study import run_study

LESION_TYPES = ['round', 'epidural', 'subdural']


def pedsilicoich(input_csv, keep_raw=False):

    params = pd.read_csv(input_csv)
    n_params = len(params)

    try:
        patientids = [int(os.environ['SLURM_ARRAY_TASK_ID']) - 1]
    except KeyError:
        print('SLURM_ARRAY_TASK_ID not set, running in serial')
        patientids = list(range(n_params))

    for patientid in patientids:
        patient = params.iloc[patientid]
        print(f'{patientid+1}/{n_params}')

        patient_name = f'case_{patientid:03}'
        study = run_study(patient['output_directory'],
                          patient_name,
                          age=float(patient['age']),
                          kVp=float(patient['kVp']),
                          mA=float(patient['mA']),
                          intensity=float(patient['intensity']),
                          volume=float(patient['volume']),
                          lesion_type=patient['subtype'],
                          mass_effect=patient['mass_effect'],
                          views=patient['views'],
                          zspan=patient['zspan'],
                          kernel=patient['kernel'],
                          slice_thickness=patient['slice_thickness'],
                          keep_raw=keep_raw,
                          edema=int(float(patient['edema'])),
                          seed=int(patient['seed']))
        study.metadata['edema'] = int(float(patient['edema']))
        study.metadata.to_csv(Path(patient['output_directory']) / patient_name /
                              f'metadata_{patientid}.csv',
                              index=False)


def flatten_dict(layered_dict):
    config = dict()
    [config.update(k) for k in layered_dict.values()]
    return config


def pedsilicoich_cli():
    parser = ArgumentParser(
        description='Runs XCIST CT simulations of ICH models',
        epilog='''
        arguments can be given as toml config files or command line
        flags, each overriding defaults
        ''',
        fromfile_prefix_chars='@')
    parser.add_argument('input_csv', nargs='?', type=str,
                        help='input csv to recreate prior dataset')
    parser.add_argument('--keep_raw', type=bool,
                        help='''
                        whether to keep raw projection data and ground
                        truth phantoms, greatly increases
                        storage requirements.
                        ''')
    args = parser.parse_args()
    pedsilicoich(args.input_csv, args.keep_raw)


if __name__ == '__main__':
    pedsilicoich_cli()
