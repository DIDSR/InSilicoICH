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
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from insilicoICH.study import run_study

LESION_TYPES = ['IPH', 'EDH', 'SDH']


def insilicoich(input_csv, output_directory=None, keep_raw=False):

    params = pd.read_csv(input_csv)
    n_params = len(params)

    try:
        patientids = [int(os.environ['SLURM_ARRAY_TASK_ID']) - 1]
    except KeyError:
        print('SLURM_ARRAY_TASK_ID not set, running in serial')
        patientids = list(range(n_params))

    for patientid in patientids:
        patient = params.iloc[patientid]
        output_directory = patient['OutputDirectory'] or output_directory
        output_directory = Path(output_directory)
        print(f'{patientid+1}/{n_params}')

        # for old .csv files:
        if 'Scanner' in patient:
            scanner_model = str(patient['Scanner'])
        else:
            scanner_model = "Scanner_Default"
        subtype = patient['Subtype']
        volume = patient['LesionVolume(mL)']
        attenuation = patient['LesionAttenuation(HU)']
        mass_effect = patient['MassEffect']
        edema = patient['Edema']
        if not isinstance(subtype, str):
            if np.isnan(subtype):
                subtype = None
                attenuation = None
                edema = None
                volume = None
                mass_effect = None
        seed = int(patient['CaseSeed']) if not np.isnan(patient['CaseSeed']) else None
        patient_name = f'case_{patientid:03}'
        study = run_study(output_directory,
                          patient_name,
                          scanner_model=scanner_model,
                          age=patient['Age'],
                          kVp=float(patient['kVp']),
                          mA=float(patient['mA']),
                          pitch=float(patient['Pitch']),
                          intensity=attenuation,
                          volume=volume,
                          lesion_type=subtype,
                          mass_effect=mass_effect,
                          views=patient['Views'],
                          zspan=patient['ScanCoverage'],
                          kernel=patient['ReconKernel'],
                          slice_thickness=patient['SliceThickness(mm)'],
                          keep_raw=keep_raw,
                          edema=edema,
                          seed=seed)
        study.metadata['Edema'] = patient['Edema']
        study.metadata.to_csv(output_directory / patient_name /
                              f'metadata_{patientid}.csv',
                              index=False)


def flatten_dict(layered_dict):
    config = dict()
    [config.update(k) for k in layered_dict.values()]
    return config


def insilicoich_cli():
    parser = ArgumentParser(
        description='Runs XCIST CT simulations of ICH models',
        epilog='''
        arguments can be given as toml config files or command line
        flags, each overriding defaults
        ''',
        fromfile_prefix_chars='@')
    parser.add_argument('input_csv', nargs='?', type=str,
                        help='''
                          input csv to recreate prior dataset,
                          see `recruit --help` for more details
                        ''')
    parser.add_argument('--output_directory', type=str,
                        help='optional save directory')
    parser.add_argument('--keep_raw', type=bool,
                        help='''
                        whether to keep raw projection data and ground
                        truth phantoms, greatly increases
                        storage requirements.
                        ''')
    args = parser.parse_args()
    if args.input_csv:
        input_csv = args.input_csv
    elif not sys.stdin.isatty():
        input_csv = sys.stdin.read().strip()
    else:
        parser.print_help()
    insilicoich(input_csv, args.keep_raw)


if __name__ == '__main__':
    insilicoich_cli()
