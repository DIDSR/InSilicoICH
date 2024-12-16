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
import tomllib
import os

import numpy as np
import pandas as pd

from pedsilicoICH.study import run_study
from pedsilicoICH.ground_truth_definition.phantoms import possible_ages

LESION_TYPES = ['round', 'epidural', 'subdural']


def pedsilicoich(output_directory, views=1000, desired_cases=1,
                 zspan='dynamic', age=[0, 100],
                 subtypes=[None] + LESION_TYPES,
                 mass_effect=True,
                 edema=[0, 15],
                 volume=dict(zip(LESION_TYPES,
                                 len(LESION_TYPES)*[[0.1, 60]])),
                 attenuation=dict(zip(LESION_TYPES,
                                  len(LESION_TYPES)*[[0, 90]])),
                 kVp=[120],
                 mA=[300],
                 kernel='soft',
                 slice_thickness=5,
                 keep_raw=False, seed=None):
    # load volume and HU distributions
    output_directory = Path(output_directory)
    assert (zspan == 'dynamic') or isinstance(zspan, list)
    if isinstance(zspan, list):
        if len(zspan) < 2:
            zspan = zspan[0].split(' ')
    if isinstance(zspan, list):
        zspan = list(map(int, zspan))
        for o in zspan:
            assert isinstance(o, int | float)
    if isinstance(volume, dict):
        df_volume = pd.DataFrame(volume).rename({'subdural': 'SDH_weight',
                                                 'epidural': 'EDH_weight',
                                                 'round': 'IPH_weight'},
                                                axis='columns')
    elif isinstance(volume, str | Path):
        df_volume = pd.read_csv(volume)
        df_volume['EDH_weight'] /= df_volume['EDH_weight'].sum()
        df_volume['SDH_weight'] /= df_volume['SDH_weight'].sum()
        df_volume['IPH_weight'] /= df_volume['IPH_weight'].sum()
    else:
        raise ValueError(f'`volume` {type(volume)} is not a dict\
or csv filepath')

    if isinstance(attenuation, dict):
        df_volume = pd.DataFrame(volume).rename({'subdural': 'SDH_weight',
                                                 'epidural': 'EDH_weight',
                                                 'round': 'IPH_weight'},
                                                axis='columns')
    elif isinstance(attenuation, str | Path):
        df_HU = pd.read_csv(attenuation).rename({'subdural': 'SDH_weight',
                                                 'epidural': 'EDH_weight',
                                                 'round': 'IPH_weight'},
                                                axis='columns')
        df_HU['EDH_weight'] /= df_HU['EDH_weight'].sum()
        df_HU['SDH_weight'] /= df_HU['SDH_weight'].sum()
        df_HU['IPH_weight'] /= df_HU['IPH_weight'].sum()
    else:
        raise ValueError(f'`attenuation` {type(attenuation)} is not a dict\
or csv filepath')

    ages = [yr for yr in possible_ages if (yr > age[0]) and (yr < age[1])]
    kVp_list = kVp if isinstance(kVp, list | tuple) else [kVp]
    mA_list = kVp if isinstance(mA, list | tuple) else [mA]
    edema_list = list(range(*edema))  # IPH only
    random = np.random.default_rng(seed)
    seed = random.integers(0, 1e6)
    l_parameter_comb = []

    for case_idx in range(desired_cases):
        lesion_id = random.choice(subtypes)  # select a random lesion type
        edema = None
        if lesion_id is None:
            vol = 0
            intensity = 0
        elif lesion_id == 'epidural':
            vol = random.choice(df_volume['EDH_volume'],
                                p=df_volume['EDH_weight'])
            intensity = random.choice(df_HU['EDH_HU'],
                                      p=df_HU['EDH_weight'])
        elif lesion_id == 'subdural':
            vol = random.choice(df_volume['SDH_volume'],
                                p=df_volume['SDH_weight'])
            intensity = random.choice(df_HU['SDH_HU'],
                                      p=df_HU['SDH_weight'])
        elif lesion_id == 'round':
            vol = random.choice(df_volume['IPH_volume'],
                                p=df_volume['IPH_weight'])
            intensity = random.choice(df_HU['IPH_HU'],
                                      p=df_HU['IPH_weight'])
            edema = random.choice(edema_list)

        l_parameter_comb.append([
            random.choice(ages),  # age
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
                          patient_name,
                          age=age,
                          kVp=kVp,
                          mA=mA,
                          intensity=intensity,
                          volume=volume,
                          lesion_type=lesion_type,
                          mass_effect=mass_effect,
                          views=views,
                          zspan=zspan,
                          kernel=kernel,
                          slice_thickness=slice_thickness,
                          keep_raw=keep_raw,
                          edema=edema,
                          seed=seed)
        study.metadata['edema'] = edema
        study.metadata.to_csv(output_directory / patient_name /
                              f'metadata_{patientid}.csv',
                              index=False)


def flatten_dict(layered_dict):
    config = dict()
    [config.update(k) for k in layered_dict.values()]
    return config


def pedsilicoich_cli():
    parser = ArgumentParser(
        description='Runs XCIST CT simulations of ICH models',
        epilog='arguments can be given as toml config files or command line flags, each overriding defaults',
        fromfile_prefix_chars='@')
    parser.add_argument('config', nargs='?', type=str,
                        help='Config toml file')
    parser.add_argument('--output_directory', type=str,
                        help='output directory to save simulation results')
    parser.add_argument('--views', type=int,
                        help='number of angular CT views per rotation')
    parser.add_argument('--desired_cases', type=int,
                        help='number of simulations to run')
    parser.add_argument('--zspan', nargs='+',
                        help='z range of scans [mm], defaults to dynamic')
    parser.add_argument('--keep_raw', type=bool,
                        help='whether to keep raw projection data and ground\
                        truth phantoms, greatly increases\
                        storage requirements.')
    parser.add_argument('--seed', type=int, help='seed to reproduce a dataset')
    args = parser.parse_args()
    with open(Path(__file__).parent / 'configs/default.toml', 'rb') as f:
        config = tomllib.load(f)
        config = flatten_dict(config)
    if args.config:
        with open(args.config, 'rb') as f:
            user_config = tomllib.load(f)
            user_config = flatten_dict(user_config)
        args.config = None
        config.update(user_config)

    cli_args = vars(args)
    cli_args = {k: v for k, v in cli_args.items() if v}
    config.update(cli_args)
    pedsilicoich(**config)


if __name__ == '__main__':
    pedsilicoich_cli()



