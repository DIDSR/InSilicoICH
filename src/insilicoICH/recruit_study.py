# %%
from pathlib import Path
from argparse import ArgumentParser

import tomllib
import pandas as pd
import numpy as np

from insilicoICH.ground_truth_definition.phantoms import possible_ages

# Definitions: IPH/intraparenchymal , EDH/epidural, SDH/subdural
LESION_TYPES = ['IPH', 'EDH', 'SDH'] 


def recruit_patients(output_directory, views=[1000], desired_cases=1,
                     zspan='dynamic', age=[0, 100],
                     subtypes=[None] + LESION_TYPES,
                     mass_effect=True,
                     edema=[0, 15],
                     volume=dict(zip(LESION_TYPES,
                                     len(LESION_TYPES)*[[0.1, 60]])),
                     attenuation=dict(zip(LESION_TYPES,
                                      len(LESION_TYPES)*[[0, 90]])),
                     scanner='Scanner_Default',
                     kVp=[120],
                     mA=[300],
                     save_name=None,
                     kernel=['soft'],
                     slice_thickness=[1],
                     keep_raw=False, seed=None):

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

    ages = [yr for yr in possible_ages if (yr >= min(age)) & (yr <= max(age))]
    kVp_list = kVp if isinstance(kVp, list | tuple) else [kVp]
    mA_list = mA if isinstance(mA, list | tuple) else [mA]
    edema_list = list(range(*edema))  # IPH only

    if isinstance(seed, float):
        raise ValueError('seed cannot be float, set to False or integer')
    elif (not seed) & isinstance(seed, bool):  # check if seed is bool and False
        random = np.random.default_rng()
    elif seed & isinstance(seed, bool):  # check if seed is bool and True
        raise ValueError('seed cannot be True, set to False or integer')
    elif isinstance(seed, int):  # if not True or False, check if int:
        random = np.random.default_rng(seed)
    else:
        raise ValueError('seed must be False or integer')

    global_seed = random.integers(0, 1e6)

    params = {
        'Age': [],
        'Scanner': [],
        'kVp': [],
        'mA': [],
        'Views': [],
        'ScanCoverage': [],
        'ReconKernel': [],
        'SliceThickness(mm)': [],
        'LesionAttenuation(HU)': [],
        'Subtype': [],
        'LesionVolume(mL)': [],
        'Edema': [],
        'MassEffect': [],
        'GlobalSeed': [],
        'CaseSeed': [],
        'OutputDirectory': []
    }

    for _ in range(desired_cases):
        lesion_id = random.choice(subtypes)  # select a random lesion type
        if lesion_id is None:
            vol = 0
            intensity = 0
            edema = 0
        elif lesion_id == 'EDH':
            vol = random.choice(df_volume['EDH_volume'],
                                p=df_volume['EDH_weight'])
            intensity = 0
            while intensity < 45:
                intensity = random.choice(df_HU['EDH_HU'],
                                          p=df_HU['EDH_weight'])
            edema = 0
        elif lesion_id == 'SDH':
            vol = random.choice(df_volume['SDH_volume'],
                                p=df_volume['SDH_weight'])
            intensity = 0
            while intensity < 45:
                intensity = random.choice(df_HU['SDH_HU'],
                                          p=df_HU['SDH_weight'])
            edema = 0
        elif lesion_id == 'IPH':
            vol = random.choice(df_volume['IPH_volume'],
                                p=df_volume['IPH_weight'])
            while vol > 25:
                vol = random.choice(df_volume['IPH_volume'],
                                    p=df_volume['IPH_weight'])
            intensity = 0
            while intensity < 45:
                intensity = random.choice(df_HU['IPH_HU'],
                                          p=df_HU['IPH_weight'])

            edema = random.choice(edema_list)

        params['Age'].append(float(random.choice(ages)))
        params['Scanner'].append(random.choice(scanner))
        params['kVp'].append(float(random.choice(kVp_list)))
        params['mA'].append(float(random.choice(mA_list)))
        params['Views'].append(float(random.choice(views)))
        params['ScanCoverage'].append(zspan)
        params['ReconKernel'].append(random.choice(kernel))
        params['SliceThickness(mm)'].append(random.choice(slice_thickness))
        params['LesionAttenuation(HU)'].append(float(intensity))
        params['Subtype'].append(lesion_id)
        params['LesionVolume(mL)'].append(vol)
        params['Edema'].append(edema)
        params['MassEffect'].append(mass_effect)
        params['GlobalSeed'].append(global_seed)
        params['CaseSeed'].append(random.integers(0, 1e6))
        params['OutputDirectory'].append(output_directory)

    df = pd.DataFrame(params)
    save_name = save_name or output_directory / \
        (output_directory.name + '.csv')
    save_name.parent.mkdir(exist_ok=True, parents=True)
    print(save_name)
    df.to_csv(save_name, index=False)


def flatten_dict(layered_dict):
    config = dict()
    [config.update(k) for k in layered_dict.values()]
    return config


def recruitment_cli():
    parser = ArgumentParser(
        description='''Generates full patient list to conduct study from
          provided distributions.

          Output: a .csv file with scans to perform,
          the input for the `generate` command
        ''',
        epilog='''
        arguments can be given as toml config files or command line
        flags, each overriding defaults
        ''',
        fromfile_prefix_chars='@')
    parser.add_argument('config', nargs='?', type=str,
                        help='''Inclusion criteria config .toml file
                        specifying ranges of parameters that will be
                        uniformily randomly sample to generate a recruited
                        patient list for scanning with `generate`''')
    parser.add_argument('--output_directory', type=str,
                        help='output directory to save simulation results')
    parser.add_argument('--input_csv', type=str,
                        help='input csv to recreate prior dataset')
    parser.add_argument('--views', type=int,
                        help='number of angular CT views per rotation')
    parser.add_argument('--desired_cases', type=int,
                        help='number of simulations to run')
    parser.add_argument('--zspan', nargs='+',
                        help='z range of scans [mm], defaults to dynamic')
    parser.add_argument('--keep_raw', type=bool,
                        help='''
                        whether to keep raw projection data and ground
                        truth phantoms, greatly increases
                        storage requirements.
                        ''')
    parser.add_argument('--seed', type=int, help='seed to reproduce a dataset')
    args = parser.parse_args()
    pkg_dir = Path(__file__).parent
    with open(pkg_dir / 'configs/default.toml', 'rb') as f:
        config = tomllib.load(f)
        config = flatten_dict(config)
        config['volume'] = pkg_dir / config['volume']
        config['attenuation'] = pkg_dir / config['attenuation']
    if args.config:
        with open(args.config, 'rb') as f:
            user_config = tomllib.load(f)
            user_config = flatten_dict(user_config)
        args.config = None
        config.update(user_config)

    cli_args = vars(args)
    cli_args = {k: v for k, v in cli_args.items() if v}
    config.update(cli_args)
    config['subtypes'] = list(map(lambda o: o or None, config['subtypes']))
    recruit_patients(**config)


if __name__ == '__main__':
    recruitment_cli()
