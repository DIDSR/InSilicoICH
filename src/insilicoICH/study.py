'''
pipeline: this high level module organizes the healthy head phantoms,
lesion definitions, augmentations, and CT simulation together into the final
ct_simulation function.
'''
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pydicom
import SimpleITK as sitk
import pandas as pd
import tomllib
from scipy.ndimage import center_of_mass
from monai.transforms import RandAffine

from VITools import Phantom, Study, get_available_phantoms, load_vol

from .phantoms.head_phantoms import LesionPhantom


def load_phantom(name='Densitometry Phantom', shape=None):  # this likely belongs in InSilicoGUI
    """
    Load a phantom object based on the provided name or file path.

    Parameters:
        name (str|float|int|Path): Name of the phantom, patient age, or file path to a phantom image.
        shape (tuple or list, optional): Desired shape for the phantom.

    Returns:
        Phantom: Loaded phantom object.

    Raises:
        ValueError: If the phantom name or path is not recognized.
    """
    available_phantoms = get_available_phantoms()
    matrix_size = max(shape) if shape else 400
    if name in available_phantoms:
        phantom_cls = available_phantoms[name]
        if name.endswith('Head'):
            phantom = phantom_cls(shape=shape)
        else:
            phantom = phantom_cls(matrix_size=matrix_size)
    elif isinstance(name, str) and Path(name).exists():
        img = sitk.ReadImage(name)
        phantom = Phantom(sitk.GetArrayFromImage(img),
                          spacings=img.GetSpacing()[::-1])
    elif isinstance(name, float | int):
        name = [o for o in available_phantoms.keys() if
                o.startswith(str(name))][0]
        phantom_cls = available_phantoms[name]
        phantom = phantom_cls(shape=shape)
    else:
        raise ValueError(f'{name} is not in {list(available_phantoms.keys())}\
                         nor is it a path')
    return phantom


LESION_TYPES = list(LesionPhantom.lesion_types)


class ICHStudy(Study):
    """
    Study class for generating and managing in silico ICH (Intracerebral Hemorrhage) datasets.
    Extends the base Study class with lesion simulation, augmentation, and mask generation.
    """
    def generate_from_distributions(Phantoms: list[str],
                                    StudyCount: int = 1,
                                    Subtype: list[str] = [None] + LESION_TYPES,
                                    LesionVolume=dict(
                                        zip(LESION_TYPES,
                                            len(LESION_TYPES)*[[0.1, 60]])
                                        ),
                                    LesionAttenuation=dict(
                                        zip(LESION_TYPES,
                                            len(LESION_TYPES)*[[0, 90]])
                                        ),
                                    Edema=[0, 15],
                                    MassEffect=True,
                                    AddAugmentation=True,
                                    **kwargs):
        """
        Generate a DataFrame of study parameters by sampling from specified distributions.

        Parameters:
            Phantoms (list[str]): List of phantom names to use.
            StudyCount (int): Number of studies to generate.
            Subtype (list[str]): List of lesion subtypes to sample.
            LesionVolume (dict|str|Path): Volume distributions or CSV for each lesion type.
            LesionAttenuation (dict|str|Path): Attenuation distributions or CSV for each lesion type.
            Edema (list[int]): Range for edema values (for IPH).
            MassEffect (bool): Whether to simulate mass effect.
            AddAugmentation (bool): Whether to apply augmentation transforms.
            **kwargs: Additional arguments for Study.generate_from_distributions.

        Returns:
            pd.DataFrame: DataFrame with study parameters for each case.
        """
        base_df = Study.generate_from_distributions(Phantoms,
                                                    StudyCount,
                                                    **kwargs)
        random = np.random.default_rng(base_df['GlobalSeed'].iloc[0])

        if isinstance(LesionVolume, dict):
            temp_volume = pd.DataFrame(
                {f'{name}_volume': np.linspace(min_max[0], min_max[1]) for
                 name, min_max in LesionVolume.items()})
            temp_weight = pd.DataFrame(
                {f'{name}_weight': len(temp_volume)*[1/len(temp_volume)] for
                 name in LesionVolume})
            df_volume = pd.concat([temp_volume, temp_weight], axis=1)
        elif isinstance(LesionVolume, str | Path):
            df_volume = pd.read_csv(LesionVolume)
            df_volume['EDH_weight'] /= df_volume['EDH_weight'].sum()
            df_volume['SDH_weight'] /= df_volume['SDH_weight'].sum()
            df_volume['IPH_weight'] /= df_volume['IPH_weight'].sum()
        else:
            raise ValueError(f'`volume` {type(LesionVolume)} is not a dict\
    or csv filepath')

        if isinstance(LesionAttenuation, dict):
            temp_atten = pd.DataFrame(
                {f'{name}_HU': np.linspace(min_max[0], min_max[1]) for
                 name, min_max in LesionAttenuation.items()})
            temp_weight = pd.DataFrame(
                {f'{name}_weight': len(temp_atten)*[1/len(temp_volume)] for
                 name in LesionAttenuation})
            df_HU = pd.concat([temp_atten, temp_weight], axis=1)
        elif isinstance(LesionAttenuation, str | Path):
            df_HU = pd.read_csv(LesionAttenuation).rename({
                                                    'subdural': 'SDH_weight',
                                                    'epidural': 'EDH_weight',
                                                    'round': 'IPH_weight'},
                                                    axis='columns')
            df_HU['EDH_weight'] /= df_HU['EDH_weight'].sum()
            df_HU['SDH_weight'] /= df_HU['SDH_weight'].sum()
            df_HU['IPH_weight'] /= df_HU['IPH_weight'].sum()
        else:
            raise ValueError(f'`attenuation` {type(LesionAttenuation)} is not\
                             a dict or csv filepath')

        edema_list = list(range(*Edema))  # IPH only

        params = {
            'Age': [],
            'LesionAttenuation': [],
            'Subtype': [],
            'LesionVolume': [],
            'Edema': [],
            'MassEffect': [],
            'AddAugmentation': []
            }

        for i in range(StudyCount):
            phantom_class = get_available_phantoms()[
                base_df['Phantom'].iloc[i]
                ]
            lesion_id = None
            if hasattr(phantom_class, 'func') and\
               issubclass(phantom_class.func, LesionPhantom):
                lesion_id = random.choice(Subtype)
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
                while vol > 50:
                    vol = random.choice(df_volume['IPH_volume'],
                                        p=df_volume['IPH_weight'])
                intensity = 0
                while intensity < 45:
                    intensity = random.choice(df_HU['IPH_HU'],
                                              p=df_HU['IPH_weight'])

                edema = random.choice(edema_list)

            age = phantom_class.keywords['age'] if\
                hasattr(phantom_class, 'keywords') and\
                ('age' in phantom_class.keywords) else 0
            params['Age'].append(age)
            params['LesionAttenuation'].append(float(intensity))
            params['Subtype'].append(lesion_id)
            params['LesionVolume'].append(vol)
            params['Edema'].append(edema)
            params['MassEffect'].append(MassEffect)
            params['AddAugmentation'].append(AddAugmentation)

        ich_df = pd.DataFrame(params)
        input_df = base_df.join(ich_df)
        return input_df

    def append(self, *args,
               Subtype: str | None = None,
               LesionVolume: float = 5,
               LesionAttenuation: float = 80,
               Edema: int = 1,
               MassEffect=True,
               AddAugmentation=True,
               **kwargs):
        """
        Append a new study case to the metadata with lesion and augmentation parameters.

        Parameters:
            *args: Arguments for the base append method.
            Subtype (str|None): Lesion subtype.
            LesionVolume (float): Lesion volume in mL.
            LesionAttenuation (float): Lesion attenuation in HU.
            Edema (int): Edema value.
            MassEffect (bool): Whether to simulate mass effect.
            AddAugmentation (bool): Whether to apply augmentation.
            **kwargs: Additional arguments for the base append method.

        Returns:
            ICHStudy: The updated study object.
        """
        super().append(*args, **kwargs)
        last_idx = len(self) - 1
        self.metadata.at[last_idx, 'Subtype'] = Subtype
        self.metadata.at[last_idx, 'LesionVolume'] = LesionVolume
        self.metadata.at[last_idx, 'LesionAttenuation'] = LesionAttenuation
        self.metadata.at[last_idx, 'Edema'] = Edema
        self.metadata.at[last_idx, 'MassEffect'] = MassEffect
        self.metadata.at[last_idx, 'AddAugmentation'] = AddAugmentation
        return self

    def load_phantom(self,  patientid: int = 0):
        series = self.metadata.iloc[patientid]
        phantom = super().load_phantom(patientid)
        if series.Subtype and ((series.LesionVolume > 0) and
                               hasattr(phantom, 'insert_lesion')):
            phantom.insert_lesion(series.Subtype,
                                  volume=series.LesionVolume,
                                  intensity=series.LesionAttenuation,
                                  mass_effect=series.MassEffect,
                                  seed=series.CaseSeed,
                                  edema=int(series.Edema))
        if os.name == 'nt':
            series.AddAugmentation = False
            # windows compatibility, monai transform crashes windows kernel
        if series.AddAugmentation:
            transform = RandAffine(prob=1,
                                   rotate_range=[np.pi/4, np.pi/20, np.pi/20],
                                   translate_range=[10, 10, 10],
                                   scale_range=[0.1, 0.1, 0.1],
                                   padding_mode="border",
                                   mode='nearest')
            if hasattr(phantom, 'apply_transform'):
                phantom.apply_transform(transform, seed=series.CaseSeed)
        return phantom

    def run_study(self, patientid: int = 0):
        results = super().run_study(patientid)
        series = self.metadata.iloc[patientid]
        mask_files = None
        lesion_coords = None
        vol_by_slice_mL = 0
        if series.Subtype:
            lesion_only = self.scanner
            startZ, endZ = self.scanner.ScanCoverage
            mask = self.scanner.get_lesion_mask(
                startZ=startZ, endZ=endZ,
                slice_thickness=series.SliceThickness,
                fov=series.FOV
                )
            lesion_only.recon = mask
            dicom_path = Path(series.OutputDirectory) / 'lesion_masks'
            patient_name = self.scanner.phantom.patient_name
            mask_files = lesion_only.write_to_dicom(dicom_path /
                                                    f'{patient_name}_mask.dcm')
            mask = load_vol(mask_files)
            lesion = mask & (self.scanner.recon > self.scanner.recon.mean())

            dcm = pydicom.dcmread(mask_files[0])
            spacings = list(map(float, [dcm.SliceThickness] +
                            list(dcm.PixelSpacing)))

            vol_by_slice_mL = np.prod(spacings) *\
                lesion.sum(axis=(1, 2)) / 1000
            z, x, y = center_of_mass(mask)
            lesion_coords = str(
                list(map(lambda o: round(float(o), ndigits=1), (z, x, y)))
                )

        rows = results.CaseID == f'case_{patientid:04d}'
        results.loc[rows, 'Subtype'] = series.Subtype
        results.loc[rows, 'LesionVolume(mL)'] = vol_by_slice_mL
        slice_intensity = np.zeros_like(vol_by_slice_mL)
        slice_intensity[vol_by_slice_mL > 0] =\
            self.scanner.phantom.lesion_intensity if\
            hasattr(self.scanner.phantom, 'lesion_intensity') else 0
        results.loc[rows, 'LesionAttenuation(HU)'] = slice_intensity
        results.loc[rows, 'MassEffect'] = series.MassEffect
        results.loc[rows, 'LesionLocation(z, y, x)'] = lesion_coords  # in voxels NOT mm, need to add a mm location too
        results.loc[rows, 'MaskFilePath'] = mask_files
        return results

    def get_masks(self, patientid: int = 0):
        """
        Retrieve the lesion mask volume(s) for a given patient.

        Parameters:
            patientid (int): Index of the patient/case.

        Returns:
            np.ndarray: Loaded mask volume(s).
        """
        return load_vol(self.results[self.results.CaseID ==
                                     f'case_{patientid:04d}']['MaskFilePath'])


def insilicoich_cli(arg_list: list[str] | None = None):
    '''
    Command-line interface for InSilicoICH simulations.

    Parses command-line arguments to specify an input CSV file for study
    parameters and an option to run simulations in parallel. If no input CSV
    is provided via arguments, it attempts to read from stdin.

    The input CSV can define a study to be run. This function initializes a
    `Study` object with this CSV and then calls its `run_all` method.

    Args:
        arg_list (list[str] | None, optional):
            A list of command-line arguments to parse. If None, `sys.argv[1:]`
            is used.
            Defaults to None.
    '''
    parser = ArgumentParser(
        description='Runs InSilicoICH simulations',
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
    parser.add_argument('--parallel', '-p', type=bool,
                        default=False,
                        help='run simulations in parallel')
    args = parser.parse_args(arg_list)
    if args.input_csv:
        input_csv = args.input_csv
    elif not sys.stdin.isatty():
        input_csv = sys.stdin.read().strip()
    else:
        parser.print_help()

    ICHStudy(input_csv).run_all(args.parallel)


def recruit_patients(OutputDirectory, **config):
    OutputDirectory = Path(OutputDirectory)
    config['OutputDirectory'] = OutputDirectory
    age_range = config.pop('Age')
    phantoms = get_available_phantoms()
    patients = {k: v for k, v in phantoms.items() if hasattr(v, 'keywords')}
    patients = {k: v for k, v in patients.items() if 'age' in v.keywords}
    patients = {k: v for k, v in patients.items() if
                (v.keywords['age'] > age_range[0]) and
                (v.keywords['age'] < age_range[1])}

    df = ICHStudy.generate_from_distributions(patients, **config)
    save_name = OutputDirectory / (OutputDirectory.name + '.csv')
    save_name.parent.mkdir(exist_ok=True, parents=True)
    print(save_name)
    df.to_csv(save_name, index=False)


def flatten_dict(layered_dict):
    config = dict()
    [config.update(k) for k in layered_dict.values()]
    return config


def recruitment_cli(arg_list: list[str] | None = None):
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
    parser.add_argument('--OutputDirectory', '-o', type=str, default='results',
                        help='output directory to save simulation results')
    parser.add_argument('--input_csv', '-i', type=str,
                        help='input csv to recreate prior dataset')
    parser.add_argument('--Views', type=int,
                        help='number of angular CT views per rotation')
    parser.add_argument('--StudyCount', type=int,
                        help='number of simulations to run')
    parser.add_argument('--ScanCoverage', nargs='+',
                        help='z range of scans [mm], defaults to dynamic')
    parser.add_argument('--RemoveRawData', type=bool, default=True,
                        help='''
                        whether to keep raw projection data and ground
                        truth phantoms, greatly increases
                        storage requirements.
                        ''')
    parser.add_argument('--Seed', type=int, help='seed to reproduce a dataset')
    args = parser.parse_args(arg_list)
    pkg_dir = Path(__file__).parent
    with open(pkg_dir / 'configs/default.toml', 'rb') as f:
        config = tomllib.load(f)
        config = flatten_dict(config)
        config['LesionVolume'] = pkg_dir / config['LesionVolume']
        config['LesionAttenuation'] = pkg_dir / config['LesionAttenuation']
    if args.config:
        with open(args.config, 'rb') as f:
            user_config = tomllib.load(f)
            user_config = flatten_dict(user_config)
        args.config = None
        config.update(user_config)

    cli_args = vars(args)
    cli_args = {k: v for k, v in cli_args.items() if v}
    config.update(cli_args)
    config['Subtype'] = list(map(lambda o: o or None, config['Subtype']))
    recruit_patients(**config)


if __name__ == '__main__':
    insilicoich_cli()
