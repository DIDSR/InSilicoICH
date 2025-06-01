'''
pipeline: this high level module organizes the healthy head phantoms,
lesion definitions, augmentations, and CT simulation together into the final
ct_simulation function.
'''
import os
from pathlib import Path
from shutil import rmtree
import numpy as np
import ast
import pydicom
import SimpleITK as sitk
import pandas as pd
from scipy.ndimage import center_of_mass
from monai.transforms import RandAffine

from VITools import Phantom, Scanner, Study, get_available_phantoms, read_dicom

from .phantoms.head_phantoms import LesionPhantom

def load_vol(file_list):
    return np.stack(list(map(read_dicom, file_list)))


def load_phantom(name='Densitometry Phantom', shape=None):
    '''
    Loads appropriate phantom based on age as a keyword

    :param name: phantom name, if a head phanton this is patient age in years, MIDA currently hard coded at 38 yrs
        see `ground_truth_definitions.phantoms.possible_ages` for ages
    :param shape: shape of that the ground truth phantom will be interpolated
    :param name: patient name to be saved in DICOM header
    '''
    available_phantoms = get_available_phantoms()
    matrix_size = max(shape) if shape else 400
    if name in available_phantoms:
        phantom_cls = available_phantoms[name]
        if name.endswith('Head'):  # add UNC, NIHPD to phantomdir
            phantom = phantom_cls(shape=shape)
        else:
            phantom = phantom_cls(matrix_size=matrix_size)
    elif isinstance(name, str) and Path(name).exists():
        img = sitk.ReadImage(name)
        phantom = Phantom(sitk.GetArrayFromImage(img),
                          spacings=img.GetSpacing()[::-1])
    elif isinstance(name, float | int):
        name = [o for o in available_phantoms.keys() if o.startswith(str(name))][0]
        phantom_cls = available_phantoms[name]
        phantom = phantom_cls(shape=shape)
    else:
        raise ValueError(f'{name} is not in {list(available_phantoms.keys())} nor is it a path')
    return phantom


LESION_TYPES = ['IPH', 'EDH', 'SDH']


class ICHStudy(Study):

    def generate_from_distributions(Phantoms: list[str],
                                    StudyCount: int = 1,
                                    Subtype: list[str] = [None] + LESION_TYPES,
                                    LesionVolume=dict(zip(LESION_TYPES,
                                                      len(LESION_TYPES)*[[0.1, 60]])),
                                    LesionAttenuation=dict(zip(LESION_TYPES,
                                                            len(LESION_TYPES)*[[0, 90]])),
                                    Edema=[0, 15],
                                    MassEffect=True,
                                    **kwargs):

        base_df = Study.generate_from_distributions(Phantoms,
                                                    StudyCount,
                                                    **kwargs)
        random = np.random.default_rng(base_df['GlobalSeed'].iloc[0])

        if isinstance(LesionVolume, dict):
            temp_volume = pd.DataFrame({f'{name}_volume': np.linspace(min_max[0], min_max[1]) for name, min_max in LesionVolume.items()})
            temp_weight = pd.DataFrame({f'{name}_weight': len(temp_volume)*[1/len(temp_volume)] for name in LesionVolume})
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
            temp_atten = pd.DataFrame({f'{name}_HU': np.linspace(min_max[0], min_max[1]) for name, min_max in LesionAttenuation.items()})
            temp_weight = pd.DataFrame({f'{name}_weight': len(temp_atten)*[1/len(temp_volume)] for name in LesionAttenuation})
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
            raise ValueError(f'`attenuation` {type(LesionAttenuation)} is not a dict\
            or csv filepath')

        edema_list = list(range(*Edema))  # IPH only

        params = {
            'Age': [],
            'LesionAttenuation(HU)': [],
            'Subtype': [],
            'LesionVolume(mL)': [],
            'Edema': [],
            'MassEffect': [],
            }

        for i in range(StudyCount):
            phantom_class = get_available_phantoms()[base_df['Phantom'].iloc[i]]
            lesion_id = None
            if hasattr(phantom_class, 'func') and issubclass(phantom_class.func, LesionPhantom):
                lesion_id = random.choice(Subtype)  # select a random lesion type
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
            params['LesionAttenuation(HU)'].append(float(intensity))
            params['Subtype'].append(lesion_id)
            params['LesionVolume(mL)'].append(vol)
            params['Edema'].append(edema)
            params['MassEffect'].append(MassEffect)

        ich_df = pd.DataFrame(params)
        input_df = base_df.join(ich_df)
        return input_df

    def append(self, *args,
               Subtype: str | None = None,
               LesionVolume: float=5,
               LesionAttenuation: float = 80,
               Edema:int=1,
               MassEffect=True,
               **kwargs):
        super()(*args, **kwargs)
        ## add here

    def run_study(self, patientid: int = 0):
        # work on the rest below....
        patient_name = self.phantom.patient_name
        age = self.phantom.age
        lesion_type = self.phantom.lesion_type if hasattr(self.phantom, 'lesion_type') else None
        intensity = self.phantom.lesion_intensity if hasattr(self.phantom, 'lesion_intensity') else None

        ct = self.scanner
        if isinstance(zspan, float):
            if np.isnan(zspan):
                zspan = 'dynamic'
        if isinstance(zspan, str):
            if zspan == 'dynamic':
                startZ, endZ = ct.recommend_scan_range()
            if zspan.startswith('['):  # using 'recruit' to create .csv can result in zspan being string of list
                zspan = ast.literal_eval(zspan)  # convert from 'string of list' to just list
                startZ, endZ = zspan
        elif isinstance(zspan, tuple | list):
            startZ, endZ = zspan
        views = int(views)
        ct.run_scan(startZ=startZ, endZ=endZ, views=views,
                    mA=mA, kVp=kVp, pitch=pitch)
        ct.run_recon(fov=fov, kernel=kernel,
                     sliceThickness=slice_thickness,
                     sliceIncrement=slice_increment)
        self.scanner = ct
        self.images = ct.recon
        if output_directory is None:
            output_directory = self.scanner.output_dir
        else:
            output_directory = Path(output_directory) / patient_name
        dicom_path = output_directory / 'dicoms'
        dcm_files = ct.write_to_dicom(dicom_path / f'{patient_name}.dcm')

        mask_files = [None]*len(dcm_files)
        z, x, y = 3*[None]
        vol_by_slice_mL = [0]*len(dcm_files)
        vol_ml = 0
        if lesion_type:
            lesion_only = ct
            mask = ct.get_lesion_mask(startZ=startZ, endZ=endZ,
                                      slice_thickness=slice_thickness, fov=fov)

            lesion_only.recon = mask
            dicom_path = output_directory / 'lesion_masks'
            mask_files = lesion_only.write_to_dicom(dicom_path /
                                                    f'{patient_name}_mask.dcm')
            mask = load_vol(mask_files)
            self.lesion = mask & (self.images > self.images.mean())
            self.scanner.recon = self.images

            dcm = pydicom.dcmread(mask_files[0])
            spacings = list(map(float, [dcm.SliceThickness] +
                            list(dcm.PixelSpacing)))

            vol_ml = np.prod(spacings) * mask.sum() / 1000
            vol_by_slice_mL = np.prod(spacings) *\
                self.lesion.sum(axis=(1, 2)) / 1000
            z, x, y = center_of_mass(mask)
            self.lesion_coords = (z, x, y)
        ages = []
        names = []
        files = []
        kVps = []
        mA_list = []
        fovs = []
        kernels = []
        views_list = []
        masks = []
        intensity_list = []
        lesion_type_list = []
        mass_effect = []
        center_x_list = []
        center_y_list = []
        center_z_list = []
        lesion_volume_list = []

        for f, m, vol_ml in zip(dcm_files, mask_files, vol_by_slice_mL):
            names.append(patient_name)
            ages.append(age)
            files.append(f)
            kVps.append(kVp)
            mA_list.append(mA)
            fovs.append(fov)
            kernels.append(kernel)
            views_list.append(views)
            masks.append(m)

            if vol_ml > 0:
                slice_mass_effect = self.phantom.mass_effect
                slice_intensity = intensity
                slice_x = int(x)
                slice_y = int(y)
                slice_z = int(z)
                slice_type = lesion_type
            else:
                slice_mass_effect = None
                slice_intensity = None
                slice_x = None
                slice_y = None
                slice_z = None
                slice_type = None

            intensity_list.append(slice_intensity)
            lesion_type_list.append(slice_type)
            mass_effect.append(slice_mass_effect)
            center_x_list.append(slice_x)
            center_y_list.append(slice_y)
            center_z_list.append(slice_z)
            lesion_volume_list.append([float(vol_ml)])

        metadata = pd.DataFrame({'Name': names,
                                 'Age': ages,
                                 'kVp': kVps,
                                 'mA': mA_list,
                                 'Views': views_list,
                                 'ReconKernel': kernels,
                                 'SliceThickness(mm)': slice_thickness,
                                 'LesionAttenuation(HU)': intensity_list,
                                 'LesionVolume(mL)': lesion_volume_list,
                                 'Subtype': lesion_type_list,
                                 'MassEffect': mass_effect,
                                 'CenterX': center_x_list,
                                 'CenterY': center_y_list,
                                 'CenterZ': center_z_list,
                                 'FOV(mm)': fovs,
                                 'ImageFilePath': files,
                                 'MaskFilePath': masks})
        self.metadata = metadata
        return self


def run_study(output_directory=None, patient_name=None, scanner_model='Scanner_Default', age=6.5, kVp=120,
              mA=200, pitch=0, intensity=200, volume=5, lesion_type=None,
              mass_effect=True, add_positioning_augmentation=True,
              views=1000, zspan='dynamic', kernel='standard',
              slice_thickness=1, slice_increment=None, keep_raw=False, seed=None, **kwargs) -> Study:

    phantom = load_phantom(age)
    if patient_name:
        phantom.patient_name = patient_name
    if lesion_type and (volume > 0) and hasattr(phantom, 'insert_lesion'):
        phantom.insert_lesion(lesion_type,
                              volume=volume,
                              intensity=intensity,
                              mass_effect=mass_effect,
                              seed=seed,
                              **kwargs)
    if os.name == 'nt':
       add_positioning_augmentation = False  # windows compatibility, monai transform crashes windows kernel
    if add_positioning_augmentation:
        transform = RandAffine(prob=1,
                               rotate_range=[np.pi/4, np.pi/20, np.pi/20],
                               translate_range=[10, 10, 10],
                               scale_range=[0.1, 0.1, 0.1],
                               padding_mode="border",
                               mode='nearest')
        if hasattr(phantom, 'apply_transform'):
            phantom.apply_transform(transform, seed=seed)

    scanner = Scanner(phantom, scanner_model=scanner_model, output_dir=output_directory)
    study = Study(scanner, 'pilot')
    study.run_study(kVp=kVp, mA=mA, pitch=pitch, views=views, zspan=zspan,
                    kernel=kernel, slice_thickness=slice_thickness,
                    slice_increment=slice_increment)
    study.metadata['CaseSeed'] = seed
    if keep_raw is False:
        rmtree(study.scanner.output_dir / 'phantoms')
        rmtree(study.scanner.output_dir / 'simulations')
    return study
