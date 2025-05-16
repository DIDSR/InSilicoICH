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
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import center_of_mass
from monai.transforms import RandAffine

from .image_acquisition import Scanner, read_dicom
from .phantoms.base_phantoms import Phantom

import pluggy
from . import hooks  # Your hooks.py


def get_phantoms_dict():
    pm = pluggy.PluginManager(hooks.PROJECT_NAME)
    pm.add_hookspecs(hooks.PhantomSpecs)
    num_loaded = pm.load_setuptools_entrypoints(group=hooks.PROJECT_NAME)

    # --- Call the hook to get all registered phantom types ---
    # The hook returns a list of lists (one list per plugin implementation that returned something)
    list_of_results = pm.hook.register_phantom_types()
    # Flatten the list of lists and filter out None or empty lists from plugins
    discovered_phantom_classes = {}
    for result_list in list_of_results:
        if result_list:  # Check if the plugin returned a non-empty list
            discovered_phantom_classes.update(result_list)
    return discovered_phantom_classes


def load_vol(file_list):
    return np.stack(list(map(read_dicom, file_list)))


available_phantoms = get_phantoms_dict()


def load_phantom(name='Densitometry Phantom', shape=None):
    '''
    Loads appropriate phantom based on age as a keyword

    :param name: phantom name, if a head phanton this is patient age in years, MIDA currently hard coded at 38 yrs
        see `ground_truth_definitions.phantoms.possible_ages` for ages
    :param shape: shape of that the ground truth phantom will be interpolated
    :param name: patient name to be saved in DICOM header
    '''

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


class Study:
    def __init__(self, scanner: Scanner, study_name='default'):
        self.scanner = scanner
        self.phantom = scanner.phantom
        self.study_name = study_name
        self.metadata = None

    def __repr__(self) -> str:
        repr = f'''
        study name: {self.study_name}
        Phantom details:
        ----------------
        {self.scanner.phantom.__repr__()}

        Scanner details:
        ----------------
        Scanner: {self.scanner.__repr__()}

        Study details:
        --------------
        {self.metadata}
        '''
        return repr

    @property
    def shape(self):
        return list(self.phantom._phantom.shape)

    @property
    def size(self):
        return np.array(self.phantom.spacings)*self.phantom._phantom.shape

    def run_study(self, output_directory=None, kVp=120, mA=200, pitch=0, views=1000,
                  fov=250, zspan='dynamic',
                  kernel='standard', slice_thickness=1, **kwargs):
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
        ct.run_recon(fov=fov, kernel=kernel, sliceThickness=slice_thickness)
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
              slice_thickness=1, keep_raw=False, seed=None, **kwargs) -> Study:

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
                    kernel=kernel, slice_thickness=slice_thickness)
    study.metadata['CaseSeed'] = seed
    if keep_raw is False:
        rmtree(study.scanner.output_dir / 'phantoms')
        rmtree(study.scanner.output_dir / 'simulations')
    return study
