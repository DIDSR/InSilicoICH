'''
pipeline: this module organizes the healthy head phantoms, lesion definitions,
augmentations, and CT simulation together into the final ct_simulation function
'''
from pathlib import Path

from monai.transforms import RandAffine
import numpy as np

from .ground_truth_definition.phantoms import NIHPD_Head, MIDA_Head
from .image_acquisition import CTobj
from .lesion_insertion import (add_sphere_lesion,
                               add_epidural_lesion,
                               add_subdural_lesion)
from .artifact_generation import transform_image_label_pair

nihpd_dir = Path('/gpfs_projects/brandon.nelson/pedsilicoICH/brain_atlases/obj1_analyze/')
MIDA_dir = Path('MIDA Head Phantom')


def ct_simulation(output_directory, patient_name, age, lesion_type, radius,
                  contrast, views=1000, fov=250, mA=200, kVp=120,
                  zspan='dynamic', add_positioning_augmentation=True):
    print(f'{age} years, {lesion_type}, {contrast} HU')
    mida_shape = (480, 480, 350)  # default shape of MIDA
    mida_age = 38
    if age == mida_age:
        phantom = MIDA_Head(MIDA_dir, shape=mida_shape)
    else:
        phantom = NIHPD_Head(nihpd_dir, age=age, shape=mida_shape)

    if lesion_type == 'sphere':
        lesion_func = add_sphere_lesion
        material = 'white matter'  # brain region where SPHERE lesion will be inserted options based on `material_lut` materials
        mask = phantom.get_material_mask(material).astype(int)
        params = {'radius': radius, 'contrast': contrast}
    elif lesion_type == 'epidural':
        lesion_func = add_epidural_lesion
        mask = phantom.get_dura_map()
        params = {'spacing': (phantom.dz, phantom.dx, phantom.dy),
                  'contrast': contrast}
    else:
        lesion_func = add_subdural_lesion
        mask = phantom.get_dura_map()
        params = {'spacing': (phantom.dz, phantom.dx, phantom.dy),
                  'contrast': contrast}
    ground_truth_image = phantom.get_CT_number_phantom()

    img_w_lesion, lesion_image, _ = lesion_func(ground_truth_image,
                                                mask, **params)

    if add_positioning_augmentation:
        transform = RandAffine(prob=0.5,
                               rotate_range=[np.pi/4, np.pi/20, np.pi/20],
                               translate_range=[10, 10, 10],
                               scale_range=[0.1, 0.1, 0.1],
                               padding_mode="border")
        img_w_lesion, lesion_image = transform_image_label_pair(transform,
                                                                img_w_lesion,
                                                                lesion_image)

    output_dir = Path(output_directory) / patient_name
    output_dir.mkdir(exist_ok=True, parents=True)
    ct = CTobj(img_w_lesion, spacings=phantom.spacings,
               patientname=patient_name,
               age=age,
               studyname='full volume long scan',
               output_dir=output_dir)

    if zspan == 'dynamic':
        startZ, endZ = ct.recommend_scan_range()
    elif isinstance(zspan, tuple):
        startZ, endZ = zspan

    ct.run_scan(startZ=startZ, endZ=endZ, views=views, mA=mA, kVp=kVp)
    ct.run_recon(fov=fov)
    dicom_path = output_dir / 'dicoms'
    dcm_files = ct.write_to_dicom(dicom_path / f'{patient_name}.dcm')

    lesion_only = ct
    mask = ct.get_lesion_mask(lesion_image,
                              startZ=startZ, endZ=endZ)

    lesion_only.recon = mask
    dicom_path = output_dir / 'lesion_masks'
    mask_files = lesion_only.write_to_dicom(dicom_path / f'{patient_name}_mask.dcm')
    return dcm_files, mask_files
