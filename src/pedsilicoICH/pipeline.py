'''
pipeline: this module organizes the healthy head phantoms, lesion definitions,
augmentations, and CT simulation together into the final ct_simulation function
'''
from pathlib import Path

from monai.transforms import RandAffine
import numpy as np

from .image_acquisition import CTobj


def ct_simulation(output_directory, phantom, views=1000, fov=250, mA=200,
                  kVp=120, zspan='dynamic', add_positioning_augmentation=True):
    patient_name = phantom.patient_name
    age = phantom.age
    lesion_type = phantom.lesion_type
    radius = phantom.lesion_radius
    contrast = phantom.lesion_contrast

    print(f'{age} years, {lesion_type}, {contrast} HU')

    phantom.insert_lesion(lesion_type, radius=radius, contrast=contrast)
    img_w_lesion = phantom.get_CT_number_phantom()
    lesion_image = phantom._lesion[0]

    if add_positioning_augmentation:
        transform = RandAffine(prob=0.5,
                               rotate_range=[np.pi/4, np.pi/20, np.pi/20],
                               translate_range=[10, 10, 10],
                               scale_range=[0.1, 0.1, 0.1],
                               padding_mode="border")
        phantom.apply_transform(transform)
        img_w_lesion = phantom.get_CT_number_phantom()
        lesion_image = phantom._lesion[0]

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
