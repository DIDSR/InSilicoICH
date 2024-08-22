'''
test xcist CT simulation functionality
'''
from pathlib import Path
from shutil import rmtree

import numpy as np
from monai.transforms import RandAffine

from pedsilicoICH.image_acquisition import read_dicom, CTobj
from pedsilicoICH.lesion_insertion import add_random_sphere_lesion
from pedsilicoICH.artifact_generation import transform_image_label_pair

from utils import get_effective_diameter, cosine_similarity


test_dir = Path(__file__).parent.absolute()
print(test_dir)
dcm = test_dir / 'CTP404_groundtruth.dcm'
phantom = read_dicom(dcm)
phantom = np.repeat(phantom[None], 11, axis=0)
diameter_pixels = get_effective_diameter(phantom[0], 1)
known_diameter_mm = 200
fov_mm = phantom.shape[-1]*known_diameter_mm/diameter_pixels
dx = fov_mm/phantom.shape[-1]

result_dir = test_dir / 'test_result'
if Path(result_dir).exists():
    rmtree(result_dir)

radius = 6
contrast = 400
img_w_lesion, lesion_vol, (z, x, y) = add_random_sphere_lesion(phantom,
                                                               phantom == 0,
                                                               radius,
                                                               contrast)
ct = CTobj(img_w_lesion, spacings=3*[dx], patientname='test',
           output_dir=result_dir)

views = 10
ct.run_scan(views=views)


def test_scan_shape():
    '''
    basic test of xcist simulations
    '''
    ct.run_recon(sliceCount=1)
    dcms = ct.write_to_dicom(result_dir / 'test.dcm')
    dcms_in_dir = list(result_dir.glob('*.dcm'))
    assert ct.recon.shape == (1, 512, 512)
    assert ct.projections.shape == (views,
                                    ct.xcist.cfg.scanner.detectorRowCount,
                                    ct.xcist.cfg.scanner.detectorColCount)
    assert len(dcms) == len(ct.start_positions)
    assert dcms_in_dir == dcms


def test_get_lesion_mask():
    ct.run_recon(sliceThickness=1)
    mask = ct.get_lesion_mask(lesion_vol)
    assert mask.shape == ct.recon.shape

    predicted_volume = 4/3*np.pi*radius**3
    measured_volume = mask.sum()
    rel_error = (predicted_volume - measured_volume)/predicted_volume
    tol = 0.5
    assert rel_error < tol


def test_get_lesion_mask_slicecount_1():
    ct.run_recon(sliceCount=1)
    mask = ct.get_lesion_mask(lesion_vol)
    if mask.shape != ct.recon.shape:
        Warning(f'mask shape != recon shape --> {mask.shape} != {ct.recon.shape}')

    predicted_volume = 4/3*np.pi*radius**3
    measured_volume = mask.sum()
    rel_error = (predicted_volume - measured_volume)/predicted_volume
    tol = 0.5
    assert rel_error < tol


def test_transform_image_label_pair():
    '''
    tests that the patient positioning augmentation actually applies and
    that following augmentation the result is not the same as the original,
    also tests repeatability by providing a random seed
    '''
    transform = RandAffine(prob=0.5,
                           rotate_range=[np.pi/4, np.pi/20, np.pi/20],
                           translate_range=[10, 10, 10],
                           scale_range=[0.1, 0.1, 0.1],
                           padding_mode="border")
    img_augmented, lesion_augmented = transform_image_label_pair(transform,
                                                                 img_w_lesion,
                                                                 lesion_vol,
                                                                 seed=42)
      
    # tests that the augmented results are different from the original
    assert cosine_similarity(img_augmented, img_w_lesion) < cosine_similarity(img_w_lesion, img_w_lesion)
    assert cosine_similarity(lesion_augmented, lesion_vol) < cosine_similarity(lesion_vol, lesion_vol)

    # tests repeatability given a seed value
    assert np.isclose(cosine_similarity(img_augmented, img_w_lesion), 0.869, atol=1e-4)
    assert cosine_similarity(lesion_augmented, lesion_vol) == 0
