# %%
'''
test low level pedsilicoich phantom generation functionality
'''
from pathlib import Path

import numpy as np
from copy import deepcopy
from monai.transforms import RandAffine, Affine
from torchvision.datasets.utils import download_and_extract_archive

from pedsilicoICH.ground_truth_definition.phantoms import MIDA_Head, NIHPD_Head

nihpd_ages = [6.5, 9.0, 10.5, 11.5, 12.0, 15.75]

test_dir = Path(__file__).parent.absolute()

nihpd_dir = test_dir.parent / 'NIHPD_Head_Phantom'
mida_dir = test_dir.parent / 'MIDA_Head_Phantom'

if not nihpd_dir.exists():
    url = 'https://www.bic.mni.mcgill.ca/~vfonov/nihpd/obj1_analyze.zip'
    download_and_extract_archive(url, nihpd_dir)


def lesion_added_correctly(phantom):
    phantom.insert_lesion('epidural', seed=880, mass_effect=False)
    lesion_vol = phantom.get_lesion_mask()
    z, x, y = phantom._lesion_coords[0]
    return (lesion_vol.sum() > 0) & (lesion_vol[z, x, y] == 1)


def test_MIDA_Head():
    if mida_dir.exists():
        phantom = MIDA_Head(mida_dir)
        assert lesion_added_correctly(phantom)
    else:
        Warning(f'MIDA head phantom not found in {mida_dir}, skipping...')


def test_NIHPD():
    for age in nihpd_ages:
        phantom = NIHPD_Head(nihpd_dir, age)
        assert lesion_added_correctly(phantom)


def transforms_performed_correctly(phantom, transform, lesion_type, tol=0.2,
                                   seed=None):
    print(phantom, transform, lesion_type)
    phantom = deepcopy(phantom)
    radius = 3
    volume = 4/3*np.pi*radius**3
    phantom.insert_lesion(lesion_type, volume=volume, contrast=200,
                          mass_effect=False, seed=seed)
    lesion = phantom.get_lesion_mask()
    phantom.apply_transform(transform, seed=seed)
    transformed_lesion = phantom.get_lesion_mask()
    err = np.abs(lesion.sum() - transformed_lesion.sum())/lesion.sum()
    print(f'transforms_performed_correctly error: {err:2.3f} [tol: {tol}]')
    assert err < tol


def test_transforms_on_phantoms(seed=885):
    'tests each combination of phantom and transform'
    mida_shape = (240, 240, 175)
    phantoms = [NIHPD_Head(nihpd_dir, age, shape=mida_shape) for
                age in nihpd_ages]
    ages = nihpd_ages
    if mida_dir.exists():
        mida = MIDA_Head(mida_dir)
        phantoms = [mida] + phantoms
        ages = [38] + ages
    else:
        Warning(f'MIDA head phantom not found in {mida_dir}, skipping...')

    randaffine = RandAffine(prob=0.5,
                            rotate_range=[np.pi/4, np.pi/20, np.pi/20],
                            translate_range=[10, 10, 10],
                            scale_range=[0.1, 0.1, 0.1],
                            padding_mode="border")
    affine = Affine(rotate_params=np.pi/4, padding_mode="border")

    transforms = [randaffine, affine]
    lesions = ['sphere', 'epidural', 'subdural']

    for age, phantom in zip(ages, phantoms):
        print(f'phantom of age: {age}, seed: {seed}')
        for lesion in lesions:
            for transform in transforms:
                transforms_performed_correctly(phantom, transform, lesion,
                                               seed=seed)
