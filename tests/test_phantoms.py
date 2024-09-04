'''
test pedsilicoich phantom generation functionality
'''
# %%
from pathlib import Path

import numpy as np
from monai.transforms import RandAffine, Affine

from pedsilicoICH.ground_truth_definition.phantoms import MIDA_Head, NIHPD_Head
from pedsilicoICH.lesion_insertion import (add_sphere_lesion,
                                           add_epidural_lesion,
                                           add_subdural_lesion)
from pedsilicoICH.artifact_generation import transform_image_label_pair

nihpd_ages = [6.5, 9.0, 10.5, 11.5, 12.0, 15.75]

test_dir = Path(__file__).parent.absolute()
print(test_dir)

nihpd_dir = test_dir.parent / 'NIHPD_Head_Phantom'
MIDA_dir = test_dir.parent / 'MIDA_Head_Phantom'
# %%


def lesion_added_correctly(phantom):
    ground_truth = phantom.get_CT_number_phantom()
    dura_map = phantom.get_dura_map()
    _, lesion_vol, (z, x, y) = add_epidural_lesion(ground_truth, dura_map,
                                                   spacing=phantom.spacings,
                                                   seed=880)
    return (lesion_vol.sum() > 0) & (lesion_vol[z, x, y] == 1)


def test_MIDA_Head():
    phantom = MIDA_Head(MIDA_dir)
    assert lesion_added_correctly(phantom)


def test_NIHPD():
    for age in nihpd_ages:
        phantom = NIHPD_Head(nihpd_dir, age)
        assert lesion_added_correctly(phantom)


def transforms_performed_correctly(phantom, transform, lesion_type, tol=0.2,
                                   seed=None):
    print(phantom, transform, lesion_type)
    if lesion_type == 'sphere':
        lesion_func = add_sphere_lesion
        material = 'white matter'  # brain region where SPHERE lesion will be inserted options based on `material_lut` materials
        mask = phantom.get_material_mask(material).astype(int)
        params = {'radius': 1, 'contrast': 200}
    elif lesion_type == 'epidural':
        lesion_func = add_epidural_lesion
        mask = phantom.get_dura_map()
        params = {'spacing': phantom.spacings}
    else:
        lesion_func = add_subdural_lesion
        mask = phantom.get_dura_map()
        params = {'spacing': phantom.spacings}
    ground_truth = phantom.get_CT_number_phantom()

    img_w_lesion, lesion, _ = lesion_func(ground_truth, mask,
                                          seed=seed, **params)

    _, transformed_lesion = transform_image_label_pair(transform,
                                                       img_w_lesion,
                                                       lesion,
                                                       seed=42)
    err = np.abs(lesion.sum() - transformed_lesion.sum())/lesion.sum()
    print(f'transforms_performed_correctly error: {err:2.3f} [tol: {tol}]')
    assert err < tol


def test_transforms_on_phantoms(seed=880):
    'tests each combination of phantom and transform'
    mida = MIDA_Head(MIDA_dir)
    mida_shape = mida.get_CT_number_phantom().shape
    nihpds = [NIHPD_Head(nihpd_dir, age, shape=mida_shape) for age in nihpd_ages]
    phantoms = [mida] + nihpds
    ages = [38] + nihpd_ages

    randaffine = RandAffine(prob=0.5,
                            rotate_range=[np.pi/4, np.pi/20, np.pi/20],
                            translate_range=[10, 10, 10],
                            scale_range=[0.1, 0.1, 0.1],
                            padding_mode="border")
    affine = Affine(rotate_params=np.pi/4, padding_mode="border")

    transforms = [randaffine, affine]
    lesions = ['sphere', 'epidural', 'subdural']

    for age, phantom in zip(ages, phantoms):
        print(f'phantom of age: {age}')
        for lesion in lesions:
            for transform in transforms:
                transforms_performed_correctly(phantom, transform, lesion,
                                               seed=seed)
