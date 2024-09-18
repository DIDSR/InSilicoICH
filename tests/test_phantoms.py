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
from pedsilicoICH.lesion_insertion import add_epidural_lesion

nihpd_ages = [6.5, 9.0, 10.5, 11.5, 12.0, 15.75]

test_dir = Path(__file__).parent.absolute()

nihpd_dir = test_dir.parent / 'NIHPD_Head_Phantom'
mida_dir = test_dir.parent / 'MIDA_Head_Phantom'

if not nihpd_dir.exists():
    url = 'https://www.bic.mni.mcgill.ca/~vfonov/nihpd/obj1_analyze.zip'
    download_and_extract_archive(url, nihpd_dir)


def lesion_added_correctly(phantom):
    ground_truth = phantom.get_CT_number_phantom()
    dura_map = phantom.get_dura_map()
    _, lesion_vol, (z, x, y) = add_epidural_lesion(ground_truth, dura_map,
                                                   spacing=phantom.spacings,
                                                   seed=880)
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
    phantom.insert_lesion(lesion_type, volume=volume, contrast=200, seed=seed)
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
# # %%
# import numpy as np
# seeds = []
# count = 0
# while len(seeds) < 1:
#     count += 1
#     seed = np.random.randint(0, 1000)
#     seed = 885
#     mida_shape = (240, 240, 175)
#     phantoms = [NIHPD_Head(nihpd_dir, age, shape=mida_shape) for
#                 age in nihpd_ages]
#     ages = nihpd_ages
#     if mida_dir.exists():
#         mida = MIDA_Head(mida_dir, shape=mida_shape)
#         phantoms = [mida] + phantoms
#         ages = [38] + ages
#     else:
#         Warning(f'MIDA head phantom not found in {mida_dir}, skipping...')

#     randaffine = RandAffine(prob=0.5,
#                             rotate_range=[np.pi/4, np.pi/20, np.pi/20],
#                             translate_range=[10, 10, 10],
#                             scale_range=[0.1, 0.1, 0.1],
#                             padding_mode="border")
#     affine = Affine(rotate_params=np.pi/4, padding_mode="border")

#     transforms = [randaffine]
#     lesions = ['sphere']

#     try:
#         for age, phantom in zip(ages, phantoms):
#             print(f'phantom of age: {age}, seed: {seed}')
#             for lesion in lesions:
#                 for transform in transforms:
#                     transforms_performed_correctly(phantom, transform, lesion,
#                                                     seed=seed)
#         seeds.append(seed)
#     except:
#         print(f'Attempt: {count}, {seed} failed...')
# print(seeds)
# # %%

# %%
