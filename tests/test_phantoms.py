# %%
'''
test low level pedsilicoich phantom generation functionality
'''
from pathlib import Path
import os
from dotenv import load_dotenv

import numpy as np
from copy import deepcopy
from monai.transforms import RandAffine, Affine
from torchvision.datasets.utils import download_and_extract_archive

from pedsilicoICH.ground_truth_definition.phantoms import (MIDA_Head,
                                                           NIHPD_Head,
                                                           load_phantom)

nihpd_ages = [6.5, 9.0, 10.5, 11.5, 12.0, 15.75]

load_dotenv()
if 'PHANTOM_DIRECTORY' in os.environ:
    phantom_dir = Path(os.environ['PHANTOM_DIRECTORY'])
else:
    print('''
Please `export PHANTOM_DIRECTORY=/path/to/phantoms` or add your `.env`
file with PHANTOM_DIRECTORY=/path/to/phantoms
''')
    phantom_dir = Path(__file__).parent.absolute()

nihpd_dir = phantom_dir / 'NIHPD_Head_Phantom'
mida_dir = phantom_dir / 'MIDA_Head_Phantom'

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
    volume = 5
    phantom.insert_lesion(lesion_type, volume=volume, intensity=200,
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
    lesions = ['round', 'epidural', 'subdural']

    for age, phantom in zip(ages, phantoms):
        print(f'phantom of age: {age}, seed: {seed}')
        for lesion in lesions:
            for transform in transforms:
                transforms_performed_correctly(phantom, transform, lesion,
                                               seed=seed)


def mass_effect_works(seed, mass_effect, lesion_type='subdural'):
    intensity = 100
    age = 9
    volume = 40
    phantom = load_phantom(age)
    phantom.insert_lesion(lesion_type, volume=volume, intensity=intensity,
                          mass_effect=mass_effect, seed=seed)
    return phantom.mass_effect


def test_passing_mass_effect():
    passing_seed = 242
    mass_effect = 0.9
    for lesion_type in ['epidural', 'subdural']:
        mass_effect_flag = mass_effect_works(passing_seed, mass_effect,
                                             lesion_type)
    assert mass_effect_flag == mass_effect


def test_big_epidural_lesion():
    intensity = 100
    seed = 41
    age = 9
    desired_volume = 100
    phantom = load_phantom(age)
    phantom.insert_lesion('epidural', volume=desired_volume,
                          intensity=intensity, seed=seed)
    measured_volume = phantom._lesion[0].sum() *\
        (phantom.dx*phantom.dy*phantom.dz)/1000
    rel_vol_error = (desired_volume - measured_volume)/desired_volume*100
    assert abs(rel_vol_error) < 50


def test_big_subdural_lesion():
    intensity = 100
    seed = 41
    age = 9
    desired_volume = 80
    phantom = load_phantom(age)
    phantom.insert_lesion('subdural', volume=desired_volume,
                          intensity=intensity, seed=seed)
    measured_volume = phantom._lesion[0].sum() *\
        (phantom.dx*phantom.dy*phantom.dz)/1000
    rel_vol_error = (desired_volume - measured_volume)/desired_volume*100
    assert abs(rel_vol_error) < 100


def test_big_round_lesion():
    intensity = 100
    seed = 41
    age = 9
    desired_volume = 6  # mL
    phantom = load_phantom(age)
    phantom.insert_lesion('round', volume=desired_volume,
                          intensity=intensity, seed=seed, complexity=1)
    measured_volume = phantom._lesion[0].sum() *\
        (phantom.dx*phantom.dy*phantom.dz)/1000
    rel_vol_error = (desired_volume - measured_volume)/desired_volume*100
    assert abs(rel_vol_error) < 30


def test_volume_accuracy_reduced_phantom_matrix():
    intensity = 100
    seed = 41
    age = 9
    desired_volume = 6  # mL
    phantom = load_phantom(age, shape=(128, 128, 128))
    phantom.insert_lesion('round', volume=desired_volume,
                          intensity=intensity, seed=seed, complexity=1)
    measured_volume = phantom._lesion[0].sum() *\
        (phantom.dx*phantom.dy*phantom.dz)/1000
    rel_vol_error = (desired_volume - measured_volume)/desired_volume*100
    assert abs(rel_vol_error) < 40
