'''
test low level insilicoich phantom generation functionality
'''
from pathlib import Path
import os
from dotenv import load_dotenv
from monai.transforms import RandAffine
import numpy as np

from insilicoICH.ground_truth_definition.utils import download_and_extract_archive
from insilicoICH.ground_truth_definition.phantoms import load_phantom

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

shape = 3*[128]
seed = 41


def test_big_epidural_lesion():
    intensity = 100
    age = 9
    mass_effect = True
    desired_volume = 100
    phantom = load_phantom(age, shape=shape)
    phantom.insert_lesion('EDH', volume=desired_volume,
                          intensity=intensity,
                          mass_effect=mass_effect,
                          seed=seed)
    measured_volume = phantom._lesion[0].sum() *\
        (phantom.dx*phantom.dy*phantom.dz)/1000
    rel_vol_error = (desired_volume - measured_volume)/desired_volume*100
    assert abs(rel_vol_error) < 50


def test_big_subdural_lesion():
    intensity = 100
    age = 9
    desired_volume = 80
    mass_effect = True
    phantom = load_phantom(age, shape=shape)
    phantom.insert_lesion('SDH', volume=desired_volume,
                          intensity=intensity,
                          mass_effect=mass_effect,
                          seed=seed)
    measured_volume = phantom._lesion[0].sum() *\
        (phantom.dx*phantom.dy*phantom.dz)/1000
    rel_vol_error = (desired_volume - measured_volume)/desired_volume*100
    assert abs(rel_vol_error) < 100


def test_big_round_lesion():
    intensity = 100
    age = 9
    desired_volume = 6  # mL
    mass_effect = True
    phantom = load_phantom(age, shape=shape)
    phantom.insert_lesion('IPH', volume=desired_volume,
                          intensity=intensity,
                          mass_effect=mass_effect,
                          seed=seed, complexity=1)
    measured_volume = phantom._lesion[0].sum() *\
        (phantom.dx*phantom.dy*phantom.dz)/1000
    rel_vol_error = (desired_volume - measured_volume)/desired_volume*100
    assert abs(rel_vol_error) < 41


def test_volume_accuracy_full_matrix():
    intensity = 100
    age = 9
    desired_volume = 6  # mL
    mass_effect = False
    phantom = load_phantom(age)
    phantom.insert_lesion('IPH', volume=desired_volume,
                          intensity=intensity, seed=seed,
                          mass_effect=mass_effect,
                          complexity=3)
    measured_volume = phantom._lesion[0].sum() *\
        (phantom.dx*phantom.dy*phantom.dz)/1000
    rel_vol_error = (desired_volume - measured_volume)/desired_volume*100
    assert abs(rel_vol_error) < 40


def test_transforms(threshold=-585):
    for age in [6.5]:
        phantom = load_phantom(age)
        transform = RandAffine(prob=1,
                               rotate_range=[np.pi/4, np.pi/20, np.pi/20],
                               translate_range=[10, 10, 10],
                               scale_range=[0.1, 0.1, 0.1],
                               padding_mode="border")

        phantom.apply_transform(transform)
        test_val = phantom.get_CT_number_phantom().mean()
        assert test_val > threshold
