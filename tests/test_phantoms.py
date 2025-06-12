'''
test low level insilicoich phantom generation functionality
'''
from functools import partial

from monai.transforms import RandAffine
import numpy as np
from VITools import get_available_phantoms

from insilicoICH.phantoms.head_phantoms import MIDA_Head


available_phantoms = get_available_phantoms()


shape = 3*[128]
seed = 41


def load_phantom(age, shape=None):
    '''
    load a phantom for testing
    '''
    if age < 6.5:
        return available_phantoms[f'{age} yr UNC Head'](shape=shape)
    if age < 19.0:
        return available_phantoms[f'{age} yr NIHPD Head'](shape=shape)
    if age == 38.0:
        return available_phantoms[f'{age} yr MIDA Head'](shape=shape)


def rmse(x, y): return np.sqrt(np.mean((x-y)**2))


def test_big_epidural_lesion():
    intensity = 100
    age = 9.0
    mass_effect = True
    desired_volume = 60
    phantom = load_phantom(age, shape=shape)
    phantom.insert_lesion('EDH', volume=desired_volume,
                          intensity=intensity,
                          mass_effect=mass_effect,
                          seed=seed)
    measured_volume = phantom.get_lesion_volume()
    assert rmse(desired_volume, measured_volume) < 21


def test_big_subdural_lesion():
    intensity = 100
    age = 9.0
    desired_volume = 60
    mass_effect = True
    phantom = load_phantom(age, shape=shape)
    phantom.insert_lesion('SDH', volume=desired_volume,
                          intensity=intensity,
                          mass_effect=mass_effect,
                          seed=seed)
    measured_volume = phantom.get_lesion_volume()
    assert rmse(desired_volume, measured_volume) < 56


def test_big_intraparenchymal_lesion():
    intensity = 100
    age = 9.0
    desired_volume = 60
    mass_effect = True
    phantom = load_phantom(age, shape=shape)
    phantom.insert_lesion('IPH', volume=desired_volume,
                          intensity=intensity,
                          mass_effect=mass_effect,
                          seed=seed)
    measured_volume = phantom.get_lesion_volume()
    assert rmse(desired_volume, measured_volume) < 56


def test_transforms(threshold=-685):
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


def check_volumes(inputs=list(range(1, 10)), **kwargs):
    outs = []
    for input_vol in inputs:
        phantom = load_phantom(15.75)
        phantom.insert_lesion(lesion_type='IPH', volume=input_vol, **kwargs)
        outs.append(phantom.get_lesion_volume())
    return outs


def test_IPH_volume_accuracy():
    '''
    tests IPH volume accuracy across different degress of IPH
    complexity (multiple sub IPHs) `complexity`>1 and `overlap`
    of these sub IPHs
    '''
    inputs = np.linspace(1, 70, 3)
    for overlap in [0.2, 0.4]:
        for complexity in range(1, 4):
            corrected = check_volumes(inputs=inputs,
                                      iph_kwargs=dict(complexity=complexity,
                                                      overlap=overlap),
                                      seed=seed)
            assert rmse(inputs, corrected) < 20


def test_load_phantoms():
    '''
    tests that all phantoms load successfully
    '''
    for name, phantom_class in available_phantoms.items():
        if isinstance(phantom_class, partial) and issubclass(phantom_class.func,
                                                             MIDA_Head):
            continue
        phantom = phantom_class()
        print(f'{name} {phantom}')


def test_mass_effect():
    '''
    tests that mass effect is applied correctly
    '''
    age = 6.5
    vol = 20
    seed = 42
    phantom = load_phantom(age)
    phantom.insert_lesion('EDH', volume=vol, mass_effect=False, seed=seed)
    phantom_no_me_image = phantom.get_CT_number_phantom()[
        phantom._lesion_coords[0][0]]

    phantom_me = load_phantom(age)
    phantom_me.insert_lesion('EDH', volume=vol, mass_effect=0.5, seed=seed)
    phantom_me_image = phantom_me.get_CT_number_phantom()[
        phantom_me._lesion_coords[0][0]
        ]

    me_05 = phantom_me_image - phantom_no_me_image
    assert (np.linalg.norm(me_05) > 300) & (np.linalg.norm(me_05) < 1000)

    phantom_me = load_phantom(age)
    phantom_me.insert_lesion('EDH', volume=vol, mass_effect=1.0, seed=seed)
    phantom_me_image = phantom_me.get_CT_number_phantom()[
        phantom_me._lesion_coords[0][0]
        ]

    me_10 = phantom_me_image - phantom_no_me_image

    assert np.linalg.norm(me_10) > np.linalg.norm(me_05)
