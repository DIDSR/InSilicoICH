# %%
'''
tests high level Study functionality
'''
import numpy as np
from monai.transforms import RandAffine

from pedsilicoICH.pipeline import Study, run_study
from pedsilicoICH.ground_truth_definition.phantoms import load_phantom
from pedsilicoICH.image_acquisition import Scanner

age = 6.5
transform = RandAffine(prob=0.5,
                       rotate_range=[np.pi/4, np.pi/20, np.pi/20],
                       translate_range=[10, 10, 10],
                       scale_range=[0.1, 0.1, 0.1],
                       padding_mode="border")


def test_run_study():
    study = run_study('test', age=age, views=100, zspan=(-5, 5))
    assert study.images.mean() > -1000


sphere_lesion_tol = 32


def test_sphere_lesion_study():
    phantom = load_phantom(age)
    phantom.insert_lesion('round', volume=5, intensity=300, seed=336)
    lesion_level_mm = (phantom.get_CT_number_phantom().shape[0]/2 -
                       phantom._lesion_coords[0][0])*phantom.dz
    center = lesion_level_mm
    width = 8

    scanner = Scanner(phantom)
    study = Study(scanner, 'test')
    study.run_study('test', zspan=(center-width//2, center+width//2), views=50)
    measured_lesion_signal = study.images[study.lesion.astype(bool)].mean()
    print(measured_lesion_signal)
    assert measured_lesion_signal > sphere_lesion_tol


def test_sphere_augmented_position_study():
    phantom = load_phantom(age)
    phantom.insert_lesion('round', volume=5, intensity=300, seed=336)
    phantom.apply_transform(transform, seed=42)
    lesion_level_mm = (phantom.get_CT_number_phantom().shape[0]/2 -
                       phantom._lesion_coords[0][0])*phantom.dz
    center = lesion_level_mm
    width = 8

    scanner = Scanner(phantom)
    study = Study(scanner, 'test')
    study.run_study('test', zspan=(center-width//2, center+width//2),
                    views=100)
    measured_lesion_signal = study.images[study.lesion.astype(bool)].mean()
    assert measured_lesion_signal > sphere_lesion_tol


def test_epidural_lesion_study():
    phantom = load_phantom(age)
    phantom.insert_lesion('epidural', volume=5, intensity=300, seed=336)
    lesion_level_mm = (phantom.get_CT_number_phantom().shape[0]/2 -
                       phantom._lesion_coords[0][0])*phantom.dz
    center = lesion_level_mm
    width = 8

    scanner = Scanner(phantom)
    study = Study(scanner, 'test')
    study.run_study('test', zspan=(center-width//2, center+width//2),
                    views=100)
    measured_lesion_signal = study.images[study.lesion.astype(bool)].mean()
    assert measured_lesion_signal > 38


def test_epidural_augmented_position_study():
    phantom = load_phantom(age)
    phantom.insert_lesion('epidural', volume=5, intensity=300, seed=336)
    phantom.apply_transform(transform, seed=42)
    lesion_level_mm = (phantom.get_CT_number_phantom().shape[0]/2 -
                       phantom._lesion_coords[0][0])*phantom.dz
    center = lesion_level_mm
    width = 8

    scanner = Scanner(phantom)
    study = Study(scanner, 'test')
    study.run_study('test', zspan=(center-width//2, center+width//2),
                    views=100)
    measured_lesion_signal = study.images[study.lesion.astype(bool)].mean()
    assert measured_lesion_signal > 38


def test_subdural_lesion_study():
    phantom = load_phantom(age)
    phantom.insert_lesion('subdural', volume=5, intensity=300, seed=336)
    lesion_level_mm = (phantom.get_CT_number_phantom().shape[0]/2 -
                       phantom._lesion_coords[0][0])*phantom.dz
    center = lesion_level_mm
    width = 8

    scanner = Scanner(phantom)
    study = Study(scanner, 'test')
    study.run_study('test', zspan=(center-width//2, center+width//2),
                    views=100)
    measured_lesion_signal = study.images[study.lesion.astype(bool)].mean()
    assert measured_lesion_signal > 21
    assert study.metadata['lesion volume [mL]'].sum() > 1.7


def test_subdural_augmented_position_study():
    phantom = load_phantom(age)
    phantom.insert_lesion('subdural', volume=5, intensity=300, seed=336)
    phantom.apply_transform(transform, seed=42)
    lesion_level_mm = (phantom.get_CT_number_phantom().shape[0]/2 -
                       phantom._lesion_coords[0][0])*phantom.dz
    center = lesion_level_mm
    width = 8

    scanner = Scanner(phantom)
    study = Study(scanner, 'test')
    study.run_study('test', zspan=(center-width//2, center+width//2),
                    views=100)
    measured_lesion_signal = study.images[study.lesion.astype(bool)].mean()
    assert measured_lesion_signal > 21
