'''
tests high level Study functionality
'''
from insilicoICH.study import ICHStudy
from pathlib import Path
from shutil import rmtree

results_dir = Path('tests')

def test_control_study():
    output_dir = results_dir / 'no-lesion'
    if output_dir.exists():
        rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    study_list = ICHStudy.generate_from_distributions(
        ['9.0 yr NIHPD Head'],
        subtype=[None],
        scanner_model=['Siemens_DefinitionFlash'],
        views=[10],
        scan_coverage=(-10, 20),
        study_count=1,
        output_directory=output_dir)
    study = ICHStudy(study_list)
    study.run_all(overwrite=True, parallel=False)
    images = study.get_images(0)
    assert images.shape == (27, 512, 512)
    try:
        study.get_masks(0)
    except FileNotFoundError:
        pass

def test_mixed_study():
    output_dir = results_dir / 'mixed-lesion'
    if output_dir.exists():
        rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    study_list = ICHStudy.generate_from_distributions(
        ['9.0 yr NIHPD Head'],
        subtype=[None, 'IPH'],
        scanner_model=['Siemens_DefinitionFlash'],
        views=[10],
        scan_coverage=(-10, 20),
        study_count=2,
        output_directory=output_dir,
        seed = 1)
    study = ICHStudy(study_list)
    assert study.metadata.subtype[0] is None
    assert study.metadata.subtype[1] == 'IPH'
    study.run_all(overwrite=True, parallel=False)

    for idx in range(len(study)):
        images = study.get_images(idx)
        assert images.shape == (27, 512, 512)
    images = study.get_images(1)
    masks = study.get_masks(1)
    assert masks.shape == images.shape


def test_IPH_study():
    output_dir = results_dir / 'IPH_study'
    if output_dir.exists():
        rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    desired_vol = dict(IPH=[11, 13])
    desired_atten = dict(IPH=[300, 310])
    study_list = ICHStudy.generate_from_distributions(
        ['9.0 yr NIHPD Head'],
        subtype=['IPH'],
        lesion_volume=desired_vol,
        lesion_attenuation=desired_atten,
        scanner_model=['Siemens_DefinitionFlash'],
        views=[300],
        scan_coverage=(-10, 20),
        study_count=1,
        seed=206245,
        output_directory=output_dir)

    study = ICHStudy(study_list)
    study.run_all(overwrite=True, parallel=False)
    images = study.get_images(0)
    masks = study.get_masks(0)

    measured_lesion_signal = images[masks.astype(bool)].mean()
    contrast_err = measured_lesion_signal - desired_atten['IPH'][0]
    rel_contrast_err = abs(contrast_err) / desired_atten['IPH'][0]
    assert rel_contrast_err < 0.78  # do better, much has to do with making the lesion rather than CT sim

    vol_err = study.results['lesion_volume(mL)'].sum() - desired_vol['IPH'][0]
    rel_vol_err = abs(vol_err) / desired_vol['IPH'][0]
    assert rel_vol_err < 0.6


def test_EDH_study():
    output_dir = results_dir / 'EDH_study'
    if output_dir.exists():
        rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    desired_vol = dict(EDH=[11, 13])
    desired_atten = dict(EDH=[300, 310])
    study_list = ICHStudy.generate_from_distributions(
        ['9.0 yr NIHPD Head'],
        subtype=['EDH'],
        lesion_volume=desired_vol,
        lesion_attenuation=desired_atten,
        scanner_model=['Siemens_DefinitionFlash'],
        views=[300],
        scan_coverage=(25, 55),
        study_count=1,
        seed=206245,
        output_directory=output_dir)
    study = ICHStudy(study_list)
    study.run_all(overwrite=True, parallel=False)
    images = study.get_images(0)
    masks = study.get_masks(0)

    measured_lesion_signal = images[masks.astype(bool)].mean()
    contrast_err = measured_lesion_signal - desired_atten['EDH'][0]
    rel_contrast_err = abs(contrast_err) / desired_atten['EDH'][0]
    assert rel_contrast_err < 0.55

    vol_err = desired_vol['EDH'][0] - study.results['lesion_volume(mL)'].sum()
    rel_vol_err = abs(vol_err) / desired_vol['EDH'][0]
    assert rel_vol_err < 1.2  # too high, fix this


def test_SDH_study():
    output_dir = results_dir / 'SDH_study'
    if output_dir.exists():
        rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    desired_vol = dict(SDH=[11, 13])
    desired_atten = dict(SDH=[300, 310])
    study_list = ICHStudy.generate_from_distributions(
        ['9.0 yr NIHPD Head'],
        subtype=['SDH'],
        lesion_volume=desired_vol,
        lesion_attenuation=desired_atten,
        scanner_model=['Siemens_DefinitionFlash'],
        views=[300],
        scan_coverage=(25, 55),
        study_count=1,
        seed=206245,
        output_directory=output_dir)
    study = ICHStudy(study_list)
    study.run_all(overwrite=True, parallel=False)
    images = study.get_images(0)
    masks = study.get_masks(0)

    measured_lesion_signal = images[masks.astype(bool)].mean()
    contrast_err = measured_lesion_signal - desired_atten['SDH'][0]
    rel_contrast_err = abs(contrast_err) / desired_atten['SDH'][0]
    assert rel_contrast_err < 0.5

    vol_err = study.results['lesion_volume(mL)'].sum() - desired_vol['SDH'][0]
    rel_vol_err = abs(vol_err) / desired_vol['SDH'][0]
    assert rel_vol_err < 0.1
