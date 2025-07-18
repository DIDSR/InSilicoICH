'''
tests high level Study functionality
'''
from insilicoICH.study import ICHStudy


def test_IPH_study():
    desired_vol = dict(IPH=[11, 13])
    desired_atten = dict(IPH=[300, 310])
    study_list = ICHStudy.generate_from_distributions(
        ['9.0 yr NIHPD Head'],
        subtype=['IPH'],
        lesion_volume=desired_vol,
        lesion_attenuation=desired_atten,
        scanner_model=['Siemens_DefinitionFlash'],
        views=100,
        scan_coverage=(-10, 20),
        study_count=1,
        seed=206245)
    study = ICHStudy(study_list)
    study.run_all()
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

    desired_vol = dict(EDH=[11, 13])
    desired_atten = dict(EDH=[300, 310])
    study_list = ICHStudy.generate_from_distributions(
        ['9.0 yr NIHPD Head'],
        subtype=['EDH'],
        lesion_volume=desired_vol,
        lesion_attenuation=desired_atten,
        scanner_model=['Siemens_DefinitionFlash'],
        views=100,
        scan_coverage=(25, 55),
        study_count=1,
        seed=206245)
    study = ICHStudy(study_list)
    study.run_all()
    images = study.get_images(0)
    masks = study.get_masks(0)

    measured_lesion_signal = images[masks.astype(bool)].mean()
    contrast_err = measured_lesion_signal - desired_atten['EDH'][0]
    rel_contrast_err = abs(contrast_err) / desired_atten['EDH'][0]
    assert rel_contrast_err < 0.55

    vol_err = desired_vol['EDH'][0] - study.results['lesion_volume(mL)'].sum()
    rel_vol_err = abs(vol_err) / desired_vol['EDH'][0]
    assert rel_vol_err < 0.6  # too high, fix this


def test_SDH_study():
    study = ICHStudy()
    desired_vol = dict(SDH=[11, 13])
    desired_atten = dict(SDH=[300, 310])
    desired_vol = dict(SDH=[11, 13])
    desired_atten = dict(SDH=[300, 310])
    study_list = ICHStudy.generate_from_distributions(
        ['9.0 yr NIHPD Head'],
        subtype=['SDH'],
        lesion_volume=desired_vol,
        lesion_attenuation=desired_atten,
        scanner_model=['Siemens_DefinitionFlash'],
        views=100,
        scan_coverage=(25, 55),
        study_count=1,
        seed=206245)
    study = ICHStudy(study_list)
    study.run_all()
    images = study.get_images(0)
    masks = study.get_masks(0)

    measured_lesion_signal = images[masks.astype(bool)].mean()
    contrast_err = measured_lesion_signal - desired_atten['SDH'][0]
    rel_contrast_err = abs(contrast_err) / desired_atten['SDH'][0]
    assert rel_contrast_err < 0.2

    vol_err = study.results['lesion_volume(mL)'].sum() - desired_vol['SDH'][0]
    rel_vol_err = abs(vol_err) / desired_vol['SDH'][0]
    assert rel_vol_err < 0.1
