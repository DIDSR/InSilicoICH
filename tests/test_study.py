'''
tests high level Study functionality
'''
from insilicoICH.study import ICHStudy


def test_IPH_study():
    study = ICHStudy()
    desired_vol = 12
    desired_atten = 300
    study.append('9.0 yr NIHPD Head',
                 Subtype='IPH', LesionVolume=desired_vol,
                 LesionAttenuation=desired_atten,
                 ScannerModel='Siemens_DefinitionFlash',
                 ScanCoverage=(17, 26), Views=100, Seed=206245)
    study.run_all()
    images = study.get_images(0)
    masks = study.get_masks(0)

    measured_lesion_signal = images[masks.astype(bool)].mean()
    contrast_err = measured_lesion_signal - desired_atten
    rel_contrast_err = abs(contrast_err) / desired_atten
    assert rel_contrast_err < 0.78  # do better

    vol_err = study.results['LesionVolume(mL)'].sum() - desired_vol
    rel_vol_err = abs(vol_err) / desired_vol
    assert rel_vol_err < 0.2


def test_EDH_study():
    study = ICHStudy()
    desired_vol = 12
    desired_atten = 300
    study.append('9.0 yr NIHPD Head',
                 Subtype='EDH', LesionVolume=desired_vol,
                 LesionAttenuation=desired_atten,
                 ScannerModel='Siemens_DefinitionFlash',
                 ScanCoverage=(10, 30), Views=100, Seed=206245)
    study.run_all()
    images = study.get_images(0)
    masks = study.get_masks(0)

    measured_lesion_signal = images[masks.astype(bool)].mean()
    contrast_err = measured_lesion_signal - desired_atten
    rel_contrast_err = abs(contrast_err) / desired_atten
    assert rel_contrast_err < 0.22

    vol_err = study.results['LesionVolume(mL)'].sum() - desired_vol
    rel_vol_err = abs(vol_err) / desired_vol
    assert rel_vol_err < 0.93  # too high, fix this


def test_SDH_study():
    study = ICHStudy()
    desired_vol = 12
    desired_atten = 300
    study.append('9.0 yr NIHPD Head',
                 Subtype='SDH', LesionVolume=desired_vol,
                 LesionAttenuation=desired_atten,
                 ScannerModel='Siemens_DefinitionFlash',
                 ScanCoverage=(20, 28), Views=100, Seed=206245)
    study.run_all()
    images = study.get_images(0)
    masks = study.get_masks(0)

    measured_lesion_signal = images[masks.astype(bool)].mean()
    contrast_err = measured_lesion_signal - desired_atten
    rel_contrast_err = abs(contrast_err) / desired_atten
    assert rel_contrast_err < 0.24

    vol_err = study.results['LesionVolume(mL)'].sum() - desired_vol
    rel_vol_err = abs(vol_err) / desired_vol
    assert rel_vol_err < 0.78
