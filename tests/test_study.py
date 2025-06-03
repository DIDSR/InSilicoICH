# %%
'''
tests high level Study functionality
'''
from insilicoICH.study import ICHStudy


def test_IPH_study():
    study = ICHStudy()
    desired_vol = 5
    desired_atten = 300
    study.append('10.5 yr NIHPD Head',
                 Subtype='IPH', LesionVolume=desired_vol,
                 LesionAttenuation=desired_atten,
                 ScanCoverage=(8, 17), Views=100, Seed=206245)
    study.run_all()
    images = study.get_images(0)
    masks = study.get_masks(0)
    measured_lesion_signal = images[masks.astype(bool)].mean()
    contrast_err = measured_lesion_signal - desired_atten
    vol_err = study.results['LesionVolume(mL)'].sum() - desired_vol
    assert abs(contrast_err) / desired_atten < 0.5
    assert abs(vol_err) / desired_vol < 0.8


def test_EDH_study():
    study = ICHStudy()
    desired_vol = 5
    desired_atten = 300
    study.append('10.5 yr NIHPD Head',
                 Subtype='EDH', LesionVolume=desired_vol,
                 LesionAttenuation=desired_atten,
                 ScanCoverage=(40, 48), Views=100, Seed=206245)
    study.run_all()
    images = study.get_images(0)
    masks = study.get_masks(0)
    measured_lesion_signal = images[masks.astype(bool)].mean()
    contrast_err = measured_lesion_signal - desired_atten
    vol_err = study.results['LesionVolume(mL)'].sum() - desired_vol
    assert abs(contrast_err) / desired_atten < 0.2
    assert abs(vol_err) / desired_vol < 0.5


def test_SDH_study():
    study = ICHStudy()
    desired_vol = 5
    desired_atten = 300
    study.append('10.5 yr NIHPD Head',
                 Subtype='SDH', LesionVolume=desired_vol,
                 LesionAttenuation=desired_atten,
                 ScanCoverage=(20, 28), Views=100, Seed=206245)
    study.run_all()
    images = study.get_images(0)
    masks = study.get_masks(0)
    measured_lesion_signal = images[masks.astype(bool)].mean()
    contrast_err = measured_lesion_signal - desired_atten
    vol_err = study.results['LesionVolume(mL)'].sum() - desired_vol
    assert abs(contrast_err) / desired_atten < 0.2
    assert abs(vol_err) / desired_vol < 0.7
