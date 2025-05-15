'''
tests low level CT simulation functionality, using XCIST
'''
from pathlib import Path
from shutil import rmtree

import numpy as np

from insilicoICH.phantoms.base_phantoms import Phantom
from insilicoICH.study import Scanner, load_phantom
from insilicoICH.image_acquisition import read_dicom, Scanner


test_dir = Path(__file__).parent.absolute()


def get_effective_diameter(ground_truth_mu, pixel_width_mm):
    '''
    effective diameter defined in AAPM TG204:
       https://www.aapm.org/pubs/reports/RPT_204.pdf
    '''
    A = np.sum(ground_truth_mu > -1000)*pixel_width_mm**2
    return 2*np.sqrt(A/np.pi)


def scan_CTP404(test_dir, views=100):
    result_dir = test_dir / 'test_result'
    if Path(result_dir).exists():
        rmtree(result_dir)
    dcm = test_dir / 'CTP404_groundtruth.dcm'

    img = np.repeat(read_dicom(dcm)[None], 11, axis=0)
    diameter_pixels = get_effective_diameter(img[0], 1)
    known_diameter_mm = 200
    fov_mm = img.shape[-1]*known_diameter_mm/diameter_pixels
    dx = fov_mm/img.shape[-1]
    dy = dx
    dz = 1
    phantom = Phantom(img, spacings=[dz, dx, dy])
    ct = Scanner(phantom, output_dir=result_dir)
    ct.run_scan(views=views)
    ct.run_recon(sliceThickness=6)
    return ct


def test_scan_shape():
    '''
    basic test of xcist simulations
    '''
    views = 100

    ct = scan_CTP404(test_dir, views)
    dcms = ct.write_to_dicom(ct.output_dir / 'test.dcm')
    dcms_in_dir = list(ct.output_dir.glob('*.dcm'))
    assert ct.recon.mean() > -800
    assert ct.recon.shape == (1, 512, 512)
    assert ct.projections.shape == (views,
                                    ct.xcist.cfg.scanner.detectorRowCount,
                                    ct.xcist.cfg.scanner.detectorColCount)
    assert len(dcms) == ct.recon.shape[0]
    assert dcms_in_dir == dcms


def test_load_scanner_config():
    phantom = load_phantom(6.5)
    scanner = Scanner(phantom, 'Siemens_DefinitionFlash')
    original_sid = scanner.xcist.cfg.scanner.sid
    original_collimation = scanner.nominal_aperature
    scanner.load_scanner_config(test_dir / 'Scanner_Test')
    new_sid = scanner.xcist.cfg.scanner.sid
    new_collimation = scanner.nominal_aperature
    assert new_sid != original_sid
    assert new_collimation != original_collimation


def test_load_custom_scanner():
    phantom = load_phantom(6.5)
    custom_scanner = test_dir / 'Scanner_Test'
    scanner = Scanner(phantom, custom_scanner)
    assert scanner.xcist.cfg.scanner.sid == 42
