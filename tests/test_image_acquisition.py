'''
tests low level CT simulation functionality, using XCIST
'''
from pathlib import Path
from shutil import rmtree

import numpy as np

from pedsilicoICH.ground_truth_definition.phantoms import Phantom
from pedsilicoICH.image_acquisition import read_dicom, Scanner


def get_effective_diameter(ground_truth_mu, pixel_width_mm):
    '''
    effective diameter defined in AAPM TG204:
       https://www.aapm.org/pubs/reports/RPT_204.pdf
    '''
    A = np.sum(ground_truth_mu > -1000)*pixel_width_mm**2
    return 2*np.sqrt(A/np.pi)


test_dir = Path(__file__).parent.absolute()
print(test_dir)
dcm = test_dir / 'CTP404_groundtruth.dcm'

img = np.repeat(read_dicom(dcm)[None], 11, axis=0)
diameter_pixels = get_effective_diameter(img[0], 1)
known_diameter_mm = 200
fov_mm = img.shape[-1]*known_diameter_mm/diameter_pixels
dx = fov_mm/img.shape[-1]

phantom = Phantom(img, spacings=3*[dx])

result_dir = test_dir / 'test_result'
if Path(result_dir).exists():
    rmtree(result_dir)

ct = Scanner(phantom, output_dir=result_dir)

views = 100
ct.run_scan(views=views)


def test_scan_shape():
    '''
    basic test of xcist simulations
    '''
    ct.run_recon(sliceCount=1)
    dcms = ct.write_to_dicom(result_dir / 'test.dcm')
    dcms_in_dir = list(result_dir.glob('*.dcm'))
    assert ct.recon.mean() > -800
    assert ct.recon.shape == (1, 512, 512)
    assert ct.projections.shape == (views,
                                    ct.xcist.cfg.scanner.detectorRowCount,
                                    ct.xcist.cfg.scanner.detectorColCount)
    assert len(dcms) == len(ct.start_positions)
    assert dcms_in_dir == dcms


def test_get_lesion_mask():
    ct.run_recon(sliceThickness=1)
    mask = ct.get_lesion_mask()
    assert not mask
