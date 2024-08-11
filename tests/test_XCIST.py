from pathlib import Path
from shutil import rmtree

import numpy as np

from pedsilicoICH.image_acquisition import read_dicom, CTobj


def get_effective_diameter(ground_truth_mu, pixel_width_mm):
    A = np.sum(ground_truth_mu>-1000)*pixel_width_mm**2
    return 2*np.sqrt(A/np.pi)

base_dir = Path(__file__).parent
dcm = base_dir / '350mm_CTP404_groundtruth.dcm'
phantom = read_dicom(dcm)
phantom = np.repeat(phantom[None], 11, axis=0)
diameter_pixels = get_effective_diameter(phantom[0], 1)
known_diameter_mm = 200
fov_mm = phantom.shape[-1]*known_diameter_mm/diameter_pixels
dx = fov_mm/phantom.shape[-1]

test_dir = base_dir / 'test_result'
if Path(test_dir).exists():
    rmtree(test_dir)
ct = CTobj(phantom, spacings=3*[dx], patientname='test', output_dir=test_dir)

def test_scan():
    views=10
    ct.run_scan(views=views)
    ct.run_recon(sliceCount=1)
    dcms = ct.write_to_dicom(test_dir / 'test.dcm')
    from pathlib import Path
    dcms_in_dir = list(test_dir.glob('*.dcm'))
    assert(ct.recon.shape==(1, 512, 512))
    assert(ct.projections.shape==(views, ct.xcist.cfg.scanner.detectorRowCount, ct.xcist.cfg.scanner.detectorColCount))
    assert(len(dcms) == len(ct.start_positions))
    assert(dcms_in_dir == dcms)
