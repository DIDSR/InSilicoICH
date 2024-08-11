# %%
import numpy as np
from shutil import rmtree

from pedsilicoICH.image_acquisition import read_dicom, CTobj

dcm = '/gpfs_projects/brandon.nelson/RSTs/pediatricIQphantoms/CTP404/diameter_350mm/350mm_CTP404_groundtruth.dcm'
phantom = read_dicom(dcm)
phantom = np.repeat(phantom[None], 11, axis=0)
                    
# %%
from pathlib import Path

def get_effective_diameter(ground_truth_mu, pixel_width_mm):
    A = np.sum(ground_truth_mu>-1000)*pixel_width_mm**2
    return 2*np.sqrt(A/np.pi)

diameter_pixels = get_effective_diameter(phantom[0], 1)
known_diameter_mm = 200
fov_mm = phantom.shape[-1]*known_diameter_mm/diameter_pixels
fov_mm
dx = fov_mm/phantom.shape[-1]

if Path('test').exists():
    rmtree('test')
ct = CTobj(phantom, spacings=3*[dx], patientname='test')

def test_scan():
    views=10
    ct.run_scan(views=views)
    ct.run_recon(sliceCount=1)
    dcms = ct.write_to_dicom('test/test.dcm')
    from pathlib import Path
    dcms_in_dir = list(Path('test').glob('*.dcm'))
    assert(ct.recon.shape==(1, 512, 512))
    assert(ct.projections.shape==(views, ct.xcist.cfg.scanner.detectorRowCount, ct.xcist.cfg.scanner.detectorColCount))
    assert(len(dcms) == len(ct.start_positions))
    assert(dcms_in_dir == dcms)

# %%
