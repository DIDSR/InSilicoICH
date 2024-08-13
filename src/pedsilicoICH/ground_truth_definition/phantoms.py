'''
module for working with phantoms
'''

from pathlib import Path
from . import dicom_to_voxelized_phantom


def voxelize_ground_truth(dicom_path, phantom_path, material_threshold_dict=None):
    """
    Used to convert ground truth image into segmented volumes used by XCIST to run simulations

    Inputs:
    dicom_path    (string)           Path where the DICOM images are located.
    phantom_path  (string)           Path where the phantom files are to be written
    dicom_path [str]: directory containing ground truth dicom images, these are typically the output of `convert_to_dicom`
    material_threshold_dict [dict]: dictionary mapping XCIST materials to appropriate lower thresholds in the ground truth image, see the .cfg here for examples <https://github.com/xcist/phantoms-voxelized/tree/main/DICOM_to_voxelized>
    """
    nfiles = len(list(Path(dicom_path).rglob('*.dcm')))
    slice_range = list(range(nfiles))
    if not material_threshold_dict:
        material_threshold_dict = dict(zip(
        ['ICRU_lung_adult_healthy', 'ICRU_adipose_adult2', 'ICRU_liver_adult', 'water', 'ICRU_skeleton_cortical_bone_adult'],
        [-1000, -200, 0, 100, 300]))

    cfg_file_str = f"""
# Path where the DICOM images are located:
phantom.dicom_path = '{dicom_path}'
# Path where the phantom files are to be written (the last folder name will be the phantom files' base name):
phantom.phantom_path = '{phantom_path}'
phantom.materials = {list(material_threshold_dict.keys())}
phantom.mu_energy = 60                  # Energy (keV) at which mu is to be calculated for all materials.
phantom.thresholds = {list(material_threshold_dict.values())}	# Lower threshold (HU) for each material.
phantom.slice_range = [{[slice_range[0], slice_range[-1]]}]			  # Range of DICOM image numbers to include. (first, last slice)
phantom.show_phantom = False                # Flag to turn on/off image display.
phantom.overwrite = True                   # Flag to overwrite existing files without warning.
"""

    dicom_to_voxel_cfg = phantom_path / 'dicom_to_voxelized.cfg'

    with open(dicom_to_voxel_cfg, 'w') as f:
        f.write(cfg_file_str)

    dicom_to_voxelized_phantom.run_from_config(dicom_to_voxel_cfg)
