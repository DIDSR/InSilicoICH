import os
import sys
from pathlib import Path
from skull_fracture import SkullFracture

main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
sys.path.append(main_directory)

dict_skull_paths = {
        "path_mesh_brainmask": os.path.join(
        main_directory,
        "src/insilicoICH/annotations/skull/NIHPD_Head_Phantom/assets",
        "mesh_brain.vtk",
    ),
        "path_mask_brain": os.path.join(
        main_directory, "src/NIHPD_Head_Phantom", "nihpd_asym_04.5-08.5_mask.nii"
    ),
        "path_file_config": os.path.join(
        main_directory,
        "src/insilicoICH/annotations/skull/NIHPD_Head_Phantom/assets",
        "config.toml",
    ),
        "path_skull_mesh": os.path.join(
        main_directory,
        "src/insilicoICH/annotations/skull/NIHPD_Head_Phantom/assets",
        "skull_mesh.vtk",
    )
    }

def fetch_fracture():
    """
    Fetch fracture for NIHPD from SkullFracture class.
    """
    object_skull_fracture = SkullFracture(
        path_mesh_brainmask=dict_skull_paths["path_mesh_brainmask"], path_mask_brain=dict_skull_paths["path_mask_brain"], path_file_config=dict_skull_paths["path_file_config"],
        path_skull_mesh=dict_skull_paths["path_skull_mesh"]
    )
    object_skull_fracture.initialize()
    object_skull_fracture.load_data()

    # Save particular fracture generation
    nifti_fracture_seg = object_skull_fracture.get_nifti_fracture(length=200, phi_degree=30, theta_degree=0)

    return nifti_fracture_seg

data = fetch_fracture()
print(type(data))
print(data.get_fdata().shape)
