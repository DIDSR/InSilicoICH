"""
Generate skull fracture with the given mess/voxel data.
"""

import os
import sys
import nibabel as nib
from .process_skull import SkullProcess

main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
sys.path.append(main_directory)


class SkullFracture(SkullProcess):
    def __init__(self, path_mesh_brainmask: str, path_mask_brain: str, path_file_config: str, path_skull_mesh: str):
        super().__init__(path_mesh_brainmask=path_mesh_brainmask, path_mask_brain=path_mask_brain, path_file_config=path_file_config)
        self.mesh_brain = None
        self.skull_center = None
        self.config = None
        self.path_skull_mesh = path_skull_mesh

    def load_data(self):
        self.skull_mesh = self.load_mesh(self.path_skull_mesh)
        self.skull_mesh = self.skull_mesh.extract_surface()
        self.skull_center = self.config["skull_mesh_params"]["center"]


# if __name__ == "__main__":
#     path_mesh_brainmask = os.path.join(
#         main_directory,
#         "src/insilicoICH/annotations/skull/NIHPD_Head_Phantom/assets",
#         "mesh_brain.vtk",
#     )

#     path_mask_brain = os.path.join(
#         main_directory, "src/NIHPD_Head_Phantom", "nihpd_asym_04.5-08.5_mask.nii"
#     )

#     path_file_config = os.path.join(
#         main_directory,
#         "src/insilicoICH/annotations/skull/NIHPD_Head_Phantom/assets",
#         "config.toml",
#     )

#     path_skull_mesh = os.path.join(
#         main_directory,
#         "src/insilicoICH/annotations/skull/NIHPD_Head_Phantom/assets",
#         "skull_mesh.vtk",
#     )

#     object_skull_fracture = SkullFracture(
#         path_mesh_brainmask=path_mesh_brainmask, path_mask_brain=path_mask_brain, path_file_config=path_file_config,
#         path_skull_mesh=path_skull_mesh
#     )
#     object_skull_fracture.initialize()
#     object_skull_fracture.load_data()

#     # Save particular fracture generation
#     nifti_fracture_seg = object_skull_fracture.get_nifti_fracture(length=200, phi_degree=30, theta_degree=0)
#     PATH_DIR_SAVE_SAMPLE = ""
#     nib.save(nifti_fracture_seg, os.path.join(PATH_DIR_SAVE_SAMPLE, "NIHPD_Head_Phantom_fracture_sample.nii.gz"))

#     # Save multiple fractures
#     object_skull_fracture.save_fractures(PATH_DIR_SAVE_SAMPLE)
