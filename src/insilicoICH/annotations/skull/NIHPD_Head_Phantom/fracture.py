"""
Export skull mesh into its segmentation.
"""

import os
import sys
import numpy as np
from skull import Skull
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import nrrd

main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
sys.path.append(main_directory)


class Fracture(Skull):
    def __init__(self):
        super().__init__()
        self.dist_seg_skull = None
        self.header = None

    def set_skull_seg(self, path_nifti: str) -> None:
        self.seg_skull, self.header = nrrd.read(path_nifti)

    def get_skull_parameters(self):
        # inverse_seg_skull = 1 - self.seg_skull
        self.dist_seg_skull = distance_transform_edt(self.seg_skull)
        # plt.imshow(self.dist_seg_skull[100], cmap="gray")
        


if __name__ == "__main__":
    obj_fracture = Fracture()
    obj_fracture.set_skull_seg(
        path_nifti="/home/dhaval.kadia/code/research/PedSilicoICH/PedSilicoICH/src/pedsilicoICH/annotations/skull/NIHPD_Head_Phantom/assets/segmentation_skull.nrrd"
    )
    obj_fracture.get_skull_parameters()
    obj_fracture.save_nrrd(obj_fracture.dist_seg_skull, obj_fracture.header, "/home/dhaval.kadia/code/research/PedSilicoICH/PedSilicoICH/src/pedsilicoICH/annotations/skull/NIHPD_Head_Phantom/assets/dist_segmentation_skull.nrrd")
