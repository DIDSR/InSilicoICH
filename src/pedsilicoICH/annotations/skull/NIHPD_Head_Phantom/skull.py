"""
Class Skull for various operations.
"""

import numpy as np


class Skull(object):
    def __init__(self, mesh_brainmask):
        self.mesh_brainmask = mesh_brainmask
        self.mesh_skull = None

    def compute_normals(self, mesh):
        pass

    def mesh_to_segmentation(self, mesh):
        pass

    def save_mesh(self, mesh, filepath):
        pass
