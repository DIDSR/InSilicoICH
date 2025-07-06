"""
Class Skull for various operations.
"""


class Skull(object):
    def __init__(self):
        self.mesh_brainmask = None
        self.skull_mesh = None
        self.skull_seg = None
        self.skull_parameters = None

    def compute_normals(self, mesh):
        pass

    def mesh_to_segmentation(self, mesh):
        pass

    def save_mesh(self, mesh, filepath):
        pass
