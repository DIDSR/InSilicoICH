"""
Class Skull for various operations.
"""


class Skull(object):
    def __init__(self):
        self.mesh_brainmask = None
        self.mesh_skull = None
        self.seg_skull = None
        self.skull_parameters = None

    def compute_normals(self, mesh):
        pass

    def mesh_to_segmentation(self, mesh):
        pass

    def save_mesh(self, mesh, filepath):
        pass
