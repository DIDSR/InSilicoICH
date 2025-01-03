"""
Extract skull from brain mask (or skull-stripped brain segmentation).
"""

import os
import sys
import numpy as np
import pyvista as pv
import vtk
from skull import Skull

main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
sys.path.append(main_directory)


class SkullProcess(Skull):
    def __init__(self, path_mesh_brainmask: str):
        self.mesh_brain = self.load_mesh(path_mesh_brainmask)
        super().__init__(self.mesh_brain)

    def load_mesh(self, path_mesh_brainmask: str):
        return pv.read(path_mesh_brainmask)

    def compute_normals(self, mesh) -> pv.PolyData:
        return mesh.compute_normals()

    def save_mesh(self, mesh: pv.PolyData, filepath: str):
        mesh.save(filepath)
        print("Saved", filepath)

    def _get_bounding_sphere_mesh(self, mesh: pv.PolyData) -> pv.PolyData:
        """
        Estimate the sphere bounding brain, visually check it in a 3D viewer (e.g. 3D Slicer) to estimate value_division that will allow to retain only skull region, and exclude extra cells for skull mesh extraction.

        Args:
            mesh (pv.PolyData): Brain mesh.

        Returns:
            pv.PolyData: Sphere used for extracting skull mesh.
        """
        # Get points as numpy array
        points = mesh.points

        # Convert points to flat array of doubles
        pts_flat = points.ravel()

        # Create array to store sphere parameters
        sphere = np.zeros(4, dtype=np.float64)

        # Create hints array (any two valid indices will work)
        hints = np.array([0, 1], dtype=np.int64)

        # Compute bounding sphere
        vtk.vtkSphere.ComputeBoundingSphere(pts_flat, points.shape[0], sphere, hints)

        # Extract results
        center = sphere[0:3]
        radius = sphere[3]

        # Make the sphere smaller (value is decided based on visualization, specifically to extract skull)
        value_division = 2
        radius_to_consider = radius / value_division

        # Create mesh
        mesh_sphere = pv.Sphere(center=center, radius=radius_to_consider)

        return mesh_sphere, center, radius_to_consider

    def remove_cells_inside_sphere(
        self, mesh: pv.PolyData, center: list[float], radius: float
    ):
        # Convert mesh to unstructured grid for ghost cell support
        grid = mesh.cast_to_unstructured_grid()

        # Create mask for cells inside sphere
        cell_centers = grid.cell_centers().points
        distances = np.linalg.norm(cell_centers - center, axis=1)
        inside_cells = np.argwhere(distances < radius).ravel()

        # Remove the ghost cells
        grid.remove_cells(inside_cells, inplace=True)

        # Convert back to PolyData if needed
        return grid.extract_surface()

    def extract_skull(self) -> None:
        """
        Calculate surface normals and remove certain cells (mesh).

        Polar angle for pv.cartesian_to_spherical(): https://docs.pyvista.org/api/utilities/_autosummary/pyvista.cartesian_to_spherical
        """
        mesh = self.compute_normals(self.mesh_brainmask)

        # Compute the normals in-place
        normals = mesh["Normals"]

        # Convert normals from cartesian to spherical coordinates
        r, phi, theta = pv.cartesian_to_spherical(
            normals[:, 0], normals[:, 1], normals[:, 2]
        )

        # phi is the polar angle (in radians)
        # Convert to degrees for comparison
        phi_degrees = np.rad2deg(phi)
        threshold_degree = 100

        # Find cells where phi > 90 degrees
        cells_to_remove = np.where(phi_degrees > threshold_degree)[0]
        print("Number of cells removed =", len(cells_to_remove))

        # Remove the identified cells
        mesh.remove_cells(cells_to_remove, inplace=True)

        mesh_sphere, center, radius = self._get_bounding_sphere_mesh(mesh)
        mesh = self.remove_cells_inside_sphere(mesh=mesh, center=center, radius=radius)

        self.mesh_skull = mesh


if __name__ == "__main__":
    path_mesh_brainmask = os.path.join(
        main_directory,
        "src/pedsilicoICH/annotations/skull/NIHPD_Head_Phantom/assets",
        "mesh_brain.vtk",
    )

    object_skull_process = SkullProcess(path_mesh_brainmask=path_mesh_brainmask)
    object_skull_process.extract_skull()
    object_skull_process.save_mesh(
        mesh=object_skull_process.mesh_skull,
        filepath=os.path.join(
            main_directory,
            "src/pedsilicoICH/annotations/skull/NIHPD_Head_Phantom/assets",
            "mesh_skull.vtk",
        ),
    )
