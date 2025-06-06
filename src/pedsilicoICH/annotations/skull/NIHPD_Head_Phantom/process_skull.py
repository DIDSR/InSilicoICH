"""
Extract skull from brain mask (or skull-stripped brain segmentation).
"""

import os
import sys
import numpy as np
import pyvista as pv
import vtk
from skull import Skull
from pyransac3d import Sphere
import random
import nibabel as nib

main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
sys.path.append(main_directory)


class SkullProcess(Skull):
    def __init__(self, path_mesh_brainmask: str):
        super().__init__()
        self.mesh_brain = self.load_mesh(path_mesh_brainmask)
        self.skull_center = None

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
    
    def _check(self):
        # Load the mesh
        mesh = self.mesh_brain.extract_geometry()

        # Set voxel size
        voxel_size = 1

        # Get cell centers (triangles/quads/etc.)
        cell_centers = mesh.cell_centers().points

        # Compute voxel grid bounds
        min_bounds = cell_centers.min(axis=0)
        max_bounds = cell_centers.max(axis=0)
        # dims = np.ceil((max_bounds - min_bounds) / voxel_size).astype(int)
        dims = [197, 233, 189]

        print("dims", dims)

        # Initialize empty voxel grid
        voxels = np.zeros(dims, dtype=np.uint8)

        # Fill voxel positions using cell centers
        for pt in cell_centers:
            idx = ((pt - min_bounds) / voxel_size).astype(int)
            if np.all(idx >= 0) and np.all(idx < dims):
                voxels[tuple(np.add(idx, [24, 24, 0]))] = 1
        
        voxels = np.flip(voxels, axis=1)
        
        path_mask_brain = os.path.join(
            main_directory, "src/NIHPD_Head_Phantom", "nihpd_asym_04.5-08.5_mask.nii"
        )

        shape, origin, spacing, affine, array  = self.get_nifti_info(path_mask_brain)
        # print('shape', shape)
        # print('unique', np.unique(array))

        # indices = np.argwhere(array.astype(int) == 1)
        # offset = [np.min(indices[:, 0]), np.min(indices[:, 0]), np.min(indices[:, 0])]
        # centroid = [np.mean(indices[:, 0]), np.mean(indices[:, 0]), np.mean(indices[:, 0])]
        # print(offset)

        nifti_img = nib.Nifti1Image(voxels.astype(np.uint8), affine)

        # Save as numpy voxel grid
        nib.save(
            nifti_img,
            os.path.join(
                main_directory,
                "src/pedsilicoICH/annotations/skull/NIHPD_Head_Phantom/assets",
                "mesh_brain_voxel.nii.gz",
            ),
        )

    def extract_primary_skull_mesh(self, maxIteration=1000):
        """
        Removes the skull mesh from the lower hemisphere. (not in use. function needs to be updated)
        """
        grid = self.mesh_skull.cast_to_unstructured_grid()

        np_points = np.array(
            [grid.GetPoint(i) for i in range(grid.GetNumberOfPoints())]
        )

        # Detect a sphere using RANSAC
        sphere = Sphere()
        # thresh=0.01, maxIteration=1000
        self.skull_center, radius, inliers = sphere.fit(
            np_points, maxIteration=maxIteration
        )
        z_center = self.skull_center[2]

        # Get cell centers
        cell_centers = grid.cell_centers()
        z_centers = cell_centers.points[:, 2]

        # Find indices where cell center z < sphere center z
        offset_relative_radii = 0  # future use
        cell_ids_below = np.where(
            z_centers < (z_center - offset_relative_radii * radius)
        )[0]

        # Remove those cells
        grid.remove_cells(cell_ids_below, inplace=True)
        self.mesh_skull = grid.extract_surface()

    def get_nifti_info(self, nifti_path):
        img = nib.load(nifti_path)
        array = img.get_fdata()
        shape = img.shape                      # (z, y, x) or (x, y, z), depending on orientation
        affine = img.affine                    # 4x4 affine transformation matrix
        spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))  # voxel spacing
        origin = affine[:3, 3]                 # origin (translation component)

        return shape, origin, spacing, affine, array

    def mesh_to_voxel_center(self):
        # Load the mesh
        mesh = self.mesh_skull.extract_geometry()

        # Set voxel size
        voxel_size = 1

        # Get cell centers (triangles/quads/etc.)
        cell_centers = mesh.cell_centers().points

        # Compute voxel grid bounds
        min_bounds = cell_centers.min(axis=0)
        max_bounds = cell_centers.max(axis=0)
        dims = np.ceil((max_bounds - min_bounds) / voxel_size).astype(int)

        print("dims", dims)

        # Initialize empty voxel grid
        voxels = np.zeros(dims, dtype=np.uint8)

        # Fill voxel positions using cell centers
        for pt in cell_centers:
            idx = ((pt - min_bounds) / voxel_size).astype(int)
            if np.all(idx >= 0) and np.all(idx < dims):
                voxels[tuple(idx)] = 1

        path_mask_brain = os.path.join(
            main_directory, "src/NIHPD_Head_Phantom", "nihpd_asym_04.5-08.5_mask.nii"
        )

        shape, origin, spacing, affine, array  = self.get_nifti_info(path_mask_brain)

        nifti_img = nib.Nifti1Image(voxels.astype(np.uint8), affine)

        # Save as numpy voxel grid
        nib.save(
            nifti_img,
            os.path.join(
                main_directory,
                "src/pedsilicoICH/annotations/skull/NIHPD_Head_Phantom/assets",
                "mesh_skull_voxel.nii.gz",
            ),
        )

    def remove_voxel_spherical_coordi(self, list_start=None, list_direction=None):
        """
        Remove the mesh intersecting the gien vector.
        """
        # Define line segment
        list_stop = [
            np.add(start, direction)
            for start, direction in zip(list_start, list_direction)
        ]

        list_indice_intersect = []

        # Perform ray trace
        for start, stop in zip(list_start, list_stop):
            points, inds = self.mesh_skull.ray_trace(start, stop)
            list_indice_intersect.extend(inds)

        self.mesh_skull = self.mesh_skull.extract_surface()

        if len(list_indice_intersect) > 0:
            non_intersected = np.setdiff1d(
                np.arange(self.mesh_skull.n_cells), list_indice_intersect
            )
            self.mesh_skull = self.mesh_skull.extract_cells(non_intersected)

    def get_angular_spacing_specific_cell_trace_degree(self, mesh, start, stop):
        """
        Get angular spacing for shifting the angle for removing mesh to add fracture.
        (start is the center of skull here)
        """
        # Perform ray tracing
        points, ind = mesh.ray_trace(start, stop)

        if len(ind) == 0:
            print("No intersection found.")
        else:
            hit_cell_index = ind[0]

            # Details about the target cell
            center_hit_cell = mesh.cell_centers().points[hit_cell_index]
            cell_center_rel_origin = np.subtract(center_hit_cell, start)
            r_hit_cell, phi_hit_cell, theta_hit_cell = pv.cartesian_to_spherical(
                *cell_center_rel_origin
            )

            list_neighbor_phi_radian = []
            list_neighbor_theta_radian = []
            neighbors = mesh.cell_neighbors(hit_cell_index)
            for i, neighbor in enumerate(neighbors):
                pt = mesh.cell_centers().points[neighbor]
                pt_rel_origin = np.subtract(pt, start)
                r, phi, theta = pv.cartesian_to_spherical(*pt_rel_origin)
                list_neighbor_phi_radian.append(phi)
                list_neighbor_theta_radian.append(theta)

            list_angular_spacing_radian_phi = [
                np.absolute(np.subtract(neighbor_phi, phi_hit_cell))
                for neighbor_phi in list_neighbor_phi_radian
            ]
            list_angular_spacing_radian_theta = [
                np.absolute(np.subtract(neighbor_theta, theta_hit_cell))
                for neighbor_theta in list_neighbor_theta_radian
            ]

            average_list_angular_spacing_degree_phi = np.rad2deg(
                np.average(list_angular_spacing_radian_phi)
            )
            average_list_angular_spacing_degree_theta = np.rad2deg(
                np.average(list_angular_spacing_radian_theta)
            )

            return (
                average_list_angular_spacing_degree_phi,
                average_list_angular_spacing_degree_theta,
            )

    def add_fracture(self):
        """
        Add fracture by removing meshes with given procedure.
        """
        phi_degree = 30
        theta_degree = 0

        # Degrees to shift to next point
        direction = pv.spherical_to_cartesian(
            1, np.deg2rad(phi_degree), np.deg2rad(theta_degree)
        )
        delta_shift_degree_phi, delta_shift_degree_theta = (
            self.get_angular_spacing_specific_cell_trace_degree(
                self.mesh_skull,
                self.skull_center,
                np.add(self.skull_center, np.multiply(direction, 100)),
            )
        )

        # Allow some duplicates for better continuity
        continuity_factor = 0.8
        delta_shift_degree_phi *= continuity_factor
        delta_shift_degree_theta *= continuity_factor

        # Number of iterations to try removing cells to remove (depends on direction, may have duplicates (less number of removals))
        n_iterations = 3000

        list_start = []
        list_direction = []

        list_switch = [2, 3]

        switch_counter = 1
        list_reset_counter_wait = [10, 20, 30, 40, 50]
        list_reset_counter_wait_index = 0
        pointer = list_switch[random.randint(0, len(list_switch) - 1)]

        for i in range(n_iterations):
            direction = pv.spherical_to_cartesian(
                1, np.deg2rad(phi_degree), np.deg2rad(theta_degree)
            )
            list_start.append(self.skull_center)
            list_direction.append(np.multiply(direction, 100))

            if (
                switch_counter % list_reset_counter_wait[list_reset_counter_wait_index]
                == 0
            ):
                list_reset_counter_wait_index = random.randint(
                    0, len(list_reset_counter_wait) - 1
                )
                switch_counter = 1
                pointer = list_switch[random.randint(0, len(list_switch) - 1)]

            switch_counter += 1

            if pointer == 0:
                phi_degree += delta_shift_degree_phi * random.choice([-1, 1])
            elif pointer == 1:
                theta_degree += delta_shift_degree_theta * random.choice([-1, 1])
            elif pointer == 2:
                phi_degree += delta_shift_degree_phi * random.choice([-1, 1])
                theta_degree += delta_shift_degree_theta
            elif pointer == 3:
                phi_degree += delta_shift_degree_phi
                theta_degree += delta_shift_degree_theta * random.choice([-1, 1])
            elif pointer == 4:
                phi_degree += delta_shift_degree_phi * random.choice([-1, 1])
                theta_degree += delta_shift_degree_theta * random.choice([-1, 1])

        self.remove_voxel_spherical_coordi(list_start, list_direction)

    def extract_skull(self) -> None:
        """
        Calculate surface normals and remove certain cells (mesh).

        Polar angle for pv.cartesian_to_spherical(): https://docs.pyvista.org/api/utilities/_autosummary/pyvista.cartesian_to_spherical
        """
        mesh = self.compute_normals(self.mesh_brain)

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

        # Find cells where phi > threshold_degree
        cells_to_remove = np.where(phi_degrees > threshold_degree)[0]
        # print("Number of cells removed =", len(cells_to_remove))

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
    object_skull_process._check()
    # object_skull_process.extract_skull()
    # object_skull_process.extract_primary_skull_mesh()
    # object_skull_process.add_fracture()

    # object_skull_process.save_mesh(
    #     mesh=object_skull_process.mesh_skull,
    #     filepath=os.path.join(
    #         main_directory,
    #         "src/pedsilicoICH/annotations/skull/NIHPD_Head_Phantom/assets",
    #         "mesh_skull.vtk",
    #     ),
    # )

    # object_skull_process.mesh_to_voxel_center()
