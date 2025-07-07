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
import tomli
import tomli_w

main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
sys.path.append(main_directory)


class SkullProcess(Skull):
    def __init__(self, path_mesh_brainmask: str, path_mask_brain: str, path_file_config: str):
        super().__init__()
        self.path_mesh_brainmask = path_mesh_brainmask
        self.mesh_brain = None
        self.skull_center = None
        self.path_mask_brain = path_mask_brain
        self.config = None
        self.path_file_config = path_file_config
        self.skull_mesh = None
        self.skull_fracture_mesh = None
        self.threshold_degree_phi = 100
        self.nifti_skull_seg = None

    def load_mesh(self, path_mesh_brainmask: str):
        return pv.read(path_mesh_brainmask)

    def compute_normals(self, mesh) -> pv.PolyData:
        return mesh.compute_normals()

    def save_mesh(self, mesh: pv.PolyData, filepath: str):
        mesh.save(filepath)
        print("Saved", filepath)

    def initialize(self) -> None:
        """
        Initialize with the configuration file.
        """        
        if not os.path.exists(self.path_file_config):
            with open(self.path_file_config, "wb") as f:
                tomli_w.dump({}, f)
            print("Created empty config.toml")

        with open(self.path_file_config, "rb") as file_config:
            self.config = tomli.load(file_config)
            print("Reading", self.path_file_config)

        self.mesh_brain = self.load_mesh(self.path_mesh_brainmask)

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

    def extract_primary_skull_mesh(self, maxIteration=1000):
        """
        Removes the skull mesh from the lower hemisphere. (not in use. function needs to be updated)
        """
        grid = self.skull_mesh.cast_to_unstructured_grid()

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

        self.config.setdefault("skull_mesh_params", {})
        self.config["skull_mesh_params"]["center"] = self.skull_center

        with open(self.path_file_config, "wb") as f:
            tomli_w.dump(self.config, f)

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
        self.skull_mesh = grid.extract_surface()

    def get_nifti_info(self, nifti_path):
        img = nib.load(nifti_path)
        array = img.get_fdata()
        shape = img.shape  # (z, y, x) or (x, y, z), depending on orientation
        affine = img.affine  # 4x4 affine transformation matrix
        spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))  # voxel spacing
        origin = affine[:3, 3]  # origin (translation component)

        return shape, origin, spacing, affine, array

    def get_center_voxel_space(self):
        """
        While extracting the skull, center of the mesh is lost. This method provides the skull center relative to voxel space (from mesh space), to be used later while converting to voxels.
        """
        shape, origin, spacing, affine, array = self.get_nifti_info(
            self.path_mask_brain
        )

        # Load the mesh
        mesh = self.mesh_brain.extract_geometry()

        # Set voxel size
        voxel_size = 1

        # Get cell centers (triangles/quads/etc.)
        cell_centers = mesh.cell_centers().points

        # Compute voxel grid bounds
        min_bounds = cell_centers.min(axis=0)

        idx = ((self.skull_center - min_bounds) / voxel_size).astype(int)

        return idx

    def get_nifti_fracture_seg(
        self, nifti_skull_seg, nifti_skull_fracture_seg):
        affine = nifti_skull_seg.affine

        array_skull = nifti_skull_seg.get_fdata()
        array_skull_fracture = nifti_skull_fracture_seg.get_fdata()

        diff_fracture = array_skull - array_skull_fracture

        nifti_fracture_seg = nib.Nifti1Image(diff_fracture, affine)

        return nifti_fracture_seg

    def save_seg_fracture(
        self, path_seg_skull, path_seg_skull_fracture, path_save_seg_fracture
    ):
        shape, origin, spacing, affine, array_skull = self.get_nifti_info(
            path_seg_skull
        )

        shape, origin, spacing, affine, array_skull_fracture = self.get_nifti_info(
            path_seg_skull_fracture
        )

        diff_fracture = array_skull - array_skull_fracture

        nifti_img = nib.Nifti1Image(diff_fracture, affine)

        # Save as numpy voxel grid
        nib.save(nifti_img, path_save_seg_fracture)

    def _mesh_to_voxel_nifti(self, path_nifti_save):
        # Load NIfTI info
        shape, origin, spacing, affine, array = self.get_nifti_info(self.path_mask_brain)
        indices = np.argwhere(array.astype(int) == 1)
        offset = [np.min(indices[:, 0]) + 1, np.min(indices[:, 1]) - 2, np.min(indices[:, 2])]

        # Extract mesh geometry
        mesh = self.mesh_brain.extract_geometry()

        # Set voxel size
        voxel_size = 1

        # Get cell centers (triangles/quads/etc.)
        cell_centers = mesh.cell_centers().points

        # Compute voxel grid bounds
        min_bounds = cell_centers.min(axis=0)

        # Convert mesh points from world space to voxel indices using inverse affine
        inv_affine = np.linalg.inv(affine)
        voxel_indices = nib.affines.apply_affine(inv_affine, cell_centers)
        voxel_indices = np.round(voxel_indices).astype(int)

        # Initialize empty voxel grid
        voxels = np.zeros(shape, dtype=np.uint8)

        # Fill voxel positions using cell centers
        for pt in cell_centers:
            idx = ((pt - min_bounds) / voxel_size).astype(int)
            if np.all(idx >= 0) and np.all(idx < shape):
                voxels[tuple(np.add(idx, offset))] = 1

        # (Optional) Flip axes if needed (use only if verified visually)
        # voxels = np.flip(voxels, axis=1)
        voxels = np.rot90(voxels, axes=(0, 1), k=2)

        # Create and save NIfTI
        nifti_img = nib.Nifti1Image(voxels.astype(np.uint8), affine)
        nib.save(nifti_img, path_nifti_save)

    def mesh_to_voxel_nifti_skull(self, mesh, path_nifti_save=None):
        # Load NIfTI info
        shape, origin, spacing, affine, array = self.get_nifti_info(self.path_mask_brain)
        indices = np.argwhere(array.astype(int) == 1)
        offset = [np.min(indices[:, 0]) + 1, np.min(indices[:, 1]) - 2, np.min(indices[:, 2])]
        offset[2] = self.get_center_voxel_space()[2]

        # Extract mesh geometry
        mesh = mesh.extract_geometry()

        # Set voxel size
        voxel_size = 1

        # Get cell centers (triangles/quads/etc.)
        cell_centers = mesh.cell_centers().points

        # Compute voxel grid bounds
        min_bounds = cell_centers.min(axis=0)

        # Convert mesh points from world space to voxel indices using inverse affine
        inv_affine = np.linalg.inv(affine)
        voxel_indices = nib.affines.apply_affine(inv_affine, cell_centers)
        voxel_indices = np.round(voxel_indices).astype(int)

        # Initialize empty voxel grid
        voxels = np.zeros(shape, dtype=np.uint8)

        # # Fill in voxels
        # for idx in voxel_indices:
        #     if np.all(idx >= 0) and np.all(idx < shape):
        #         voxels[tuple(idx)] = 1

        # Fill voxel positions using cell centers
        for pt in cell_centers:
            idx = ((pt - min_bounds) / voxel_size).astype(int)
            if np.all(idx >= 0) and np.all(idx < shape):
                voxels[tuple(np.add(idx, offset))] = 1

        # (Optional) Flip axes if needed (use only if verified visually)
        # voxels = np.flip(voxels, axis=1)
        voxels = np.rot90(voxels, axes=(0, 1), k=2)

        # Create and save NIfTI
        nifti_img = nib.Nifti1Image(voxels.astype(np.uint8), affine)

        if path_nifti_save is not None:
            nib.save(nifti_img, path_nifti_save)

        return nifti_img

    def get_neighbour_cells_info(self, mesh, ind_maincell):
        """
        Get the list of indices of the neighbor cells.
        """
        neighbors = mesh.cell_neighbors(ind_maincell)

        return neighbors

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

        self.skull_fracture_mesh = self.skull_mesh.copy()

        # Perform ray trace
        for start, stop in zip(list_start, list_stop):
            points, inds = self.skull_fracture_mesh.ray_trace(start, stop)
            list_indice_intersect.extend(inds)

            for ind in inds:
                list_indice_intersect.extend(
                    self.get_neighbour_cells_info(self.skull_fracture_mesh, ind)
                )

        self.skull_fracture_mesh = self.skull_fracture_mesh.extract_surface()

        if len(list_indice_intersect) > 0:
            non_intersected = np.setdiff1d(
                np.arange(self.skull_fracture_mesh.n_cells), list_indice_intersect
            )
            self.skull_fracture_mesh = self.skull_fracture_mesh.extract_cells(non_intersected)

    def get_angular_spacing_specific_cell_trace_degree(self, start, stop):
        """
        Get angular spacing for shifting the angle for removing mesh to add fracture.
        (start is the center of skull here)
        """
        # Perform ray tracing
        points, ind = self.skull_mesh.ray_trace(start, stop)

        if len(ind) == 0:
            print("No intersection found.")
        else:
            hit_cell_index = ind[0]

            # Details about the target cell
            center_hit_cell = self.skull_mesh.cell_centers().points[hit_cell_index]
            cell_center_rel_origin = np.subtract(center_hit_cell, start)
            r_hit_cell, phi_hit_cell, theta_hit_cell = pv.cartesian_to_spherical(
                *cell_center_rel_origin
            )

            list_neighbor_phi_radian = []
            list_neighbor_theta_radian = []
            neighbors = self.skull_mesh.cell_neighbors(hit_cell_index)
            for i, neighbor in enumerate(neighbors):
                pt = self.skull_mesh.cell_centers().points[neighbor]
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

    def save_fractures(self, dir_save_nifti_fracture: str, n_fractures=10, min_max_fracture_length=[50, 800]):
        """
        Save multiple fractures with the given parameters.
        """
        for n in range(n_fractures):
            length=random.randint(min_max_fracture_length[0], min_max_fracture_length[1])
            phi_degree = random.uniform(0, self.threshold_degree_phi)
            theta_degree = random.uniform(0, 360)

            nifti_fracture_seg = self.get_nifti_fracture(length=length, phi_degree=phi_degree, theta_degree=theta_degree)
            
            nib.save(nifti_fracture_seg, os.path.join(dir_save_nifti_fracture, "NIHPD_Head_Phantom_fracture_" + str(n + 1) + ".nii.gz"))

    def get_nifti_fracture(self, length, phi_degree, theta_degree):
        """
        Get fractures with the given parameters.
        """
        if self.nifti_skull_seg is None:
            self.nifti_skull_seg = self.mesh_to_voxel_nifti_skull(
                mesh=self.skull_mesh
            )

        self.add_fracture(length, phi_degree, theta_degree)

        nifti_skull_fracture_seg = self.mesh_to_voxel_nifti_skull(
                mesh=self.skull_fracture_mesh
            )
        nifti_fracture_seg = self.get_nifti_fracture_seg(self.nifti_skull_seg, nifti_skull_fracture_seg)

        return nifti_fracture_seg

    def add_fracture(self, length, phi_degree, theta_degree):
        """
        Add fracture by removing meshes with given procedure.
        """
        if self.skull_center is None:
            self.skull_center = self.config["skull_mesh_params"]["center"]

        # Degrees to shift to next point
        direction = pv.spherical_to_cartesian(
            1, np.deg2rad(phi_degree), np.deg2rad(theta_degree)
        )

        delta_shift_degree_phi, delta_shift_degree_theta = (
            self.get_angular_spacing_specific_cell_trace_degree(
                self.skull_center,
                np.add(self.skull_center, np.multiply(direction, 100)),
            )
        )

        # Allow some duplicates for better continuity
        continuity_factor = 0.8
        delta_shift_degree_phi *= continuity_factor
        delta_shift_degree_theta *= continuity_factor

        # Number of iterations to try removing cells to remove (depends on direction, may have duplicates (less number of removals))
        n_iterations = length

        list_start = []
        list_direction = []

        list_switch = [2, 3]

        switch_counter = 1
        list_reset_counter_wait = [10, 20, 30, 40, 50]
        list_reset_counter_wait_index = 0
        pointer = list_switch[random.randint(0, len(list_switch) - 1)]

        for i in range(n_iterations):
            if phi_degree > self.threshold_degree_phi:
                break

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
        
        # Find cells where phi > threshold_degree_phi
        cells_to_remove = np.where(phi_degrees > self.threshold_degree_phi)[0]

        # Remove the identified cells
        mesh.remove_cells(cells_to_remove, inplace=True)

        mesh_sphere, center, radius = self._get_bounding_sphere_mesh(mesh)
        mesh = self.remove_cells_inside_sphere(mesh=mesh, center=center, radius=radius)

        self.skull_mesh = mesh


if __name__ == "__main__":
    path_mesh_brainmask = os.path.join(
        main_directory,
        "src/insilicoICH/annotations/skull/NIHPD_Head_Phantom/assets",
        "mesh_brain.vtk",
    )

    path_mask_brain = os.path.join(
        main_directory, "src/NIHPD_Head_Phantom", "nihpd_asym_04.5-08.5_mask.nii"
    )

    path_file_config = os.path.join(
        main_directory,
        "src/insilicoICH/annotations/skull/NIHPD_Head_Phantom/assets",
        "config.toml",
    )
    
    object_skull_process = SkullProcess(
        path_mesh_brainmask=path_mesh_brainmask, path_mask_brain=path_mask_brain, path_file_config=path_file_config
    )
    object_skull_process.initialize()
    object_skull_process.extract_skull()
    object_skull_process.extract_primary_skull_mesh()

    object_skull_process.save_mesh(
        mesh=object_skull_process.skull_mesh,
        filepath=os.path.join(
            main_directory,
            "src/insilicoICH/annotations/skull/NIHPD_Head_Phantom/assets",
            "skull_mesh.vtk",
        ),
    )
