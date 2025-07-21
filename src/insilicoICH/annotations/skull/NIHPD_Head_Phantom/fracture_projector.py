import numpy as np
import os
import sys
import skimage as ski
from typing import Optional
# import random  # suggest using numpy's rng for reproducibility
import pyvista as pv

main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
sys.path.append(main_directory)


class SkullFractureProjector:
    """
    Complete skull fracture projection system using centroid-based ray casting.
    """

    def __init__(self, skull_mask, fracture_annotations=None, seed: Optional[int] = None):
        self.skull_mask = skull_mask
        self.fracture_annotations = fracture_annotations
        self.projected_fractures = None
        self.threshold_degree_phi = 100
        self.seed = seed if seed is not None else np.random.randint(0, 2**31 - 1)
        self.random = np.random.default_rng(self.seed)

    def morph_closing(self, array: np.ndarray, kernel_size: int):
        array_closed = ski.morphology.closing(array, np.ones(3*[kernel_size]))

        return array_closed

    def centroid_ray_casting(self, skull_mask=None, fracture_annotations=None, centroid=None,
                           intensity_falloff=True, step_size=0.5):
        """
        Project fracture annotations radially through thick skull using centroid-based ray casting.

        Parameters:
        skull_mask: 3D binary array where 1 = skull, 0 = background
        fracture_annotations: 3D binary array where 1 = fracture on inner surface
        intensity_falloff: bool, whether to apply distance-based intensity reduction
        step_size: float, ray marching step size for accuracy

        Returns:
        projected_fractures: 3D array with fractures projected through skull thickness
        """
        if skull_mask is None:
            skull_mask = self.skull_mask
        if fracture_annotations is None:
            fracture_annotations = self.fracture_annotations

        if skull_mask is None or fracture_annotations is None:
            raise ValueError("Skull mask and fracture annotations must be provided")

        if centroid is None:
            # print("Computing skull centroid...")
            # # Find skull centroid using center of mass
            # skull_coords = np.where(skull_mask > 0)
            # centroid = np.array([np.mean(coords) for coords in skull_coords])

            # considering the center of the volume for simplicity (real skull center is close)
            shape_skull_mask = skull_mask.shape
            centroid = [int(shape_skull_mask[0] / 2), int(shape_skull_mask[1] / 2), int(shape_skull_mask[2] / 2)]

        print("Identifying fracture points...")
        # Get fracture points with their intensities
        fracture_coords = np.where(fracture_annotations > 0)
        fracture_points = np.column_stack(fracture_coords)
        # fracture_intensities = fracture_annotations[fracture_coords]

        print(f"Found {len(fracture_points)} fracture points")
        print(f"Skull centroid at: {centroid}")

        # Initialize output with float values for intensity
        projected_fractures = np.zeros_like(skull_mask, dtype=np.float32)

        print("Casting rays through skull...")
        for i, point in enumerate(fracture_points):
            # if i % 100 == 0:
            #     print(f"Processing fracture point {i+1}/{len(fracture_points)}")

            # intensity = fracture_intensities[i]
            ray_points = self._cast_ray_through_skull(centroid, point, skull_mask, step_size)
  
            # Apply intensity with optional distance falloff
            for j, rp in enumerate(ray_points):
                if all(0 <= rp[k] < skull_mask.shape[k] for k in range(3)):
                    # if intensity_falloff:
                    #     # Reduce intensity with distance from inner surface
                    #     falloff_factor = max(0.1, 1.0 - j / max(1, len(ray_points)))
                    #     final_intensity = intensity * falloff_factor
                    # else:
                    #     final_intensity = intensity

                    # current_val = projected_fractures[tuple(rp)]
                    # projected_fractures[tuple(rp)] = max(current_val, final_intensity)

                    projected_fractures[tuple(rp)] = 1

        self.projected_fractures = projected_fractures
        print("Ray casting complete!")
        return projected_fractures
    
    def _cast_ray_through_skull(self, centroid, direction, skull_mask, step_size=0.5):
        """
        Cast ray from centroid through surface point, collecting all skull voxels along the way.
        """
        # # Calculate ray direction
        # direction = surface_point - centroid
        # direction_norm = np.linalg.norm(direction)
        
        # if direction_norm == 0:
        #     return []
            
        # direction = direction / direction_norm

        # Ray parameters
        max_distance = np.linalg.norm(np.array(skull_mask.shape))

        ray_points = []

        # March along ray
        for t in np.arange(0, max_distance, step_size):
            current_point = centroid + np.multiply(t,  direction)
            current_voxel = np.round(current_point).astype(int)

            # Check bounds
            if not all(0 <= current_voxel[i] < skull_mask.shape[i] for i in range(3)):
                break

            # If we hit skull, add to ray points
            if skull_mask[tuple(current_voxel)] > 0:
                ray_points.append(current_voxel)
            # If we've passed through skull and hit background, stop
            elif len(ray_points) > 0:
                break

        return ray_points

    def _polar_to_direction(self, phi_degree, theta_degree):
        """
        Convert polar coordinates to 3D direction vector.
        """
        phi_rad = np.deg2rad(phi_degree)
        theta_rad = np.deg2rad(theta_degree)

        # Spherical to Cartesian conversion
        x = np.sin(theta_rad) * np.cos(phi_rad)
        y = np.sin(theta_rad) * np.sin(phi_rad)
        z = np.cos(theta_rad)

        return np.array([x, y, z])

    def _get_angular_spacing(self):
        """
        Calculate angular spacing for the random walk.
        You'll need to implement this based on your existing get_angular_spacing_specific_cell_trace_degree method.
        """
        # Placeholder - replace with your actual implementation
        # This should return appropriate delta values for phi and theta
        return .5, .5  # Default small angular steps

    def centroid_ray_casting_random_walk(self, skull_mask=None, length=100,
                                         phi_degree=None, theta_degree=None,
                                         step_size=0.5, centroid=None):
        """
        Generate fractures using ray casting with polar random walk directions.

        Parameters:
        skull_mask: 3D binary array where 1 = skull, 0 = background
        length: int, number of iterations for random walk
        initial_phi_degree: float, starting phi angle in degrees
        initial_theta_degree: float, starting theta angle in degrees
        step_size: float, ray marching step size for accuracy
        centroid: tuple/list, skull centroid coordinates

        Returns:
        projected_fractures: 3D array with fractures projected through skull thickness
        """
        phi_degree = phi_degree or self.random.uniform(0, 60)
        theta_degree = theta_degree or self.random.uniform(0, 360)

        if skull_mask is None:
            skull_mask = self.skull_mask

        if skull_mask is None:
            raise ValueError("Skull mask must be provided")

        if centroid is None:
            # Use center of the volume for simplicity
            shape_skull_mask = skull_mask.shape
            centroid = np.array([int(shape_skull_mask[0] / 2), 
                            int(shape_skull_mask[1] / 2), 
                            shape_skull_mask[2] - 82])     # Note: temprary fix

        print(f"Starting random walk fracture generation with {length} iterations")
        print(f"Skull centroid at: {centroid}")

        # Initialize output
        projected_fractures = np.zeros_like(skull_mask, dtype=np.float32)

        # Random walk control parameters (from original code)
        list_switch = [2, 3]
        switch_counter = 1
        list_reset_counter_wait = [10, 20, 30, 40, 50]
        list_reset_counter_wait_index = 0
        pointer = list_switch[self.random.integers(0, len(list_switch) - 1)]

        # Calculate angular spacing (you'll need to implement this based on your existing method)
        delta_shift_degree_phi, delta_shift_degree_theta = self._get_angular_spacing()

        # Allow some duplicates for better continuity
        continuity_factor = 0.8
        delta_shift_degree_phi *= continuity_factor
        delta_shift_degree_theta *= continuity_factor

        print("Casting rays with random walk...")

        for i in range(length):
            if hasattr(self, 'threshold_degree_phi') and phi_degree > self.threshold_degree_phi:
                break

            # Note: temporary fix with 180 - phi_degree    
            # Convert polar coordinates to direction vector
            direction = pv.spherical_to_cartesian(
                1, np.deg2rad(180 - phi_degree), np.deg2rad(theta_degree)
            )

            # # Calculate surface point by extending from centroid
            # max_radius = 100 #np.linalg.norm(np.array(skull_mask.shape))
            # surface_point = centroid + direction * max_radius

            # Cast ray and get voxels to mark as fracture
            ray_points = self._cast_ray_through_skull(centroid, direction, skull_mask, step_size)

            # Mark ray points as fracture
            for rp in ray_points:
                if all(0 <= rp[k] < skull_mask.shape[k] for k in range(3)):
                    projected_fractures[tuple(rp)] = 1

            # Random walk logic (from original code)
            if (switch_counter % list_reset_counter_wait[list_reset_counter_wait_index] == 0):
                list_reset_counter_wait_index = self.random.integers(0, len(list_reset_counter_wait) - 1)
                switch_counter = 1
                pointer = list_switch[self.random.integers(0, len(list_switch) - 1)]

            switch_counter += 1

            # Update angles based on pointer
            if pointer == 0:
                phi_degree += delta_shift_degree_phi * self.random.choice([-1, 1])
            elif pointer == 1:
                theta_degree += delta_shift_degree_theta * self.random.choice([-1, 1])
            elif pointer == 2:
                phi_degree += delta_shift_degree_phi * self.random.choice([-1, 1])
                theta_degree += delta_shift_degree_theta
            elif pointer == 3:
                phi_degree += delta_shift_degree_phi
                theta_degree += delta_shift_degree_theta * self.random.choice([-1, 1])
            elif pointer == 4:
                phi_degree += delta_shift_degree_phi * self.random.choice([-1, 1])
                theta_degree += delta_shift_degree_theta * self.random.choice([-1, 1])

        self.projected_fractures = projected_fractures
        print("Random walk ray casting complete!")

        return projected_fractures

    def analyze_projection_results(self):
        """
        Analyze and compare original vs projected fractures.
        """
        if self.projected_fractures is None:
            print("No projection results to analyze. Run ray casting first.")
            return

        print("\n=== Projection Analysis ===")

        # Basic statistics
        original_fractures = np.sum(self.fracture_annotations > 0)
        projected_fractures = np.sum(self.projected_fractures > 0)
        skull_volume = np.sum(self.skull_mask > 0)

        print(f"Original fracture voxels: {original_fractures}")
        print(f"Projected fracture voxels: {projected_fractures}")
        print(f"Skull volume: {skull_volume}")
        print(f"Expansion factor: {projected_fractures / max(1, original_fractures):.2f}x")
        print(f"Skull coverage: {projected_fractures / skull_volume * 100:.2f}%")

        # Intensity analysis
        intensities = self.projected_fractures[self.projected_fractures > 0]
        if len(intensities) > 0:
            print(f"\nIntensity Statistics:")
            print(f"  Range: {intensities.min():.3f} - {intensities.max():.3f}")
            print(f"  Mean: {intensities.mean():.3f}")
            print(f"  Std: {intensities.std():.3f}")

        # Fracture region analysis
        unique_fractures = np.unique(self.fracture_annotations[self.fracture_annotations > 0])
        print(f"\nFracture Regions: {len(unique_fractures)}")

        for frac_id in unique_fractures:
            original_size = np.sum(self.fracture_annotations == frac_id)
            # Find corresponding projected voxels (approximate)
            projected_size = np.sum(self.projected_fractures > 0)  # This is simplified
            print(f"  Fracture {frac_id}: {original_size} -> ~{projected_size//len(unique_fractures)} voxels")
