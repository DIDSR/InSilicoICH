from .phantoms import Phantom
import numpy as np


class DensitometryPhantom(Phantom):
    def __init__(self, matrix_size=400, diameter=200,
                 inserts=[-1000, -80, 0, 40, 50, 60, 250, 1000],
                 patient_name='densitometry phantom', patientid=0):
        large_radius_ratio = 0.85
        diameter_pix = matrix_size*large_radius_ratio
        spacings = [diameter, diameter/diameter_pix, diameter/diameter_pix]
        img = create_circle_phantom(image_size=matrix_size,
                                    large_radius_ratio=large_radius_ratio,
                                    inner_radius_ratio=0.7,
                                    small_radius_ratio=0.1,
                                    num_small_circles=len(inserts),
                                    large_circle_value=0,
                                    small_circle_values=inserts,
                                    bg_value=-1000)
        img = img[None]
        super().__init__(img, spacings, patient_name, patientid, age=0)


class WirePhantom(Phantom):
    def __init__(self, matrix_size=400, diameter=150, wire_HU=5000,
                 wire_radius_ratio=0.01, patient_name='wire phantom', patientid=0):
        large_radius_ratio = 0.85
        diameter_pix = matrix_size*large_radius_ratio
        spacings = [diameter, diameter/diameter_pix, diameter/diameter_pix]
        lrg = create_circle_phantom(image_size=matrix_size,
                                    large_radius_ratio=large_radius_ratio,
                                    large_circle_value=850,
                                    num_small_circles=1,
                                    small_radius_ratio=wire_radius_ratio,
                                    small_circle_values=[wire_HU],
                                    inner_radius_ratio=0)
        sml = create_circle_phantom(image_size=matrix_size,
                                    large_circle_value=850,
                                    large_radius_ratio=large_radius_ratio - 0.05,
                                    num_small_circles=0)
        img = lrg - sml
        img -= 1000
        img = img[None]
        super().__init__(img, spacings, patient_name, patientid, age=0)


class LowContrastDetectabilityPhantom(Phantom):
    '''
    Module 2 frp, ACR CT phantom:
    https://accreditationsupport.acr.org/support/solutions/articles/11000053945-phantom-overview-ct-revised-3-21-2025-
    '''
    def __init__(self, matrix_size=400, bg_value=90, patient_name='low contrast detectability',patientid=0):
        large_radius_ratio = 0.85
        diameter = 200
        diameter_pix = matrix_size*large_radius_ratio
        spacings = [diameter, diameter/diameter_pix, diameter/diameter_pix]
        lrg = create_circle_phantom(image_size=matrix_size,
                                    large_radius_ratio=large_radius_ratio,
                                    num_small_circles=0,
                                    large_circle_value=0,
                                    bg_value=-1000)
        mid = create_circle_phantom(image_size=matrix_size,
                                    large_radius_ratio=large_radius_ratio - 0.05,
                                    num_small_circles=0,
                                    large_circle_value=bg_value,
                                    bg_value=0)
        small_radius_ratio = [0.125] + 4*[0.03] + 4*[0.025] + 4*[0.02] + 4*[0.015] + 4*[0.01]
        num_small_circles = len(small_radius_ratio)
        small_circle_values = num_small_circles*[6]
        sml = create_circle_phantom(image_size=matrix_size,
                                    large_radius_ratio=large_radius_ratio - 0.05,
                                    num_small_circles=num_small_circles,
                                    small_radius_ratio=small_radius_ratio,
                                    inner_radius_ratio=0.7,
                                    small_circle_values=small_circle_values,
                                    large_circle_value=0,
                                    bg_value=0)
        img = (lrg + mid + sml)[::-1]
        img = img.T[::-1]
        img = img[None]
        super().__init__(img, spacings, patient_name, patientid, age=0)


class ACRPhantom(Phantom):
    '''
    Contains Modules Module 1, 2, and 3 from the ACR CT accreditation phantom (Gammex 464):
    Module 1: CT number Accuracy
    Module 2: Low Contrast Detectability
    Module 3: Uniformity and Noise
    https://accreditationsupport.acr.org/support/solutions/articles/11000053945-phantom-overview-ct-revised-3-21-2025-
    '''
    def __init__(self, matrix_size=400, bg_value=90, patient_name='ACR phantom', patientid=0):
        self.matrix_size = matrix_size
        self.bg_value = bg_value
        self.large_radius_ratio = 0.85
        self.diameter = 200
        module_length = 40
        self.air_HU = -1000
        lcd_mod = self.low_contrast_detectability_module()
        accuracy_mod = self.CT_number_accuracy_module()
        uniformity_mod = self.uniformity_module()
        img = np.concat((accuracy_mod, lcd_mod, uniformity_mod))
        diameter_pix = self.matrix_size*self.large_radius_ratio - 0.05
        spacings = [module_length, self.diameter/diameter_pix, self.diameter/diameter_pix]
        super().__init__(img, spacings, patient_name, patientid, age=0)

    def low_contrast_detectability_module(self):
        diameter_pix = self.matrix_size*self.large_radius_ratio - 0.05
        phantom_rad_px = diameter_pix / 2.0

        # Estimate pixel sizes from mm labels (if 200mm phantom = 460px)
        # scale_px_per_mm = 460 / 200 = 2.3 px/mm
        scale_px_per_mm = diameter_pix / self.diameter

        group_radii = [
            6 * scale_px_per_mm / 2.0, # 6mm diameter -> 3mm radius
            5 * scale_px_per_mm / 2.0,
            4 * scale_px_per_mm / 2.0,
            3 * scale_px_per_mm / 2.0,
            2 * scale_px_per_mm / 2.0
        ]
        large_hole_radius = (25 * scale_px_per_mm) / 2.0

        img = create_resolution_phantom(
            image_size=self.matrix_size,
            phantom_diameter_pixels=diameter_pix,
            group_placement_radius_ratio=0.6, # Adjust for desired placement ring
            group_radii_pixels=group_radii[::-1],
            num_circles_per_group=4,
            intra_group_spacing_factor=4, # Adjust spacing between holes in a group
            large_hole_radius_pixels=large_hole_radius,
            large_hole_y_offset_pixels=int(phantom_rad_px * 0.65), # Adjust vertical pos
            bg_value=0,           # Air background
            phantom_body_value=0,   # Example: Solid Water/Plastic HU
            hole_value=6,         # Air holes
            start_angle_offset_deg = -60 # Rotate to roughly match example layout
            )

        lrg = create_circle_phantom(image_size=self.matrix_size,
                                     large_radius_ratio=self.large_radius_ratio,
                                     num_small_circles=0,
                                     large_circle_value=0,
                                     bg_value=self.air_HU)
        mid = create_circle_phantom(image_size=self.matrix_size,
                                    large_radius_ratio=self.large_radius_ratio - 0.05,
                                    num_small_circles=0,
                                    large_circle_value=self.bg_value,
                                    bg_value=0)
        img = img + lrg + mid
        img = img[None]
        return img

    def CT_number_accuracy_module(self):
        values = [-1000, 120, -95, 955]
        img = create_circle_phantom(self.matrix_size,
                                    large_radius_ratio=self.large_radius_ratio,
                                    inner_radius_ratio=0.55,
                                    small_radius_ratio=0.15,
                                    num_small_circles=len(values),
                                    large_circle_value=0,
                                    small_circle_values=values,
                                    bg_value=self.air_HU,
                                    start_angle_offset_deg=45)
        img = img[None]
        return img

    def uniformity_module(self):
        img = create_circle_phantom(self.matrix_size,
                                    large_radius_ratio=self.large_radius_ratio,
                                    num_small_circles=0,
                                    large_circle_value=0,
                                    bg_value=self.air_HU)
        img = img[None]
        return img


def create_circle_phantom(
    image_size=512,
    large_radius_ratio=0.85,  # Ratio of image_size / 2
    inner_radius_ratio=0.55,  # Ratio of large_radius
    small_radius_ratio=0.15,  # Ratio of large_radius
    num_small_circles=6,
    bg_value=0.0,         # Background intensity
    large_circle_value=0.5,  # Main circle intensity
    small_circle_values=None,  # List/array of intensities for small circles
    start_angle_offset_deg=0
):
    """
    Generates a NumPy array representing a circular phantom.

    Creates a large circle containing smaller circles placed equidistantly
    on an inner radius.

    Args:
        image_size (int): The width and height of the output image in pixels.
        large_radius_ratio (float): Radius of the large circle as a fraction
            of half the image size.
        inner_radius_ratio (float): Radius for placing small circle centers,
            as a fraction of the large circle's radius.
        small_radius_ratio (float or list or np.ndarray, optional): Radius of the small circles, as a fraction
            of the large circle's radius. If float, constant radius, else provide list of radii == num_small_circles
        num_small_circles (int): The number of smaller circles inside.
        bg_value (float): The intensity value for the background (0.0=black).
        large_circle_value (float): The intensity value for the large circle.
        small_circle_values (list or np.ndarray, optional):
            A list or array of intensity values for the small circles.
            The length must match num_small_circles. If None, defaults
            to a range of values from 0.1 to 0.9. Values can range
            from 0.0 (black) to 1.0 (white).
        start_angle_offset_deg (float): Rotates the starting position of the first group.

    Returns:
        np.ndarray: A 2D NumPy array representing the generated image.
    """
    # --- Parameter Validation and Setup ---
    if small_circle_values is None:
        # Default values similar to the example (varying gray levels)
        small_circle_values = np.linspace(0.1, 0.9, num_small_circles)
    elif len(small_circle_values) != num_small_circles:
        raise ValueError(f"Length of small_circle_values ({len(small_circle_values)}) "
                         f"must match num_small_circles ({num_small_circles})")
    if isinstance(small_radius_ratio, float):
        small_radius_ratio = num_small_circles*[small_radius_ratio]
    elif len(small_radius_ratio) != num_small_circles:
        raise ValueError(f"Length of small_radius_ratio ({len(small_radius_ratio)}) "
                         f"must match num_small_circles ({num_small_circles})")

    # --- Calculate Dimensions ---
    center_x, center_y = image_size / 2, image_size / 2
    large_radius = large_radius_ratio * (image_size / 2)
    inner_radius = inner_radius_ratio * large_radius

    # --- Create Coordinate Grid ---
    # Create arrays of x and y coordinates for each pixel
    x = np.arange(0, image_size)
    y = np.arange(0, image_size)
    xx, yy = np.meshgrid(x, y)

    # Calculate distance of each pixel from the center
    dist_from_center = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)

    # --- Initialize Image ---
    image = np.full((image_size, image_size), bg_value, dtype=np.float32)

    # --- Draw Large Circle ---
    large_circle_mask = dist_from_center <= large_radius
    image[large_circle_mask] = large_circle_value

    # --- Draw Small Circles ---
    # Calculate angles for placing small circles (in radians)
    angles = np.linspace(0, 2 * np.pi, num_small_circles, endpoint=False) # Use endpoint=False for even spacing
    angles += np.deg2rad(start_angle_offset_deg) # Apply rotation offset

    for i, angle in enumerate(angles):
        # Calculate center coordinates of the current small circle
        small_cx = center_x + inner_radius * np.cos(angle)
        small_cy = center_y + inner_radius * np.sin(angle)

        # Calculate distance of each pixel from the small circle's center
        dist_from_small_center = np.sqrt((xx - small_cx)**2 + (yy - small_cy)**2)

        # Create a mask for the current small circle
        small_radius = small_radius_ratio[i] * large_radius
        small_circle_mask = dist_from_small_center <= small_radius

        # Apply the small circle mask *only within the large circle*
        # (although usually redundant if inner_radius+small_radius < large_radius)
        image[small_circle_mask & large_circle_mask] = small_circle_values[i]

    return image


def create_resolution_phantom(
    image_size=512,
    phantom_diameter_pixels=None, # Diameter of the main phantom body in pixels
    group_placement_radius_ratio=0.65, # Radius for placing group centers (ratio of phantom_radius)
    group_radii_pixels=None,      # List of radii (in pixels) for circles in each group
    num_circles_per_group=4,
    intra_group_spacing_factor=2.5, # Spacing between circle centers within a group (factor of the circle *radius* in that group)
    large_hole_radius_pixels=None,
    large_hole_y_offset_pixels=None, # Vertical offset UP from center for the large hole
    bg_value=0.0,                # Value outside the main phantom body
    phantom_body_value=100.0,    # Value of the main phantom body
    hole_value=-1000.0,          # Value for all the holes (e.g., Air HU)
    start_angle_offset_deg=15    # Angle offset for the first group placement
):
    """
    Generates a NumPy array representing a resolution/low-contrast phantom.

    Creates a large circular phantom body containing groups of smaller circular holes.
    Each group contains multiple holes of the same size, arranged linearly.
    Group sizes vary, and groups are placed angularly. Also includes one larger hole.

    Args:
        image_size (int): The width and height of the output image in pixels.
        phantom_diameter_pixels (int, optional): Diameter of the main phantom body in pixels.
                                                 Defaults to 90% of image_size.
        group_placement_radius_ratio (float): Radius for placing the center of each group,
                                             as a fraction of the phantom body's radius.
        group_radii_pixels (list): List of radii in pixels for the circles in each group.
                                   Order matters for placement (e.g., [12, 10, 8, 6, 4]).
                                   Defaults to a sample list if None.
        num_circles_per_group (int): Number of circles arranged linearly within each group.
        intra_group_spacing_factor (float): Spacing between circle centers within a group,
                                            as a factor of the radius for that group.
                                            E.g., 2.5 means centers are 2.5 * radius apart.
        large_hole_radius_pixels (int, optional): Radius of the single larger hole in pixels.
                                                  Defaults to a sample value if None.
        large_hole_y_offset_pixels (int, optional): Vertical distance UPWARDS from the center
                                                   to place the large hole. Defaults if None.
                                                   Set to 0 for center, negative for down.
        bg_value (float): Intensity value for the background outside the phantom.
        phantom_body_value (float): Intensity value for the main phantom material.
        hole_value (float): Intensity value used to fill all the holes.
        start_angle_offset_deg (float): Rotates the starting position of the first group.

    Returns:
        np.ndarray: A 2D NumPy array representing the generated phantom image.
    """

    # --- Default Values & Parameter Setup ---
    if phantom_diameter_pixels is None:
        phantom_diameter_pixels = int(image_size * 0.9)
    phantom_radius = phantom_diameter_pixels / 2.0

    if group_radii_pixels is None:
        # Example radii roughly scaling like 6mm, 5mm, 4mm, 3mm, 2mm
        # Assuming phantom_diameter_pixels ~ 200mm -> 1mm ~ phantom_radius / 100
        scale = phantom_radius / 100.0
        group_radii_pixels = [6*scale, 5*scale, 4*scale, 3*scale, 2*scale]

    num_groups = len(group_radii_pixels)

    if large_hole_radius_pixels is None:
        scale = phantom_radius / 100.0 # Same scale assumption
        large_hole_radius_pixels = 12.5 * scale # Approx 25mm diameter

    if large_hole_y_offset_pixels is None:
        # Place it somewhat high, like in the example
        large_hole_y_offset_pixels = int(phantom_radius * 0.6)

    # --- Calculate Dimensions ---
    center_x, center_y = image_size / 2.0, image_size / 2.0
    group_placement_radius = group_placement_radius_ratio * phantom_radius

    # --- Create Coordinate Grid ---
    x = np.arange(image_size)
    y = np.arange(image_size)
    xx, yy = np.meshgrid(x, y)

    # Calculate distance of each pixel from the main center
    dist_from_center = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)

    # --- Initialize Image ---
    image = np.full((image_size, image_size), bg_value, dtype=np.float32)

    # --- Draw Phantom Body ---
    body_mask = dist_from_center <= phantom_radius
    image[body_mask] = phantom_body_value

    # --- Helper function to draw a circle ---
    def draw_circle(img, cx, cy, r, value):
        dist_sq = (xx - cx)**2 + (yy - cy)**2
        mask = dist_sq <= r**2
        # Only draw within the main body
        img[mask & body_mask] = value

    # --- Draw the Single Large Hole ---
    large_hole_cx = center_x
    large_hole_cy = center_y - large_hole_y_offset_pixels # Offset upwards from center
    draw_circle(image, large_hole_cx, large_hole_cy, large_hole_radius_pixels, hole_value)

    # --- Draw the Groups of Small Holes ---
    group_angles_rad = np.linspace(0, 2 * np.pi, num_groups, endpoint=False)
    group_angles_rad += np.deg2rad(start_angle_offset_deg) # Apply rotation offset

    for i, group_radius_px in enumerate(group_radii_pixels):
        angle = group_angles_rad[i]

        # Calculate the center point for the *group*
        group_cx = center_x + group_placement_radius * np.cos(angle)
        group_cy = center_y + group_placement_radius * np.sin(angle) # Y increases downwards in image coords

        # Calculate the direction vector for the linear arrangement (tangent to placement circle)
        # Vector from main center to group center: (group_cx - center_x, group_cy - center_y)
        # Tangent vector is perpendicular: -(group_cy - center_y), (group_cx - center_x)
        tx = -(group_cy - center_y)
        ty = (group_cx - center_x)
        # Normalize tangent vector
        norm = np.sqrt(tx**2 + ty**2)
        if norm < 1e-6: # Avoid division by zero if group is at center (shouldn't happen)
            tx, ty = 1.0, 0.0 # Default to horizontal if at center
        else:
            tx /= norm
            ty /= norm

        # Calculate spacing between centers within the group
        spacing = intra_group_spacing_factor * group_radius_px

        # Calculate offsets from the group center along the tangent vector
        # Example for num_circles_per_group = 4: offsets are -1.5, -0.5, +0.5, +1.5
        offsets = np.arange(num_circles_per_group) - (num_circles_per_group - 1) / 2.0

        for offset_factor in offsets:
            # Calculate center of the individual small circle
            small_cx = group_cx + offset_factor * spacing * tx
            small_cy = group_cy + offset_factor * spacing * ty

            # Draw the small circle (hole)
            draw_circle(image, small_cx, small_cy, group_radius_px, hole_value)

    return image
