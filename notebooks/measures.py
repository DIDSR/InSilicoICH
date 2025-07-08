import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.measure import label
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import marching_cubes, mesh_surface_area

from data import load_hssayeni_image_mask_pair


def calculate_glcm_metrics_3d(volume, mask, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32):
    """
    Calculates 3D Gray-Level Co-occurrence Matrix (GLCM) metrics for a
    region of interest (ROI) by averaging metrics from 2D slices.

    Args:
        volume (np.ndarray): A 3D NumPy array representing the CT voxel volume.
        mask (np.ndarray): A 3D binary NumPy array of the same shape as `volume`,
                           with `True` or `1` indicating the ROI.
        distances (list of int): List of pixel pair distances for GLCM calculation.
        angles (list of float): List of angles in radians for GLCM calculation.
        levels (int): The number of gray levels to use for quantizing the
                      intensity values within the ROI.

    Returns:
        dict: A dictionary containing the averaged GLCM properties (contrast,
              correlation, energy, homogeneity) across all valid slices in the ROI.
              Returns None if the mask is empty.
    """
    # Ensure the mask is boolean for indexing
    if mask.dtype != bool:
        mask = mask.astype(bool)

    # Check if the volume and mask have the same shape
    if volume.shape != mask.shape:
        raise ValueError("Volume and mask must have the same shape.")

    # Find the voxels within the mask
    roi_voxels = volume[mask]

    if roi_voxels.size == 0:
        print("Warning: Mask is empty. No GLCM metrics can be calculated.")
        return None

    # --- Intensity Quantization ---
    # Scale the intensity values within the ROI to the specified number of levels.
    # This is a crucial step for GLCM.
    vmin = roi_voxels.min()
    vmax = roi_voxels.max()
    scaled_volume = np.zeros_like(volume, dtype=np.uint8)

    # Apply scaling only to the ROI voxels
    if vmax > vmin:
        scaled_voxels = ( (roi_voxels - vmin) / (vmax - vmin) * (levels - 1) ).astype(np.uint8)
    else: # Handle case where all voxels in ROI have the same value
        scaled_voxels = np.zeros_like(roi_voxels, dtype=np.uint8)

    scaled_volume[mask] = scaled_voxels

    # --- Slice-by-Slice GLCM Calculation ---
    # We process along the z-axis (axial slices)
    glcm_properties = {
        'contrast': [],
        'correlation': [],
        'energy': [],
        'homogeneity': [],
        'dissimilarity': []
    }

    for z in range(volume.shape[0]):
        slice_img = scaled_volume[z, :, :]
        slice_mask = mask[z, :, :]

        # Skip slices that do not contain any part of the ROI
        if not np.any(slice_mask):
            continue

        # --- Bounding Box for Efficiency ---
        # To avoid processing the entire slice, we find the bounding box of the mask on the slice.
        rows, cols = np.where(slice_mask)
        r_min, r_max = rows.min(), rows.max()
        c_min, c_max = cols.min(), cols.max()

        # Crop the image to the bounding box of the mask on that slice
        cropped_slice = slice_img[r_min:r_max+1, c_min:c_max+1]

        # Calculate GLCM on the cropped slice.
        # Note: skimage's GLCM does not support masks directly. By using a tight
        # bounding box, we minimize the influence of non-ROI pixels.
        glcm = graycomatrix(cropped_slice,
                            distances=distances,
                            angles=angles,
                            levels=levels,
                            symmetric=True,
                            normed=True)

        # Calculate properties for the current slice and average over distances/angles
        glcm_properties['contrast'].append(np.mean(graycoprops(glcm, 'contrast')))
        glcm_properties['correlation'].append(np.mean(graycoprops(glcm, 'correlation')))
        glcm_properties['energy'].append(np.mean(graycoprops(glcm, 'energy')))
        glcm_properties['homogeneity'].append(np.mean(graycoprops(glcm, 'homogeneity')))
        glcm_properties['dissimilarity'].append(np.mean(graycoprops(glcm, 'dissimilarity')))

    # Average the properties across all valid slices
    if not glcm_properties['contrast']: # Check if any slice was processed
        return None

    averaged_metrics = {prop: np.mean(values) for prop, values in glcm_properties.items()}

    return averaged_metrics


def calculate_sphericity(binary_mask):
    """
    Calculates the sphericity of a 3D object from its binary mask.

    Sphericity is a measure of how closely the shape of an object resembles a
    perfect sphere. It is defined as the ratio of the surface area of a sphere
    with the same volume as the given object to the surface area of the object.

    A perfect sphere has a sphericity of 1.

    Args:
        binary_mask (np.ndarray): A 3D NumPy array representing the binary
                                  mask of the object (1s for the object,
                                  0s for the background).

    Returns:
        float: The sphericity of the object. Returns 0 if the volume is 0.
    """
    # Ensure the input is a binary mask
    if not np.all(np.unique(binary_mask) <= [0, 1]):
        raise ValueError("Input array must be a binary mask (containing only 0s and 1s).")

    # Calculate the volume of the object by counting the number of non-zero voxels.
    volume = np.sum(binary_mask)

    if volume == 0:
        return 0.0

    # Use the marching cubes algorithm to find the surface of the object.
    # The `spacing` argument is important for accurate surface area calculation
    # if the voxels are not isotropic. Here we assume isotropic voxels (1,1,1).
    verts, faces, _, _ = marching_cubes(binary_mask, level=0.5, spacing=(1.0, 1.0, 1.0))

    # Calculate the surface area of the mesh.
    surface_area = mesh_surface_area(verts, faces)

    # Calculate the sphericity using the formula:
    # Sphericity = (pi^(1/3) * (6 * volume)^(2/3)) / surface_area
    sphericity = (np.pi**(1/3) * (6 * volume)**(2/3)) / surface_area

    return sphericity


def calculate_compactness(binary_mask):
    """
    Calculates the compactness of a 3D object from its binary mask.

    Compactness is another measure of how spherical an object is. One common
    dimensionless definition is related to sphericity. This function uses the
    definition from the Image Biomarker Standardization Initiative (IBSI),
    where compactness is defined as (36 * pi * volume^2) / surface_area^3.

    This value is equivalent to sphericity cubed (sphericity^3).
    A perfect sphere has a compactness of 1.

    Args:
        binary_mask (np.ndarray): A 3D NumPy array representing the binary
                                  mask of the object (1s for the object,
                                  0s for the background).

    Returns:
        float: The compactness of the object. Returns 0 if the volume is 0.
    """
    # Ensure the input is a binary mask
    if not np.all(np.unique(binary_mask) <= [0, 1]):
        raise ValueError("Input array must be a binary mask (containing only 0s and 1s).")

    # Calculate the volume of the object by counting the number of non-zero voxels.
    volume = np.sum(binary_mask)

    if volume == 0:
        return 0.0

    # Use the marching cubes algorithm to find the surface of the object.
    # The `spacing` argument is important for accurate surface area calculation
    # if the voxels are not isotropic. Here we assume isotropic voxels (1,1,1).
    verts, faces, _, _ = marching_cubes(binary_mask, level=0.5, spacing=(1.0, 1.0, 1.0))

    # Calculate the surface area of the mesh.
    surface_area = mesh_surface_area(verts, faces)

    # Avoid division by zero if surface area is somehow zero for a non-zero volume
    if surface_area == 0:
        return 0.0

    # Calculate compactness using the formula:
    # Compactness = (36 * pi * volume^2) / surface_area^3
    compactness = (36 * np.pi * volume**2) / (surface_area**3)

    return compactness


def calculate_ICH_features_Hssayeni(ct_ich_dataset_path, return_images=False):
    patients = pd.read_csv(ct_ich_dataset_path / 'Patient_demographics.csv', index_col=0)
    lesions = pd.read_csv(ct_ich_dataset_path / 'hemorrhage_diagnosis_raw_ct.csv')

    contrast = []
    correlation = []
    patient_numbers = []
    z_slices = []
    images = []
    masks = []
    volumes = []
    attenuation = []
    sphericity = []
    subtypes = []

    target_spacing = (1, 1, 1)
    available_lesions = ['Intraparenchymal', 'Epidural', 'Subdural', 'Intraventricular', 'Subarachnoid']

    for id in tqdm(patients.index[~patients.index.isna()].astype(int)):
        volume, mask, resample_factor = load_hssayeni_image_mask_pair(ct_ich_dataset_path, id, target_spacing=target_spacing, return_resample_factor=True)
        if volume is None:
            continue
        if mask.sum() == 0:
            continue
        mask = label(mask)
        lesion_values = np.unique(mask)[1:]
        for lesion_value in lesion_values:
            lesion_mask = np.zeros_like(mask).astype(bool)
            lesion_mask[(mask == lesion_value) & (volume > 0) & (volume <= 200)] = True # includes resegmentation described by https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1225342/full#supplementary-material
            volume_voxel = lesion_mask.sum()
            volume_mL = volume_voxel*target_spacing[0]**3/1000
            volumes.append(volume_mL)
            attenuation.append(volume[lesion_mask].mean())

            glcm_metrics = calculate_glcm_metrics_3d(volume, lesion_mask, distances=[3, 5, 7], levels=32)
            contrast.append(glcm_metrics['contrast'])
            correlation.append(glcm_metrics['correlation'])
            patient_numbers.append(id)
            z = int(lesion_mask.sum(axis=(1, 2)).argmax() / resample_factor[2])
            z_slices.append(z)
            row = lesions[(lesions['PatientNumber'] == id) & (lesions['SliceNumber'] == z)]
            lesion = ''
            for candidate in available_lesions:
                if row[candidate].item():
                    lesion = lesion + candidate + ' '
            subtypes.append(lesion)

            images.append(volume[z])
            masks.append(mask[z])
            sphericity.append(calculate_sphericity(lesion_mask))

    hssayeni = pd.DataFrame({'ID': patient_numbers, 'subtype': subtypes, 'volume': volumes, 'attenuation': attenuation, 'glcm contrast': contrast, 'glcm correlation': correlation, 'sphericity': sphericity, 'z location': z_slices})
    hssayeni['dataset'] = 'hssayeni'
    if return_images:
        hssayeni, images, masks
    return hssayeni
