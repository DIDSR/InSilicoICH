from pathlib import Path

import numpy as np
import pandas as pd
from skimage.measure import label
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import marching_cubes, mesh_surface_area
from tqdm.auto import tqdm
from typing import List, Tuple, Optional
import nibabel as nib
from scipy.ndimage import zoom


def load_hssayeni_image_mask_pair(
    ct_ich_dataset_path: Path,
    patient_id: int,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Loads and resamples a CT image and its mask to a uniform isotropic spacing.
    (This is an improved version of the function from the previous turn)
    """
    img_file = ct_ich_dataset_path / 'ct_scans' / f'{int(patient_id):03d}.nii'
    mask_file = ct_ich_dataset_path / 'masks' / f'{int(patient_id):03d}.nii'

    if not (img_file.exists() and mask_file.exists()):
        return None, None, None

    img_nii = nib.load(img_file)
    mask_nii = nib.load(mask_file)

    original_spacing = img_nii.header.get_zooms()[:3]
    resample_factor = [orig / new for orig, new in zip(original_spacing, target_spacing)]

    img_data = zoom(img_nii.get_fdata(), resample_factor, order=3, prefilter=True)
    mask_data = zoom(mask_nii.get_fdata(), resample_factor, order=0, prefilter=False)

    img_vol = np.rot90(img_data, 1).transpose(2, 0, 1)
    mask = np.rot90(mask_data, 1).transpose(2, 0, 1)

    return img_vol, mask, np.array(resample_factor)


def calculate_glcm_metrics_3d(volume, mask, distances=[1],
                              angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32):
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


def calculate_ich_features_hssayeni(
    ct_ich_dataset_path: str,
    return_images: bool = False
) -> pd.DataFrame | Tuple[pd.DataFrame, List[np.ndarray], List[np.ndarray]]:
    """
    Calculates 3D shape and texture features for each hemorrhage in the Hssayeni dataset.

    This function iterates through each patient, loads the resampled CT image and mask,
    identifies individual hemorrhage lesions, and calculates a set of radiomic features
    for each one.

    Args:
        ct_ich_dataset_path (str): Path to the root of the dataset directory.
        return_images (bool): If True, also returns lists of the 2D image and mask
                              slices corresponding to each lesion.

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to a single hemorrhage
                      lesion and columns correspond to the calculated features.
        or
        Tuple[pd.DataFrame, List, List]: DataFrame and lists of image/mask slices
                                         if return_images is True.
    """
    dataset_path = Path(ct_ich_dataset_path)
    patients_df = pd.read_csv(dataset_path / 'Patient_demographics.csv', index_col=0)
    lesions_df = pd.read_csv(dataset_path / 'hemorrhage_diagnosis_raw_ct.csv')

    all_lesion_features = []
    image_slices, mask_slices = [], []

    # Define radiomics parameters
    target_spacing = (1.0, 1.0, 1.0)
    glcm_distances = [3, 5, 7]
    glcm_levels = 32
    # HU windowing for resegmentation, as per referenced paper
    # This isolates voxels more likely to be acute blood
    hu_min, hu_max = 0, 200

    available_lesions = ['Intraparenchymal', 'Epidural', 'Subdural', 'Intraventricular', 'Subarachnoid']

    patient_ids = patients_df.index[~patients_df.index.isna()].astype(int)
    for patient_id in tqdm(patient_ids, desc="Processing Patients"):

        image_vol, mask_vol, resample_factor = load_hssayeni_image_mask_pair(dataset_path, patient_id, target_spacing)

        if image_vol is None or mask_vol.sum() == 0:
            continue

        labeled_mask, num_lesions = label(mask_vol, return_num=True)

        for lesion_idx in range(1, num_lesions + 1):
            # --- Per-Lesion Processing ---

            # Create a binary mask for the current lesion with HU windowing
            lesion_mask = (labeled_mask == lesion_idx) & (image_vol > hu_min) & (image_vol < hu_max)

            volume_voxel = lesion_mask.sum()
            if volume_voxel == 0:
                continue

            # 1. Calculate Basic Features
            volume_ml = (volume_voxel * np.prod(target_spacing)) / 1000.0
            mean_attenuation = image_vol[lesion_mask].mean()

            # 2. Calculate Shape & Texture Features
            sphericity_val = calculate_sphericity(lesion_mask)
            glcm_metrics = calculate_glcm_metrics_3d(image_vol, lesion_mask, distances=glcm_distances, levels=glcm_levels)

            # 3. Determine Lesion Subtype (Robustly)
            # Find the original z-slice with the largest cross-section for this lesion
            z_slice_resampled = lesion_mask.sum(axis=(1, 2)).argmax()
            z_slice_original = int(round(z_slice_resampled / resample_factor[2]))

            lesion_type_rows = lesions_df[(lesions_df['PatientNumber'] == patient_id) & (lesions_df['SliceNumber'] == z_slice_original)]

            subtype_str = ''
            if not lesion_type_rows.empty:
                # Concatenate all true subtypes for that slice
                subtype_str = ' '.join([
                    lesion for lesion in available_lesions if lesion_type_rows.iloc[0][lesion]
                ]).strip()

            # 4. Store Results for this Lesion
            lesion_data = {
                'ID': patient_id,
                'lesion_index': lesion_idx,
                'subtype': subtype_str if subtype_str else 'Unknown',
                'volume_ml': volume_ml,
                'mean_attenuation': mean_attenuation,
                'sphericity': sphericity_val,
                'glcm_contrast': glcm_metrics['contrast'],
                'glcm_correlation': glcm_metrics['correlation'],
                'z_location_resampled': z_slice_resampled,
                'dataset': 'hssayeni'
            }
            all_lesion_features.append(lesion_data)

            if return_images:
                image_slices.append(image_vol[z_slice_resampled])
                mask_slices.append(labeled_mask[z_slice_resampled])

    # Convert list of dictionaries to DataFrame (more efficient)
    features_df = pd.DataFrame(all_lesion_features)
    features_df['ID'] = pd.Categorical(features_df['ID'])
    features_df['subtype'] = pd.Categorical(features_df['subtype'])
    features_df['lesion_index'] = pd.Categorical(features_df['lesion_index'])
    if return_images:
        return features_df, image_slices, mask_slices

    return features_df


def load_bhsd_image_mask_pair(
    bhsd_path: str | Path,
    patient_id: str,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Loads and resamples a CT image and its mask to a uniform isotropic spacing.
    (This is an improved version of the function from the previous turn)
    """
    bhsd_path = Path(bhsd_path)
    img_file = bhsd_path / 'images' / patient_id
    mask_file = bhsd_path / 'ground_truths' / patient_id

    if not (img_file.exists() and mask_file.exists()):
        return None, None, None

    img_nii = nib.load(img_file)
    mask_nii = nib.load(mask_file)

    original_spacing = img_nii.header.get_zooms()[:3]
    resample_factor = [orig / new for orig, new in zip(original_spacing, target_spacing)]

    img_data = zoom(img_nii.get_fdata(), resample_factor, order=3, prefilter=True)
    mask_data = zoom(mask_nii.get_fdata(), resample_factor, order=0, prefilter=False)

    img_vol = np.rot90(img_data, 1).transpose(2, 0, 1)
    mask = np.rot90(mask_data, 1).transpose(2, 0, 1)

    return img_vol, mask.astype(int), np.array(resample_factor)


def calculate_ich_features_bhsd(
    bhsd_dataset_path: str,
    return_images: bool = False
) -> pd.DataFrame | Tuple[pd.DataFrame, List[np.ndarray], List[np.ndarray]]:
    """
    Calculates 3D shape and texture features for each hemorrhage in the BHSD dataset.

    This function iterates through each patient, loads the resampled CT image and mask,
    identifies individual hemorrhage lesions, and calculates a set of radiomic features
    for each one.

    Args:
        bhsd_dataset_path (str): Path to the root of the dataset directory.
        return_images (bool): If True, also returns lists of the 2D image and mask
                              slices corresponding to each lesion.

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to a single hemorrhage
                      lesion and columns correspond to the calculated features.
        or
        Tuple[pd.DataFrame, List, List]: DataFrame and lists of image/mask slices
                                         if return_images is True.
    """
    dataset_path = Path(bhsd_dataset_path)
    img_path = dataset_path / 'images'

    all_lesion_features = []
    image_slices, mask_slices = [], []

    # Define radiomics parameters
    target_spacing = (1.0, 1.0, 1.0)
    glcm_distances = [3, 5, 7]
    glcm_levels = 32
    # HU windowing for resegmentation, as per referenced paper
    # This isolates voxels more likely to be acute blood
    hu_min, hu_max = 0, 200

    available_lesions = {2: 'Intraparenchymal', 1: 'Epidural', 5: 'Subdural', 3: 'Intraventricular', 4: 'Subarachnoid'}

    patient_ids = list(map(lambda o: o.name,  img_path.glob('*.nii.gz')))
    for patient_id in tqdm(patient_ids, desc="Processing Patients"):

        image_vol, mask_vol, resample_factor = load_bhsd_image_mask_pair(dataset_path, patient_id, target_spacing)

        if image_vol is None or mask_vol.sum() == 0:
            continue

        labeled_mask, num_lesions = label(mask_vol, return_num=True)

        for lesion_idx in range(1, num_lesions + 1):
            # --- Per-Lesion Processing ---

            # Create a binary mask for the current lesion with HU windowing
            lesion_mask = (labeled_mask == lesion_idx) & (image_vol > hu_min) & (image_vol < hu_max)
            volume_voxel = lesion_mask.sum()
            if volume_voxel == 0:
                continue

            subtype_str = available_lesions[mask_vol[lesion_mask].max()]

            # 1. Calculate Basic Features
            volume_ml = (volume_voxel * np.prod(target_spacing)) / 1000.0
            mean_attenuation = image_vol[lesion_mask].mean()

            # 2. Calculate Shape & Texture Features
            sphericity_val = calculate_sphericity(lesion_mask)
            glcm_metrics = calculate_glcm_metrics_3d(image_vol, lesion_mask, distances=glcm_distances, levels=glcm_levels)

            # 3. Determine Lesion Subtype (Robustly)
            # Find the original z-slice with the largest cross-section for this lesion
            z_slice_resampled = lesion_mask.sum(axis=(1, 2)).argmax()
            z_slice_original = int(round(z_slice_resampled / resample_factor[2]))


            # 4. Store Results for this Lesion
            lesion_data = {
                'ID': patient_id,
                'lesion_index': lesion_idx,
                'subtype': subtype_str if subtype_str else 'Unknown',
                'volume_ml': volume_ml,
                'mean_attenuation': mean_attenuation,
                'sphericity': sphericity_val,
                'glcm_contrast': glcm_metrics['contrast'],
                'glcm_correlation': glcm_metrics['correlation'],
                'z_location_resampled': z_slice_resampled,
                'dataset': 'hssayeni'
            }
            all_lesion_features.append(lesion_data)

            if return_images:
                image_slices.append(image_vol[z_slice_resampled])
                mask_slices.append(labeled_mask[z_slice_resampled])

    # Convert list of dictionaries to DataFrame (more efficient)
    features_df = pd.DataFrame(all_lesion_features)
    features_df['ID'] = pd.Categorical(features_df['ID'])
    features_df['subtype'] = pd.Categorical(features_df['subtype'])
    features_df['lesion_index'] = pd.Categorical(features_df['lesion_index'])
    if return_images:
        return features_df, image_slices, mask_slices

    return features_df