import abc
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, marching_cubes, mesh_surface_area
from tqdm.auto import tqdm
from insilicoICH.phantoms.head_phantoms import HeadPhantom

from plot import create_ich_lesion_montage

# =============================================================================
# 1. FEATURE CALCULATION MODULE
# This class encapsulates all the logic for calculating radiomic features.
# It is completely independent of any specific dataset.
# =============================================================================


class RadiomicsFeatureExtractor:
    """Calculates a set of radiomic features for a given image and mask."""

    def __init__(self, hu_min: int = 0, hu_max: int = 200, glcm_levels: int = 32):
        """
        Initializes the feature extractor with common parameters.
        Args:
            hu_min: Minimum Hounsfield Unit for intensity masking.
            hu_max: Maximum Hounsfield Unit for intensity masking.
            glcm_levels: Number of gray levels for GLCM quantization.
        """
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.glcm_levels = glcm_levels

    def extract_features_for_lesion(self, image_vol: np.ndarray, full_mask: np.ndarray) -> Dict[str, Any]:
        """
        Extracts all features for a single binary lesion mask.
        Args:
            image_vol: The full 3D CT image volume.
            full_mask: A 3D binary mask of the single lesion.
        Returns:
            A dictionary containing all calculated feature names and values.
        """
        # Apply HU windowing to the mask
        lesion_mask = (full_mask) & (image_vol > self.hu_min) & (image_vol < self.hu_max)
        
        volume_voxel = lesion_mask.sum()
        if volume_voxel == 0:
            return {}

        # --- Calculate all features ---
        shape_features = self._calculate_shape_features(lesion_mask)
        texture_features = self._calculate_glcm_metrics_3d(image_vol, lesion_mask)
        first_order_features = self._calculate_first_order_features(image_vol, lesion_mask)

        # Combine all feature dictionaries
        all_features = {**shape_features, **texture_features, **first_order_features}
        all_features['volume_voxel'] = volume_voxel
        
        return all_features

    def _calculate_first_order_features(self, image_vol: np.ndarray, lesion_mask: np.ndarray) -> Dict[str, float]:
        """Calculates first-order statistics (intensity-based)."""
        roi_voxels = image_vol[lesion_mask]
        return {
            'mean_attenuation': roi_voxels.mean(),
            'std_attenuation': roi_voxels.std(),
        }

    def _calculate_shape_features(self, lesion_mask: np.ndarray) -> Dict[str, float]:
        """Calculates 3D shape features."""
        volume_voxel = lesion_mask.sum()
        if volume_voxel == 0:
            return {'sphericity': 0, 'compactness': 0}

        try:
            verts, faces, _, _ = marching_cubes(lesion_mask, level=0.5, spacing=(1.0, 1.0, 1.0))
            surface_area = mesh_surface_area(verts, faces)
            if surface_area == 0:
                return {'sphericity': 0, 'compactness': 0}
        except (RuntimeError, ValueError):
            # Marching cubes can fail on very small/thin objects
            return {'sphericity': 0, 'compactness': 0}

        sphericity = (np.pi**(1/3) * (6 * volume_voxel)**(2/3)) / surface_area
        compactness = (36 * np.pi * volume_voxel**2) / (surface_area**3)

        return {'sphericity': sphericity, 'compactness': compactness}

    def _calculate_glcm_metrics_3d(self, volume: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Calculates 3D GLCM metrics by averaging across 2D slices."""
        roi_voxels = volume[mask]
        if roi_voxels.size == 0:
            return {}

        vmin, vmax = roi_voxels.min(), roi_voxels.max()
        scaled_volume = np.zeros_like(volume, dtype=np.uint8)
        if vmax > vmin:
            scaled_voxels = ((roi_voxels - vmin) / (vmax - vmin) * (self.glcm_levels - 1)).astype(np.uint8)
        else:
            scaled_voxels = np.zeros_like(roi_voxels, dtype=np.uint8)
        scaled_volume[mask] = scaled_voxels

        glcm_props = {'contrast': [], 'correlation': [], 'energy': [], 'homogeneity': []}
        for z in range(volume.shape[0]):
            slice_img = scaled_volume[z, :, :]
            slice_mask = mask[z, :, :]
            if not np.any(slice_mask):
                continue

            rows, cols = np.where(slice_mask)
            cropped_slice = slice_img[rows.min():rows.max()+1, cols.min():cols.max()+1]
            if cropped_slice.size == 0:
                continue

            glcm = graycomatrix(
                cropped_slice, distances=[3, 5, 7], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                levels=self.glcm_levels, symmetric=True, normed=True
            )
            for prop in glcm_props.keys():
                glcm_props[prop].append(np.mean(graycoprops(glcm, prop)))
        
        return {prop: np.mean(values) if values else 0 for prop, values in glcm_props.items()}


# =============================================================================
# 2. DATA LOADING MODULE
# This section defines a standard interface for loading data from any dataset.
# To add a new dataset, you only need to create a new class here.
# =============================================================================

class DatasetLoader(abc.ABC):
    """Abstract Base Class for dataset loaders."""
    def __init__(self, dataset_path: str, target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        self.dataset_path = Path(dataset_path)
        self.target_spacing = target_spacing

    @abc.abstractmethod
    def get_patient_ids(self) -> List[Any]:
        """Returns a list of all patient identifiers in the dataset."""
        pass

    @abc.abstractmethod
    def load_patient_data(self, patient_id: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Loads image, mask, and resample factor for a single patient."""
        pass

    @abc.abstractmethod
    def get_lesion_subtype(self, lesion_info: Dict) -> str:
        """Determines the subtype of a lesion based on dataset-specific info."""
        pass


class HssayeniLoader(DatasetLoader):
    """Loads data from the Hssayeni (CT-ICH) dataset.

    Hssayeni, M. (2020). Computed Tomography Images for Intracranial
    Hemorrhage Detection and Segmentation (version 1.3.1). PhysioNet.
    RRID:SCR_007345. https://doi.org/10.13026/4nae-zg36

    https://physionet.org/content/ct-ich/1.3.1/
    """
    def __init__(self, dataset_path: str, **kwargs):
        super().__init__(dataset_path, **kwargs)
        self.patients_df = pd.read_csv(self.dataset_path / 'Patient_demographics.csv', index_col=0)
        self.lesions_df = pd.read_csv(self.dataset_path / 'hemorrhage_diagnosis_raw_ct.csv')
        self.available_lesions = ['Intraparenchymal', 'Epidural', 'Subdural', 'Intraventricular', 'Subarachnoid']
        self.subtype_map = {1: 'Epidural', 2: 'Intraparenchymal', 3: 'Intraventricular', 4: 'Subarachnoid', 5: 'Subdural'}

    def get_patient_ids(self) -> List[int]:
        return self.patients_df.index[~self.patients_df.index.isna()].astype(int).tolist()

    def load_patient_data(self, patient_id: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        '''note: `subtype_masks` are binary masks for each subtype, not the
        full mask and were processed and downloaded from https://github.com/DIDSR/framework-for-synthetic-ICH-to-assess-AI-generalizability/tree/main/datasets/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.3.1/subtype_masks
        '''
        img_file = self.dataset_path / 'ct_scans' / f'{int(patient_id):03d}.nii'
        mask_file = self.dataset_path / 'subtype_masks' / f'{int(patient_id):03d}_1.nii'
        if not (img_file.exists() and mask_file.exists()):
            return None, None, None

        img_nii = nib.load(img_file)
        mask_nii = nib.load(mask_file)

        original_spacing = img_nii.header.get_zooms()[:3]
        resample_factor = [orig / new for orig, new in zip(original_spacing, self.target_spacing)]

        img_data = zoom(img_nii.get_fdata(), resample_factor, order=3, prefilter=True)
        mask_data = zoom(mask_nii.get_fdata(), resample_factor, order=0, prefilter=False)

        img_vol = np.rot90(img_data, 1).transpose(2, 0, 1)
        mask_vol = np.rot90(mask_data, 1).transpose(2, 0, 1)

        return img_vol, mask_vol, np.array(resample_factor)

    def get_lesion_subtype(self, lesion_info: Dict) -> str:
        lesion_mask = lesion_info['lesion_mask']
        original_mask = lesion_info['original_mask']

        # Get the integer label from the original (pre-windowing) mask
        lesion_label_val = original_mask[lesion_mask].max()
        return self.subtype_map.get(lesion_label_val, 'Unknown')

    def _get_lesion_subtype(self, lesion_info: Dict) -> str:
        """Returns the lesion subtype as a string for a given lesion mask.
    this works for the original Hssayeni dataset but not the subtype_masks
    (see docstring in load_patient_data)"""
        patient_id = lesion_info['patient_id']
        z_slice_original = lesion_info['z_slice_original']

        rows = self.lesions_df[(self.lesions_df['PatientNumber'] == patient_id) & (self.lesions_df['SliceNumber'] == z_slice_original)]
        if rows.empty:
            return 'Unknown'

        subtype_str = ' '.join([lesion for lesion in self.available_lesions if rows.iloc[0][lesion]]).strip()
        return subtype_str if subtype_str else 'Unknown'


class BHSDLoader(DatasetLoader):
    """Loads data from the Brain Hemorrhage Segmentation Dataset (BHSD).

    https://huggingface.co/datasets/Wendy-Fly/BHSD
    """
    def __init__(self, dataset_path: str, **kwargs):
        super().__init__(dataset_path, **kwargs)
        self.subtype_map = {1: 'Epidural', 2: 'Intraparenchymal', 3: 'Intraventricular', 4: 'Subarachnoid', 5: 'Subdural'}

    def get_patient_ids(self) -> List[str]:
        img_path = self.dataset_path / 'images'
        return [f.name for f in img_path.glob('*.nii.gz')]

    def load_patient_data(self, patient_id: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        img_file = self.dataset_path / 'images' / patient_id
        mask_file = self.dataset_path / 'ground_truths' / patient_id
        if not (img_file.exists() and mask_file.exists()):
            return None, None, None

        img_nii = nib.load(img_file)
        mask_nii = nib.load(mask_file)

        original_spacing = img_nii.header.get_zooms()[:3]
        resample_factor = [orig / new for orig, new in zip(original_spacing, self.target_spacing)]

        img_data = zoom(img_nii.get_fdata(), resample_factor, order=3, prefilter=True)
        mask_data = zoom(mask_nii.get_fdata(), resample_factor, order=0, prefilter=False)

        img_vol = np.rot90(img_data, 1).transpose(2, 0, 1)
        mask_vol = np.rot90(mask_data, 1).transpose(2, 0, 1)

        return img_vol, mask_vol.astype(int), np.array(resample_factor)

    def get_lesion_subtype(self, lesion_info: Dict) -> str:
        lesion_mask = lesion_info['lesion_mask']
        original_mask = lesion_info['original_mask']

        # Get the integer label from the original (pre-windowing) mask
        lesion_label_val = original_mask[lesion_mask].max()
        return self.subtype_map.get(lesion_label_val, 'Unknown')


class InstanceLoader(DatasetLoader):
    """Loads data from the INSTANCE 2022 ICH Dataset.

    https://instance.grand-challenge.org/
    """
    def __init__(self, dataset_path: str, **kwargs):
        super().__init__(dataset_path, **kwargs)

    def get_patient_ids(self) -> List[str]:
        img_path = self.dataset_path / 'train' / 'data'
        return sorted([int(f.name.split('.nii.gz')[0]) for f in img_path.glob('*.nii.gz')])

    def load_patient_data(self, patient_id: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        img_file = self.dataset_path / 'train' / 'data' / f'{patient_id:03d}.nii.gz'
        mask_file = self.dataset_path / 'train' / 'label' / f'{patient_id:03d}.nii.gz'
        if not (img_file.exists() and mask_file.exists()):
            return None, None, None

        img_nii = nib.load(img_file)
        mask_nii = nib.load(mask_file)

        original_spacing = img_nii.header.get_zooms()[:3]
        resample_factor = [orig / new for orig, new in zip(original_spacing, self.target_spacing)]

        img_data = zoom(img_nii.get_fdata(), resample_factor, order=3, prefilter=True)
        mask_data = zoom(mask_nii.get_fdata(), resample_factor, order=0, prefilter=False)

        img_vol = np.rot90(img_data, 1).transpose(2, 0, 1)
        mask_vol = np.rot90(mask_data, 1).transpose(2, 0, 1)

        return img_vol, mask_vol.astype(int), np.array(resample_factor)

    def get_lesion_subtype(self, lesion_info: Dict) -> str:
        return 'Unknown'


class SyntheticLoader(DatasetLoader):
    """Loads data from the INSTANCE 2022 ICH Dataset.

    https://instance.grand-challenge.org/
    """
    def __init__(self, phantoms: list[HeadPhantom], **kwargs):
        super().__init__('', **kwargs)
        self.subtype_map = {1: 'Epidural', 2: 'Intraparenchymal', 3: 'Intraventricular', 4: 'Subarachnoid', 5: 'Subdural'}
        self.subtype_abreviations = {'EDH': 'Epidural', 'IPH': 'Intraparenchymal', 'IVH': 'Intraventricular', 'SAH': 'Subarachnoid', 'SDH': 'Subdural'}
        self.phantoms = phantoms

    def get_patient_ids(self) -> List[str]:
        return list(range(len(self.phantoms)))

    def load_patient_data(self, patient_id: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:

        phantom = self.phantoms[patient_id]
        inverted_subtype_map = {v: k for k, v in self.subtype_map.items()}
        mask = np.zeros(phantom.lesions[0].mask.shape)
        for lesion in phantom.lesions:
            mask[lesion.mask] = inverted_subtype_map[self.subtype_abreviations[lesion.lesion_type]]

        original_spacing = phantom.spacings
        resample_factor = [orig / new for orig, new in zip(original_spacing, self.target_spacing)]

        img = phantom.get_CT_number_phantom()
        img_vol = zoom(img, resample_factor, order=3, prefilter=True)
        mask_vol = zoom(mask, resample_factor, order=0, prefilter=False)
        

        return img_vol, mask_vol.astype(int), np.array(resample_factor)

    def get_lesion_subtype(self, lesion_info: Dict) -> str:
        lesion_mask = lesion_info['lesion_mask']
        original_mask = lesion_info['original_mask']

        # Get the integer label from the original (pre-windowing) mask
        lesion_label_val = original_mask[lesion_mask].max()
        return self.subtype_map.get(lesion_label_val, 'Unknown')
# =============================================================================
# 3. MAIN PIPELINE EXECUTION MODULE
# This function orchestrates the entire process using the classes defined above.
# =============================================================================


def run_feature_extraction_pipeline(loader: DatasetLoader, return_images: bool = False, filename: str = None, montage_max=100) -> Any:
    """
    Runs the full feature extraction pipeline for a given dataset loader.

    Args:
        loader: An instantiated dataset loader (e.g., HssayeniLoader).
        return_images: If True, returns image and mask slices with the DataFrame.
        filename (str): If provided, saves the montage to this file instead of displaying.

    Returns:
        A pandas DataFrame with all extracted features, or a tuple including
        image/mask lists if return_images is True.
    """
    if filename:
        return_images = True
    feature_extractor = RadiomicsFeatureExtractor()
    all_lesions_data = []
    image_slices, mask_slices = [], []

    patient_ids = loader.get_patient_ids()
    for patient_id in tqdm(patient_ids, desc=f"Processing {type(loader).__name__}"):

        image_vol, mask_vol, resample_factor = loader.load_patient_data(patient_id)
        if image_vol is None or mask_vol.sum() == 0:
            continue

        labeled_mask, num_lesions = label(mask_vol, return_num=True, connectivity=3)

        for lesion_idx in range(1, num_lesions + 1):
            current_lesion_mask = (labeled_mask == lesion_idx)

            # Extract features
            features = feature_extractor.extract_features_for_lesion(image_vol, current_lesion_mask)
            if not features:
                continue

            # Get subtype using loader-specific logic
            z_slice_resampled = current_lesion_mask.sum(axis=(1, 2)).argmax()
            z_slice_original = int(round(z_slice_resampled / resample_factor[2]))

            subtype_info = {
                'patient_id': patient_id,
                'z_slice_original': z_slice_original,
                'lesion_mask': current_lesion_mask,
                'original_mask': mask_vol
            }
            subtype = loader.get_lesion_subtype(subtype_info)

            # Assemble final data record for this lesion
            record = {
                'ID': patient_id,
                'lesion_index': lesion_idx,
                'subtype': subtype,
                'volume_ml': (features['volume_voxel'] * np.prod(loader.target_spacing)) / 1000.0,
                'z_location_resampled': z_slice_resampled,
                'dataset': type(loader).__name__.replace("Loader", ""),
                **features  # Add all calculated features
            }
            all_lesions_data.append(record)

            if return_images and len(image_slices) < montage_max:
                image_slices.append(image_vol[z_slice_resampled])
                mask_slices.append(labeled_mask[z_slice_resampled])

    features_df = pd.DataFrame(all_lesions_data)
    # Convert appropriate columns to categorical for memory efficiency
    for col in ['ID', 'subtype', 'dataset', 'lesion_index']:
        if col in features_df.columns:
            features_df[col] = pd.Categorical(features_df[col])

    create_ich_lesion_montage(
        image_slices, mask_slices,
        dataframe=features_df,
        title=f"{type(loader).__name__.replace('Loader', '')} Dataset: {len(features_df)} Lesions",
        filename=filename
    )

    if return_images:
        return features_df, image_slices, mask_slices
    return features_df
