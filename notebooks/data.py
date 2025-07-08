import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from scipy.ndimage import find_objects, zoom
from pathlib import Path


def load_hssayeni_image_mask_pair(ct_ich_dataset_path, patient_id,
                                  target_spacing: tuple[float, float, float]=None,
                                  return_resample_factor=False):
    '''
    Loads and resamples a CT image and its corresponding ICH segmentation mask
    from the Hssayeni dataset to a uniform isotropic voxel spacing of 1x1x1 mm.

    This function incorporates a crucial preprocessing step for radiomics by
    resampling the data to a consistent voxel size, which helps ensure that
    shape and texture features are comparable across different scans.

    Dataset Source: https://physionet.org/content/ct-ich/

    Args:
        ct_ich_dataset_path (Path or str): Path to the root of the dataset.
        patient_id (float, int, or str): Patient identifier.
        target_spacing ()

    Returns:
        tuple: A tuple containing the resampled image volume and mask as numpy arrays.
               Returns (None, None) if files do not exist.
    '''
    ct_ich_dataset_path = Path(ct_ich_dataset_path)
    # Ensure patient_id is formatted correctly for the filename
    patient_id_str = f'{int(patient_id):03d}'
    img_file = ct_ich_dataset_path / 'ct_scans' / f'{patient_id_str}.nii'
    mask_file = ct_ich_dataset_path / 'masks' / f'{patient_id_str}.nii'

    img_vol = None
    mask = None
    resample_factor = None

    if img_file.exists() and mask_file.exists():
        # Load the NIfTI images using nibabel
        img_nii = nib.load(img_file)
        mask_nii = nib.load(mask_file)

        # --- Resampling Step ---
        
        img_data = img_nii.get_fdata()
        mask_data = mask_nii.get_fdata()
        if target_spacing is None:
            resampled_img_data = img_data
            resampled_mask_data = mask_data
        else:
            original_spacing = img_nii.header.get_zooms()[:3]
            resample_factor = [orig / new for orig, new in zip(original_spacing, target_spacing)]

            # Resample image with B-spline interpolation
            resampled_img_data = zoom(img_data, resample_factor, order=3, prefilter=True)

            # Resample mask with nearest-neighbor interpolation
            resampled_mask_data = zoom(mask_data, resample_factor, order=0, prefilter=False)
            # --- End Resampling Step ---

        # Apply original orientation transformations
        img_vol = np.rot90(resampled_img_data, 1).transpose(2, 0, 1)
        mask = np.rot90(resampled_mask_data, 1).transpose(2, 0, 1)

    if return_resample_factor:
        return img_vol, mask, resample_factor
    return img_vol, mask


def visualize_ich_projections(img, labeled_mask):
    """
    Displays the maximum intensity projections for a global view of all hemorrhages.
    
    Args:
        img (np.ndarray): The 3D CT image volume.
        labeled_mask (np.ndarray): The 3D mask with each hemorrhage uniquely labeled.
    """
    print("--- Global Maximum Intensity Projections ---")
    f, axs = plt.subplots(1, 3, gridspec_kw=dict(wspace=0.01, hspace=0), figsize=(12, 4), dpi=150)
    
    # Create a masked array for the overlay to handle transparency correctly
    masked_overlay = np.ma.masked_where(labeled_mask == 0, labeled_mask)

    for i, ax in enumerate(axs):
        ax.imshow(img.mean(axis=i), cmap='bone')
        ax.imshow(masked_overlay.max(axis=i), alpha=0.5, cmap='gist_rainbow', interpolation='none')
        ax.axis('off')
    plt.show()


def visualize_ich_slices(img, labeled_mask):
    """
    Generates and displays orthogonal cross-sectional views for each individual
    hemorrhage found in the labeled mask.

    Args:
        img (np.ndarray): The 3D CT image volume.
        labeled_mask (np.ndarray): The 3D mask with each hemorrhage uniquely labeled.
                                   (e.g., from skimage.measure.label).
    """
    print("\n--- Individual Hemorrhage Cross-Sectional Views ---")
    # Find the bounding box for each labeled object
    objects = find_objects(labeled_mask)
    
    # Define a brain window for better contrast
    window_center, window_width = 40, 80
    vmin = window_center - window_width / 2
    vmax = window_center + window_width / 2

    for i, obj_slice in enumerate(objects):
        if obj_slice is None:
            continue        
        label_id = i + 1

        # Create a binary mask for the current hemorrhage
        current_hemorrhage_mask = (labeled_mask == label_id)

        x_slice = current_hemorrhage_mask.sum(axis=(0, 1)).argmax()
        y_slice = current_hemorrhage_mask.sum(axis=(0, 2)).argmax()
        z_slice = current_hemorrhage_mask.sum(axis=(1, 2)).argmax()
 
        # Create a masked array for the overlay to handle transparency correctly
        masked_overlay = np.ma.masked_where(current_hemorrhage_mask == 0, current_hemorrhage_mask)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=100)
        fig.suptitle(f'Inspection of Hemorrhage ID: {label_id}', fontsize=16)

        # Axial View
        axs[0].imshow(img[z_slice, :, :], cmap='bone', vmin=vmin, vmax=vmax)
        axs[0].imshow(masked_overlay[z_slice, :, :], cmap='Reds', alpha=0.5, interpolation='none')
        axs[0].set_title(f'Axial View (Slice {z_slice})')
        axs[0].axis('off')

        # Coronal View
        axs[1].imshow(np.rot90(img[:, y_slice, :]), cmap='bone', vmin=vmin, vmax=vmax)
        axs[1].imshow(np.rot90(masked_overlay[:, y_slice, :]), cmap='Reds', alpha=0.5, interpolation='none')
        axs[1].set_title(f'Coronal View (Slice {y_slice})')
        axs[1].axis('off')

        # Sagittal View
        axs[2].imshow(np.rot90(img[:, :, x_slice]), cmap='bone', vmin=vmin, vmax=vmax)
        axs[2].imshow(np.rot90(masked_overlay[:, :, x_slice]), cmap='Reds', alpha=0.5, interpolation='none')
        axs[2].set_title(f'Sagittal View (Slice {x_slice})')
        axs[2].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()