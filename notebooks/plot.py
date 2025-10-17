import math
from typing import List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from skimage.measure import regionprops
from scipy.ndimage import find_objects
from tqdm.auto import tqdm


def ct_windowed(image: np.ndarray, window: int = 80, level: int = 40) -> np.ndarray:
    """
    Applies a window and level to a CT image (in Hounsfield Units).

    Args:
        image (np.ndarray): The input CT image data.
        window (int): The width of the HU window.
        level (int): The center of the HU window.

    Returns:
        np.ndarray: The windowed image, scaled to [0, 255] for display.
    """
    min_val = level - window // 2
    max_val = level + window // 2

    # Clip the image to the window
    clipped_image = np.clip(image, min_val, max_val)

    # Normalize to 0-1 range
    normalized_image = (clipped_image - min_val) / window

    # Scale to 0-255 for standard image display
    return (normalized_image * 255).astype(np.uint8)


def generate_distinct_colors(n: int, cmap_name: str = 'tab10') -> List[Tuple[float, ...]]:
    """
    Generates a list of n visually distinct colors from a matplotlib colormap.
    """
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i) for i in np.linspace(0, 1, n)]


def create_ich_lesion_montage(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    dataframe: pd.DataFrame,
    show_bbox: bool = True,
    show_mask: bool = True,
    display_window: str = 'brain',
    title: str = '',
    filename: str = None
) -> None:
    """
    Creates and displays a square montage of ICH lesions with annotations.

    This function is designed to work with the output of `calculate_ich_features_hssayeni`
    when `return_images=True`. It expects each item in the lists to correspond to a
    single lesion defined in a row of the dataframe.

    Args:
        images (List[np.ndarray]): A list of 2D NumPy arrays (CT slices).
        masks (List[np.ndarray]): A list of 2D NumPy arrays (labeled masks).
                                   Each mask can contain multiple labeled regions.
        dataframe (pd.DataFrame): DataFrame where each row `i` corresponds to a
                                  specific lesion in `images[i]` and `masks[i]`.
                                  Must contain 'ID', 'lesion_index', 'subtype',
                                  'volume_ml', 'glcm_contrast', 'glcm_correlation'.
        show_bbox (bool): If True, draws a bounding box around the lesion.
        show_mask (bool): If True, overlays a colored mask on the lesion.
        display_window (str): The CT window to apply. 'brain' is standard.
        filename (str): If provided, saves the montage to this file instead of displaying.
    """
    num_lesions = len(images)
    if num_lesions == 0:
        print("No images to display.")
        return

    colors = generate_distinct_colors(num_lesions)
    grid_size = math.ceil(math.sqrt(num_lesions))

    fig, axes = plt.subplots(
        grid_size, grid_size,
        figsize=(grid_size * 3, grid_size * 3),
        gridspec_kw=dict(hspace=0.1, wspace=0.1),
        dpi=150
    )
    axes = axes.flatten()

    for i in range(num_lesions):
        ax = axes[i]
        row = dataframe.iloc[i]
        img_slice = images[i]
        labeled_mask_slice = masks[i]
        
        # --- Critical Improvement: Isolate the specific lesion for this row ---
        lesion_index = row['lesion_index']
        lesion_mask = (labeled_mask_slice == lesion_index).astype(np.uint8)

        # Display the windowed CT image
        windowed_img = ct_windowed(img_slice, window=80, level=40) if display_window == 'brain' else img_slice
        ax.imshow(windowed_img, cmap='gray')

        if np.any(lesion_mask):
            # Find properties only for the current lesion
            props = regionprops(lesion_mask)
            if not props:
                ax.axis('off')
                continue # Skip if lesion mask is empty after processing
            
            reg = props[0] # There will only be one region

            if show_mask:
                masked_overlay = np.ma.masked_where(lesion_mask == 0, lesion_mask)
                ax.imshow(masked_overlay, cmap=plt.cm.colors.ListedColormap([colors[i]]), alpha=0.4, interpolation='none')

            if show_bbox:
                # Create a rectangle patch
                minr, minc, maxr, maxc = reg.bbox
                rect = patches.Rectangle(
                    (minc, minr), maxc - minc, maxr - minr,
                    linewidth=1.5, edgecolor=colors[i], facecolor='none'
                )
                ax.add_patch(rect)

        # Add annotation text
        id = row['ID']
        if isinstance(id, (int, float)):
            id = int(id)
        elif isinstance(id, str):
            id = id[:8]  # Truncate long IDs for display
        else:
            id = row['ID']
        ann_text = (
            f"ID: {id}, Lesion: {row['lesion_index']}\n"
            f"{row['subtype']}\n"
            f"Vol: {row['volume_ml']:.2f}mL, Sph: {row.get('sphericity', 0):.2f}\n"
            f"Contr: {row['contrast']:.2f}, Corr: {row['correlation']:.2f}"
        )
        ax.text(
            0.03, 0.97, ann_text,
            transform=ax.transAxes,
            fontsize=6, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        )
        ax.axis('off')

    # Hide unused subplots
    for j in range(num_lesions, len(axes)):
        axes[j].axis('off')

    fig.suptitle(title, fontsize=16)
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=600)
        print(f"Montage saved to {filename}")
    else:
        plt.show()


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


def get_lesion_slices_efficiently(dataframe, loader):
    """
    Efficiently extracts 2D image and mask slices corresponding to lesions in a dataframe.

    This function groups lesions by patient ID to ensure each 3D volume is loaded
    from disk only once, significantly speeding up the process.

    Args:
        dataframe (pd.DataFrame): A dataframe containing lesion features, including
                                  'ID' and 'z_location_resampled' columns.
        loader (DatasetLoader): An instantiated data loader object capable of
                                loading patient data via a `load_patient_data` method.

    Returns:
        tuple[list, list]: A tuple containing two lists: the extracted 2D image
                           slices and the corresponding 2D mask slices.
    """
    # Ensure the dataframe is sorted by patient ID to process sequentially
    # This isn't strictly necessary with groupby but can be good practice.
    dataframe = dataframe.sort_values('ID').reset_index()

    images = [None] * len(dataframe)
    masks = [None] * len(dataframe)

    # Group by patient ID to process one patient at a time
    for patient_id, group in tqdm(dataframe.groupby('ID'), desc="Extracting Slices"):

        # Load the full 3D data for the current patient ONCE
        img_vol, mask_vol, _ = loader.load_patient_data(patient_id)

        if img_vol is None:
            # Skip if the patient data can't be loaded
            continue

        # For each lesion from this patient, find its 2D slice
        for idx, row in group.iterrows():
            z_slice = row['z_location_resampled']

            # Place the slice in the correct position in the output lists
            # This preserves the original dataframe's order
            original_index = row['index']
            images[original_index] = img_vol[z_slice]
            masks[original_index] = mask_vol[z_slice]

    # Filter out any None values if some patients failed to load
    final_images = [img for img in images if img is not None]
    final_masks = [mask for mask in masks if mask is not None]

    return final_images, final_masks
