"""
Correct the generated suture orientations/geometry.
"""

import os
import sys
import nibabel as nib
import nrrd
import numpy as np
from scipy.ndimage import affine_transform
import SimpleITK as sitk

main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
sys.path.append(main_directory)


def load_image(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".nii", ".gz"]:
        shape, origin, spacing, affine, data = get_nifti_info(file_path)
        return data, affine
    elif ext == ".nrrd":
        data, header = nrrd.read(file_path)
        # Fake affine if not present
        spacing = np.array(header.get("space directions", np.eye(3)))
        origin = np.array(header.get("space origin", [0, 0, 0]))
        affine = np.eye(4)
        affine[:3, :3] = spacing
        affine[:3, 3] = origin
        return data, affine
    else:
        raise ValueError("Unsupported file format: {}".format(ext))


def resize_to_target(source_path, target_path):
    src_data, src_affine = load_image(source_path)
    tgt_data, tgt_affine = load_image(target_path)

    # Compute new shape and transform
    src_voxel_size = np.linalg.norm(src_affine[:3, :3], axis=0)
    tgt_voxel_size = np.linalg.norm(tgt_affine[:3, :3], axis=0)

    scale_factors = src_voxel_size / tgt_voxel_size
    tgt_shape = tgt_data.shape

    zoom_matrix = np.diag(scale_factors)
    offset = np.zeros(3)

    # Perform affine transformation
    resized = affine_transform(
        src_data,
        matrix=zoom_matrix,
        offset=offset,
        output_shape=tgt_shape,
        order=1,  # linear interpolation
    )

    return resized, tgt_affine


def save_nifti(array, affine, path_save):
    nifti_img = nib.Nifti1Image(array, affine)
    nib.save(nifti_img, path_save)
    print("saved", path_save)


def get_nifti_info(nifti_path):
    img = nib.load(nifti_path)
    array = img.get_fdata()
    shape = img.shape  # (z, y, x) or (x, y, z), depending on orientation
    affine = img.affine  # 4x4 affine transformation matrix
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))  # voxel spacing
    origin = affine[:3, 3]  # origin (translation component)

    return shape, origin, spacing, affine, array


def nrrd_to_nifti(path_nrrd, path_save_nifti):
    # Load NRRD file
    nrrd_data, nrrd_header = nrrd.read(path_nrrd)

    # Extract space directions and origin
    space_directions = nrrd_header.get("space directions")
    space_origin = nrrd_header.get("space origin")

    # Build affine matrix
    affine = np.eye(4)
    if space_directions is not None:
        affine[:3, :3] = np.array(space_directions)
    if space_origin is not None:
        affine[:3, 3] = np.array(space_origin)

    # Create NIfTI image
    nifti_img = nib.Nifti1Image(nrrd_data, affine)

    # Save as NIfTI
    nib.save(nifti_img, path_save_nifti)


def resample_to_target_sitk(source_path, target_path, output_path, interp="linear"):
    # Read both images
    source = sitk.ReadImage(source_path)
    target = sitk.ReadImage(target_path)

    # Choose interpolation method
    interp_dict = {
        "linear": sitk.sitkLinear,
        "nearest": sitk.sitkNearestNeighbor,
        "bspline": sitk.sitkBSpline,
    }
    interpolator = interp_dict.get(interp.lower(), sitk.sitkLinear)

    # Resample image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(target)
    resampler.SetInterpolator(interpolator)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetDefaultPixelValue(0)

    resampled = resampler.Execute(source)

    # Save resampled image
    sitk.WriteImage(resampled, output_path)


def adjust_array_roll_manual(data):
    shift = 36
    axis = 1
    shifted = np.roll(data, shift=shift, axis=axis)

    # Optional: zero out the wrapped-around part
    if shift > 0:
        slicer = [slice(None)] * data.ndim
        slicer[axis] = slice(0, shift)
        shifted[tuple(slicer)] = 0
    elif shift < 0:
        slicer = [slice(None)] * data.ndim
        slicer[axis] = slice(shift, None)
        shifted[tuple(slicer)] = 0

    return shifted


reference_nifti_path = os.path.join(
    main_directory, "src/NIHPD_Head_Phantom", "nihpd_asym_04.5-08.5_mask.nii"
)

suture_nrrd_path = os.path.join(
    main_directory,
    "src/insilicoICH/annotations/suture/NIHPD_Head_Phantom",
    "labelmap.nrrd",
)

suture_nifti_path = os.path.join(
    main_directory,
    "src/insilicoICH/annotations/suture/NIHPD_Head_Phantom",
    "labelmap.nii",
)

corrected_suture_nifti_path = os.path.join(
    main_directory,
    "src/insilicoICH/annotations/suture/NIHPD_Head_Phantom",
    "sutures.nii",
)

nrrd_to_nifti(
    suture_nrrd_path,
    os.path.join(
        main_directory,
        "src/insilicoICH/annotations/suture/NIHPD_Head_Phantom",
        "labelmap.nii",
    ),
)

array_corrected_suture, affine_corrected_suture = resize_to_target(
    source_path=suture_nifti_path,
    target_path=reference_nifti_path,
)

save_nifti(array_corrected_suture, affine_corrected_suture, corrected_suture_nifti_path)

resample_to_target_sitk(
    suture_nifti_path, reference_nifti_path, corrected_suture_nifti_path
)

shape, origin, spacing, affine, array = get_nifti_info(corrected_suture_nifti_path)
array = array[:, ::-1, :]
corrected_suture_nifti_path_adjusted = os.path.join(
    main_directory,
    "src/insilicoICH/annotations/suture/NIHPD_Head_Phantom",
    "sutures_adjust.nii",
)
save_nifti(array, affine, corrected_suture_nifti_path_adjusted)

corrected_suture_nifti_path_adjuste_shifted = os.path.join(
    main_directory,
    "src/insilicoICH/annotations/suture/NIHPD_Head_Phantom",
    "skull_sutures.nii.gz",
)
shape, origin, spacing, affine, array = get_nifti_info(
    corrected_suture_nifti_path_adjusted
)
array = adjust_array_roll_manual(array)
save_nifti(array, affine, corrected_suture_nifti_path_adjuste_shifted)
