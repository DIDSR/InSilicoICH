"""
Module responsible for lesion insertion
"""

import numpy as np
from scipy.ndimage import center_of_mass

from .lesion_definition import spherical_lesion, insert_dural_3D


def add_random_sphere_lesion(vol: np.ndarray, mask: np.ndarray,
                             radius: list[int] = [20],
                             contrast: list[int] = [-100]) -> tuple:
    '''
    adds lesion to vol in random location within mask of size radius
    and contrast level contrast

    :param vol: array to insert lesion into
    :param mask: mask that specifies limits inside the `vol` of
        potential insertion locations
    :param radius: int or list of ints, radius of the sphere lesion,
        if provided a list it will make concentric lesions
    :param contrast: int or list of ints, contrast of the sphere lesion,
        if provided a list it will make concentric lesions of contrasts

    :returns: img_w_lesion, lesion_vol, (z, x, y)
    '''
    if not isinstance(radius, list):
        radius = [radius]
    if not isinstance(contrast, list):
        contrast = [contrast]
    r = max(radius)
    volume = (4/3*np.pi*r**3)*0.95

    counts = 0
    sphere = np.zeros_like(vol, dtype=bool)
    while np.sum(mask & sphere) < volume: #can increase threshold to size of lesion
        lesion_vol = np.zeros_like(vol)
        z, x, y = np.argwhere(mask)[np.random.randint(0, mask.sum())]
        if mask[z].sum() < np.pi*r**2:
            continue
        counts += 1
        sphere = spherical_lesion(vol, center=(z, x, y), radius=r).transpose(1, 0, 2)
        if counts > 20:
            raise ValueError("Failed to insert lesion into mask")

    lesion_vol = np.zeros_like(vol)
    for ri in radius:
        for ci in contrast:
            sphere = spherical_lesion(vol, center=(z, x, y), radius=ri).transpose(1, 0, 2)
            lesion_vol[sphere] += ci
    img_w_lesion = vol + lesion_vol
    return img_w_lesion, lesion_vol, (z, x, y)


def _add_dural_lesion(spacing, volume, dura_map,
                      lesion_type, contrast, init_slice=None,):
    init_slice = init_slice or np.random.choice(
        np.where(dura_map.mean(axis=(1, 2)) > 0.01)[0])
    lesion_vol = insert_dural_3D(spacing, volume, dura_map, init_slice,
                                 lesion_type)
    img_w_lesion = volume.copy()
    img_w_lesion[lesion_vol == 1] = contrast
    z, x, y = center_of_mass(lesion_vol)
    return img_w_lesion, lesion_vol, (int(z), int(x), int(y))


def add_subdural_lesion(vol: np.ndarray, mask: np.ndarray, spacing: tuple,
                        contrast: float = 70, init_slice: int | None = None):
    '''
    adds subdural lesion to vol within dura mask of given contrast level

    :param vol: array to insert lesion into
    :param mask: mask that specifies limits inside the `vol` of
        potential insertion locations, here a dura mask
    :param spacing: voxel spacings in mm
    :param contrast: int or list of ints, contrast of the sphere lesion,
        if provided a list it will make concentric lesions of contrasts

    :returns: img_w_lesion, lesion_vol, (z, x, y)
    '''
    return _add_dural_lesion(spacing, vol,
                             mask, 'subdural', contrast, init_slice)


def add_epidural_lesion(vol: np.ndarray, mask: np.ndarray, spacing: tuple,
                        contrast: float = 70, init_slice: int | None = None):
    '''
    adds epidural lesion to vol within dura mask of given contrast level

    :param vol: array to insert lesion into
    :param mask: mask that specifies limits inside the `vol` of
        potential insertion locations, here a dura mask
    :param spacing: voxel spacings in mm
    :param contrast: int or list of ints, contrast of the sphere lesion,
        if provided a list it will make concentric lesions of contrasts

    :returns: img_w_lesion, lesion_vol, (z, x, y)
    '''
    return _add_dural_lesion(spacing, vol,
                             mask, 'epidural', contrast, init_slice)
