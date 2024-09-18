"""
Module responsible for lesion insertion
"""

import numpy as np
from scipy.ndimage import center_of_mass

from .lesion_definition import spherical_lesion, insert_dural_3D


def add_sphere_lesion(self, mask: np.ndarray,
                      radius: list[int] = [20],
                      contrast: list[int] = [-100],
                      seed: int | None = None,
                      tol: int = 20) -> tuple:
    '''
    adds lesion to img in random location within mask of size radius
    and contrast level contrast

    :param img: array to insert lesion into
    :param mask: mask that specifies limits inside the `img` of
        potential insertion locations
    :param volume: int or list of ints, volume of the sphere lesion,
        if provided a list it will make concentric lesions
    :param contrast: int or list of ints, contrast of the sphere lesion,
        if provided a list it will make concentric lesions of contrasts

    :returns: img_w_lesion, lesion_vol, (z, x, y)
    '''
    if seed:
        tol = 1  # no need to keep trying if the seed is going to place in the same position each time
    if not isinstance(volume, list):
        volume = [volume]
    if not isinstance(contrast, list):
        contrast = [contrast]
    radii = [sphere_radius_from_volume(v) for v in volume]
    r = max(radii)

    vol = self.get_CT_number_phantom()

    counts = 0
    sphere = np.zeros_like(img, dtype=bool)
    while overlap < 0.8:  # can increase threshold to size of lesion
        lesion_vol = np.zeros_like(img)
        rng = np.random.default_rng(seed)
        z, x, y = np.argwhere(mask)[rng.integers(0, mask.sum())]
        if mask[z].sum() < np.pi*r**2:
            continue
        counts += 1
        sphere = spherical_lesion(img, center=(z, x, y), radius=r).transpose(1, 0, 2)
        overlap = np.sum(mask & sphere)/(np.sum(sphere))
        if counts > tol:
            raise ValueError("Failed to insert lesion into mask")

    lesion_vol = np.zeros_like(img)
    for ri in radii:
        for ci in contrast:
            sphere = spherical_lesion(img, center=(z, x, y),
                                      radius=ri).transpose(1, 0, 2)
            lesion_vol[sphere] += ci
    img_w_lesion = img + lesion_vol
    return img_w_lesion, lesion_vol, (z, x, y)


def _add_dural_lesion(spacing, self,
                      lesion_type, contrast, init_slice=None, seed=None):
    rng = np.random.default_rng(seed)
    dura_map = self.get_dura_map()
    volume = self.get_CT_number_phantom()

    init_slice = init_slice or rng.choice(
        np.where(dura_map.mean(axis=(1, 2)) > 0.015)[0])
    
    lesion_vol, volume = insert_dural_3D(spacing, self, init_slice,
                                 lesion_type, mass_effect=True)
    
    if not isinstance(volume, np.ndarray):
        volume = volume.numpy()

    img_w_lesion = volume.copy()
    img_w_lesion[lesion_vol == 1] = contrast
    z, x, y = center_of_mass(lesion_vol)
    return img_w_lesion, lesion_vol, (int(z), int(x), int(y))


def add_subdural_lesion(self, spacing: tuple, contrast: float = 70, init_slice: int | None = None, seed=None):
    '''
    adds subdural lesion to img within dura mask of given contrast level

    :param img: array to insert lesion into
    :param mask: mask that specifies limits inside the `img` of
        potential insertion locations, here a dura mask
    :param spacing: voxel spacings in mm
    :param contrast: int or list of ints, contrast of the sphere lesion,
        if provided a list it will make concentric lesions of contrasts

    :returns: img_w_lesion, lesion_vol, (z, x, y)
    '''
    return _add_dural_lesion(spacing, self, 'subdural', contrast, init_slice, seed=seed)


def add_epidural_lesion(self, spacing: tuple, contrast: float = 70, init_slice: int | None = None, seed=None):
    '''
    adds epidural lesion to img within dura mask of given contrast level

    :param img: array to insert lesion into
    :param mask: mask that specifies limits inside the `img` of
        potential insertion locations, here a dura mask
    :param spacing: voxel spacings in mm
    :param contrast: int or list of ints, contrast of the sphere lesion,
        if provided a list it will make concentric lesions of contrasts

    :returns: img_w_lesion, lesion_vol, (z, x, y
    '''
    return _add_dural_lesion(spacing, self, 'epidural', contrast, init_slice, seed=seed)

