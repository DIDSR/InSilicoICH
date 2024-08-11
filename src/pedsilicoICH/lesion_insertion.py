"""
Module responsible for lesion insertion
"""

import numpy as np
from .lesion_definition import spherical_lesion


def add_random_sphere_lesion(vol: np.ndarray, mask: np.ndarray,
                             radius: list[int] = [20],
                             contrast: list[int] = [-100]):
    '''
    adds lesion to vol in random location within mask of size radius
    and contrast level contrast'''
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
