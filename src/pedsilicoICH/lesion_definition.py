"""
Module responsible for lesion definition
"""

import numpy as np


def spherical_lesion(phantom: np.ndarray,
                     center: tuple | None = None, radius: tuple | None = None):
    '''
    Returns binary sphere mask based on input phantom array and
    center coordinates and radii parameters

    sphere defined as r^2 = z^2 + x^2 + y^2

    :param phantom: 3D array to add sphere to 
    '''
    center = center or [dim//2 for dim in phantom.shape]
    radius = radius or [dim//10 for dim in phantom.shape]
    z, x, y = np.meshgrid(range(phantom.shape[0]),
                          range(phantom.shape[1]),
                          range(phantom.shape[2]))
    distance_matrix = (z - center[0])**2 + (x-center[1])**2 + (y-center[2])**2
    return np.where(distance_matrix > radius**2, False, True)
