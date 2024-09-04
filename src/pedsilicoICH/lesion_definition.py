"""
Module responsible for lesion definition
"""

import random
import math

import numpy as np
import skimage as ski
import scipy


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


def insert_dural_3D(spacing, volume, dura_map, init_slice, hematoma_type):

    boundary = dura_map
    [dz, dy, dx] = spacing  # set resolution, mm
    distances = [50/dx, 75/dx]  # length range in mm, will divide by voxel size (assuming isotropic) to convert

    num_slices = 11  # number of slices to have hemorrhage, try to make this odd
    slice_counter = 0  # will iterate on this
    iter_flag = True
    slices, rows, cols = volume.shape

    # desired_thickness = 0.5 # slice thickness in mm
    hemisphere = random.choice(['left', 'right'])  # can either be random or pre-defined

    if hemisphere == 'left':
        boundary[:, :, (int(cols/2) - 10):None] = 0.0
    elif hemisphere == 'right':
        boundary[:, :, :(int(cols/2) + 10)] = 0.0

    # import matplotlib.pyplot as plt
    # plt.imshow(boundary[150, :, :])

    hemorrhage_mask = np.zeros_like(boundary)
    while iter_flag:
        # print('iterating')
        if slice_counter == 0:  # need to do the slice in the middle of the hemorrhage, same as before
            # print('Processing center slice: ' + str(init_slice))
            temp_boundary = boundary[init_slice]
            dura_idx = np.argwhere(temp_boundary == 1.0)

            # choose a random start point, and calculate distance bfrom all available boundary voxels to start point
            start_point = dura_idx[random.choice(range(len(dura_idx)))]
            # print(start_point)

            distance_idx = np.zeros(len(dura_idx))
            for i in range(len(dura_idx)):
                distance_idx[i] = math.sqrt((start_point[0] - dura_idx[i][0])**2 + (start_point[1] - dura_idx[i][1])**2)

            # create list of possible end points and choose one at random
            close_voxel_list = np.where(np.logical_and(distance_idx > distances[0], distance_idx < distances[1]))
            end_point = dura_idx[random.choice(close_voxel_list[0])]

            orig_start = start_point
            orig_end = end_point

            # now that the two starting points for the hemorrhage have been defined, need to connect them
            # process should be the same on any given slice, and this function can be updated with new connection options
            filled_array, _, _ = connect_points(start=orig_start, end=orig_end,
                                                boundary=temp_boundary,
                                                hematoma_type=hematoma_type)
            # eventually will use boundary and connection coordinates to warp hemorrhage volume and original phantom

            hemorrhage_mask[init_slice] = filled_array

            slice_counter += 1

        # this is a bit messy but it will work for intended purpose: 
        # starting from hemorrhage origin, move down while shrinking distance between start/end point
        # then, move up from hemorrhage origin while doing the same
        if slice_counter <= (num_slices-1)/2: # move down from hemorrhage origin
            slice_idx = slice_counter
        elif slice_counter > (num_slices-1)/2: # start moving up from hemorrhage origin
            slice_idx = -1*(slice_counter - int((num_slices-1)/2))

        temp_boundary = boundary[init_slice-slice_idx]
        dura_idx = np.argwhere(temp_boundary == 1.0)
        # find closest boundary point to previous start
        distance_idx = np.zeros((len(dura_idx),2))

        distance_from_start = np.zeros(len(dura_idx))
        distance_from_end = np.zeros(len(dura_idx))
        for i in range(len(dura_idx)):
            distance_from_start[i] = math.sqrt((orig_start[0] - dura_idx[i][0])**2 + (orig_start[1] - dura_idx[i][1])**2)
            distance_from_end[i] = math.sqrt((orig_end[0] - dura_idx[i][0])**2 + (orig_end[1] - dura_idx[i][1])**2)

        new_start = dura_idx[np.argmin(distance_from_start)]
        new_end = dura_idx[np.argmin(distance_from_end)]

        filled_array, _, _ = connect_points(start=new_start, end=new_end, boundary=temp_boundary, hematoma_type=hematoma_type)

        hemorrhage_mask[init_slice-slice_idx] = filled_array

        slice_counter += 1

        if slice_counter == num_slices:
            iter_flag = False

    return hemorrhage_mask


def connect_points(start, end, boundary, hematoma_type):
    'draw a line connecting start and end points but following existing dura'
    rows, cols = boundary.shape
    costs = np.where(boundary, 0, 10000)
    path, _ = ski.graph.route_through_array(costs, start=(start[0], start[1]), end=(end[0], end[1]), fully_connected=True)

    indices = np.stack(path, axis=-1)
    boundary_route = np.zeros_like(boundary)
    boundary_route[indices[0], indices[1]] = 1.0

    boundary_coords = np.zeros((len(path), 2))  # we want to take the boundary path and save to numpy array for later
    for idx, coord in enumerate(path):
        boundary_coords[idx, :] = np.array(coord)

    # now it's time to add the bezier curve
    if hematoma_type == 'epidural':
        bezier_weight = 0.1 # weight should probably be below 0.5 to avoid ballooning too much
        bezier_middle = (int(rows/2), int(cols/2))  # center of the image, should probably randomize it somewhere along center later
    elif hematoma_type == 'subdural':
        bezier_weight = 0.5
        bezier_middle = boundary_coords[round(len(boundary_coords)/2)]  # use the middle point of the dura line 
        print(bezier_middle)
    else:
        bezier_weight = 0.0
        bezier_middle = (int(rows/2), int(cols/2))

    rr, cc = ski.draw.bezier_curve(r0=start[0], c0=start[1],
                                   r1=int(bezier_middle[0]),
                                   c1=int(bezier_middle[1]),
                                   r2=end[0], c2=end[1],
                                   weight=bezier_weight)

    connecting_route = np.zeros_like(boundary)
    connecting_route[rr, cc] = 1.0
    connect_coords = np.stack((rr, cc), axis=-1)

    # the bezier curve coordinate list isn't ordered from start point to end point
    # below is rudimentary but should order from start point to end point as long as weight < 1
    connect_coords = connect_coords.tolist()
    connect_coords.sort(key=lambda p: math.dist(p, [start[0], start[1]]))
    connect_coords = np.array(connect_coords)

    filled_array = scipy.ndimage.binary_fill_holes(np.where(np.add(connecting_route, boundary_route) > 0, 1.0, 0)).astype(int)

    return filled_array, boundary_coords, connect_coords