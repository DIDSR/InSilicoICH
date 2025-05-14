"""
Module responsible for lesion definition
"""
import math

import numpy as np
import skimage as ski
import scipy
import sys
from monai.transforms import RandAffine


def get_perimeter(lesion):
    return ski.morphology.binary_dilation(lesion, np.ones((3, 3))) ^\
           ski.morphology.binary_erosion(lesion, np.ones((3, 3)))


def elliptical_lesion(shape: tuple | list,
                      center: tuple | None = None,
                      radius: tuple | None = None,
                      random_rotate: bool | int = True):
    '''
    Returns binary elliptical mask based on input matrix shape and
    center coordinates and radii parameters

    sphere defined as r^2 = z^2 + x^2 + y^2

    :param shape: sequence of ints, shape of the new array
    :param center: sequence of ints, coord
    :param radius: sequence of 3 ints specifying the 3 semimajor axes
    :param random_rotate: bool or int, if an integer is given, sets random
        seed of transform for repeatability.
    '''
    if isinstance(radius, np.ndarray):
        radius = list(radius)
    center = center or [dim//2 for dim in shape]
    radius = radius or [dim//10 for dim in shape]
    if not isinstance(radius, list | tuple):
        radius = 3*[radius]
    ell = ski.draw.ellipsoid(*radius)
    if random_rotate:
        transform = RandAffine(prob=1, rotate_range=[np.pi/2, np.pi/2, np.pi/2],
                               scale_range=[0.1, 0.1, 0.1], padding_mode="zeros")
        if isinstance(random_rotate, int):
            transform.set_random_state(seed=random_rotate)

        ell = np.pad(ell, ((int(max(radius)-radius[0]),),
                           (int(max(radius)-radius[1]),),
                           (int(max(radius)-radius[2]),)))
        ell = transform(ell)

    starts = center - np.array(ell.shape)//2
    ends = center + np.array(ell.shape)//2 + 1
    lesion_only = np.zeros(shape)
    lesion_only[starts[0]:ends[0],
                starts[1]:ends[1],
                starts[2]:ends[2]] = ell
    return np.where(lesion_only > 0, True, False)


def insert_dural(phantom, desired_volume, hematoma_type, mass_effect, seed=None):

    random = np.random.default_rng(seed)

    num_slices = coverage_from_volume(volume=desired_volume,
                                      hematoma_type=hematoma_type,
                                      slice_thickness=phantom.dz)
    ab = (desired_volume*2000)/(num_slices * phantom.dz)  # using ABC/2 formula (although /2000 for mL and mm)
    if hematoma_type == 'EDH':
        desired_distance = math.sqrt(4*ab) # assume that length of epidural hemorrhage is about 4 times the width
    elif hematoma_type == 'SDH':
        desired_distance = math.sqrt(11*ab) # assume that length of epidural hemorrhage is about 11 times the width

    HU_array = phantom.get_CT_number_phantom()

    # TODO: better logic for hemorrhage starting slice
    init_slice = int(random.choice(np.linspace(0, int(HU_array.shape[0]/2), int(HU_array.shape[0]/2) + 1)))

    # initialize arrays, maps, and masks
    new_volume = np.copy(HU_array)
    boundary = phantom.get_dura_map()
    hemorrhage_mask = np.zeros_like(boundary)

    if phantom.__class__.__name__ == 'MIDA_Head':
        phantom_name = 'MIDA_Head'
        skull_map = phantom.get_skull_map()
        mask = skull_map
    elif phantom.__class__.__name__ == 'NIHPD_Head':
        phantom_name = 'NIHPD_Head'
        skull_map = ski.morphology.binary_dilation(phantom.get_skull_map(), np.ones(3*[5]))
        mask = phantom.mask
    else:
        skull_map = phantom.get_skull_map()

    distances = [(desired_distance-5)/phantom.spacings[2], (desired_distance+5)/phantom.spacings[2]]

    hemisphere = random.choice(['left', 'right'])  # can either be random or pre-defined
    if hemisphere == 'left':
        boundary[:, :, (int(HU_array.shape[2]/2) - 10):None] = 0.0
    elif hemisphere == 'right':
        boundary[:, :, :(int(HU_array.shape[2]/2) + 10)] = 0.0

    # begin iteration
    iter_flag = True
    slice_counter = slice_idx = 0
    while iter_flag:

        current_vol = ((phantom.dx * phantom.dy * phantom.dz) * hemorrhage_mask.sum())/1000
        if current_vol > desired_volume:
            iter_flag = False

        if slice_counter == 0:  # need to do the slice in the middle of the hemorrhage, same as before

            tol = 2000
            count = 0
            failure_occured = False
            while count < tol:
                temp_boundary = boundary[init_slice]
                dura_idx = np.argwhere(temp_boundary == 1.0)

                try:
                    # choose a random start point, and calculate distance from all available boundary voxels to start point
                    start_point = dura_idx[random.choice(range(len(dura_idx)))]

                    distance_idx = np.zeros(len(dura_idx))
                    for i in range(len(dura_idx)):
                        distance_idx[i] = math.sqrt((start_point[0] - dura_idx[i][0])**2 + (start_point[1] - dura_idx[i][1])**2)

                    # create list of possible end points and choose one at random
                    close_voxel_list = np.where(np.logical_and(distance_idx > distances[0], distance_idx < distances[1]))
                    end_point = dura_idx[random.choice(close_voxel_list[0])]

                    orig_start = new_start = start_point
                    orig_end = new_end = end_point
                except:
                    count += 1
                    init_slice = int(random.choice(np.linspace(0, int(HU_array.shape[0]/2), int(HU_array.shape[0]/2) + 1)))
                    if count == tol:
                        failure_occured = True
                else:
                    count = tol
            if failure_occured:
                raise RuntimeError(f'lesion insertion failed with requested volume: {desired_volume} mL, try a smaller volume')
            
            # connect the start and end points of the hemorrhage 
            filled_array, boundary_coords, connect_coords =\
                connect_points(start=orig_start,
                               end=orig_end,
                               boundary=temp_boundary,
                               hematoma_type=hematoma_type,
                               initial_slice=True)

            if mass_effect:
                try:
                    warped_slice = warp_slice(HU_array[init_slice, :],
                                              skull_map[init_slice, :],
                                              mask[init_slice, :],
                                              boundary_coords, connect_coords,
                                              hematoma_type,
                                              phantom_name)
                    new_volume[init_slice, :, :] = warped_slice
                except ValueError:
                    Warning(f'Failed to perform mass effect insertion for\
                          volume: {desired_volume}, now inserting with mass\
                          effect to 0')
                    new_volume[init_slice] = HU_array[init_slice]
                    phantom.mass_effect = 0

            hemorrhage_mask[init_slice, :, :] = filled_array

            slice_counter += 1

        # starting from init_slice, first move down slices 
        # then, move up from init_slice while doing the same
        if slice_counter <= (num_slices-1)/2:  # move down from init_slice
            slice_idx = slice_counter
        elif slice_counter > (num_slices-1)/2:  # start moving up from init_slice
            slice_idx = -1*(slice_counter - int((num_slices-1)/2))

        temp_boundary = boundary[init_slice-slice_idx]
        dura_idx = np.argwhere(temp_boundary == 1.0)

        if len(dura_idx) != 0:  # check that top or bottom of brain wasn't reached
            
            # find closest boundary point to previous start
            distance_idx = np.zeros((len(dura_idx), 2))

            distance_from_start = np.zeros(len(dura_idx))
            distance_from_end = np.zeros(len(dura_idx))

            if abs(slice_idx) == 1:
                for i in range(len(dura_idx)):
                    distance_from_start[i] = math.sqrt((orig_start[0] - dura_idx[i][0])**2 + (orig_start[1] - dura_idx[i][1])**2)
                    distance_from_end[i] = math.sqrt((orig_end[0] - dura_idx[i][0])**2 + (orig_end[1] - dura_idx[i][1])**2)
            else:
                for i in range(len(dura_idx)):
                    distance_from_start[i] = math.sqrt((new_start[0] - dura_idx[i][0])**2 + (new_start[1] - dura_idx[i][1])**2)
                    distance_from_end[i] = math.sqrt((new_end[0] - dura_idx[i][0])**2 + (new_end[1] - dura_idx[i][1])**2)

            new_start = dura_idx[np.argmin(distance_from_start)]
            new_end = dura_idx[np.argmin(distance_from_end)]

            filled_array, boundary_coords, connect_coords =\
                connect_points(start=new_start,
                               end=new_end,
                               boundary=temp_boundary,
                               hematoma_type=hematoma_type,
                               initial_slice=False)
            
            try:
                new_start = boundary_coords[1:-1][0]
                new_end = boundary_coords[1:-1][-1]
            except:
                new_start = dura_idx[np.argmin(distance_from_start)]
                new_end = dura_idx[np.argmin(distance_from_end)]
        
            if mass_effect:
                try:
                    warped_slice = warp_slice(HU_array[init_slice-slice_idx, :],
                                                skull_map[init_slice-slice_idx, :],
                                                mask[init_slice-slice_idx, :],
                                                boundary_coords, connect_coords, 
                                                hematoma_type,
                                                phantom_name)
                    new_volume[init_slice-slice_idx] = warped_slice
                except ValueError:
                    Warning(f'Failed to perform mass effect insertion for\
                          volume: {desired_volume}, now inserting with mass\
                          effect to 0')
                    new_volume[init_slice-slice_idx] =\
                        HU_array[init_slice-slice_idx]
                    phantom.mass_effect = 0

            hemorrhage_mask[init_slice-slice_idx] = filled_array

            slice_counter += 1

            if slice_counter == num_slices:
                iter_flag = False

        else:
            slice_counter += 1
            if slice_counter == num_slices:
                iter_flag = False
    
    return hemorrhage_mask.astype(bool), new_volume


def connect_points(start, end, boundary, hematoma_type, initial_slice=False):
    '''
    Creates two lines connecting start and end points:
    1. Line following existing dura (borders skull)
    2. Bezier curve `inside` brain
    '''

    # Define first line (following dura)
    rows, cols = boundary.shape
    costs = np.where(boundary, 0, 10000)
    path, _ = ski.graph.route_through_array(costs, start=(start[0], start[1]), end=(end[0], end[1]), fully_connected=False)

    indices = np.stack(path, axis=-1)
    boundary_route = np.zeros_like(boundary)
    boundary_route[indices[0], indices[1]] = 1.0

    boundary_coords = np.zeros((len(path), 2))
    for idx, coord in enumerate(path):
        boundary_coords[idx, :] = np.array(coord)

    # Define second line (Bezier curve)
    successful_bezier = False
    # init bezier parameters:
    if hematoma_type == 'EDH':
        bezier_weight = 0.14  # changed from 0.14, # weight should probably be below 0.2 to avoid ballooning too much, but line breaks if below 0.04....
        bezier_middle = (int(rows/2), int(cols/2))  # center of the image, should probably randomize it somewhere along center later
    elif hematoma_type == 'SDH':
        bezier_weight = 2 # previously 0.5
        bezier_middle = boundary_coords[round(len(boundary_coords)/2)]  # use the middle point of the dura line
    else:
        bezier_weight = 0.0
        bezier_middle = (int(rows/2), int(cols/2))

    while successful_bezier == False:
        rr, cc = ski.draw.bezier_curve(r0=start[0], c0=start[1],
                                    r1=int(bezier_middle[0]),
                                    c1=int(bezier_middle[1]),
                                    r2=end[0], c2=end[1],
                                    weight=bezier_weight)

        connecting_route = np.zeros_like(boundary)
        connecting_route[rr, cc] = 1.0
        connect_coords = np.stack((rr, cc), axis=-1)

        # check if connecting route includes start and end:
        if (connecting_route[start[0], start[1]] == 1) & (connecting_route[end[0], end[1]] == 1):
            successful_bezier = True
        else:
            if bezier_weight != 0: # if bezier curve was unsuccessful with a nonzero weight, try with 0:
                bezier_weight = 0
            else:
                sys.exit('Unable to create bezier curve with weight=0')

    # returned coordinate list isn't ordered from start point to end point
    # below is rudimentary but should order from start point to end point as long as weight < 1
    connect_coords = connect_coords.tolist()
    connect_coords.sort(key=lambda p: math.dist(p, [start[0], start[1]]))
    connect_coords = np.array(connect_coords)

    filled_array = scipy.ndimage.binary_fill_holes(np.where(np.add(connecting_route, boundary_route) > 0, 1.0, 0)).astype(int)

    return filled_array, boundary_coords, connect_coords


def warp_slice(axial_slice, skull_slice, mask, src, dst, hematoma_type, phantom_name):
    '''
    performs warp of 2D slice according to hematoma boundary coordinates
    while maintaining a rigid skull
    '''
    
    if phantom_name == 'MIDA_Head':
        flood_mask = ski.segmentation.flood(skull_slice, seed_point=(0, 0))
        skull_slice[flood_mask] = 1
        brain_mask = np.where(skull_slice == 1, 0, 1)
    
        # using the entire inner boundary of the skull mask seems to work great as anchor points
        skull_boundary = ski.segmentation.find_boundaries(skull_slice, mode='inner', background=0)
        skull_idx = np.argwhere(skull_boundary == True)
        skull_sample = np.argwhere(skull_boundary != 0)

    elif phantom_name == 'NIHPD_Head':
        # use brain mask to define anchor points
        skull_boundary = ski.segmentation.find_boundaries(mask, mode='outer', background=0)
        skull_idx = np.argwhere(skull_boundary == True)
        skull_sample = np.argwhere(skull_boundary != 0)
        brain_mask = mask

    # initialize warp source and destination with skull indices in both (shouldn't move!)
    warp_src = warp_dst = skull_sample # initialize warp_src and warp_dst with the skull boundary voxels

    src_subset = src[np.round(np.linspace(0, len(src)-1, 5)).astype(int)] # subsample points from the src points
    dst_subset = dst[np.round(np.linspace(0, len(dst)-1, 5)).astype(int)] # subsample points from the dst points

    warp_src = np.insert(warp_src, 0, src_subset, axis=0) # insert src subset into main warp list
    warp_dst = np.insert(warp_dst, 0, dst_subset, axis=0) # insert dst subset into main warp list

    # insert the four corner coordinates for added warp stability
    warp_src = np.insert(warp_src, 0, [[0, 0],[0, axial_slice.shape[1]],[axial_slice.shape[0], 0],[axial_slice.shape[0], axial_slice.shape[1]]], axis=0)
    warp_dst = np.insert(warp_dst, 0, [[0, 0],[0, axial_slice.shape[1]],[axial_slice.shape[0], 0],[axial_slice.shape[0], axial_slice.shape[1]]], axis=0)

    # find transform and execute warp
    tps = ski.transform.ThinPlateSplineTransform()
    tps.estimate(np.flip(warp_dst), np.flip(warp_src))
    warped_slice = ski.transform.warp(axial_slice, tps, preserve_range=True, order=0)

    # trying to warp around small subdural hematomas on superior brain slices may result
    # in a warp artifact where the image just becomes all (or mostly). if this happens, skip
    # mass effect and just use original brain slice
    # TODO: find more elegant methods for fixing warping artifacts 
    if (np.mean(warped_slice) > -10) & (np.mean(warped_slice) < 10):
        warped_slice = axial_slice
    else: # if warp was successful, check for spurious hyperdense voxels from warp and replace 
        masked_axial = axial_slice*brain_mask
        # new code to try to "fix" skull warping into brain
        problem_voxels = np.argwhere((skull_slice != 1) & (warped_slice > 50))
        for index in problem_voxels:
            if hematoma_type == 'EDH' or 'SDH':
                warped_slice[index[0], index[1]] = 40 # HU value of dura mater
            elif hematoma_type == 'IPH':
                warped_slice[index[0], index[1]] = masked_axial[masked_axial!=0].mean()

        # finally, replace all voxels outside brain with original voxels
        warped_slice = np.where(brain_mask==1, warped_slice, axial_slice)

    return warped_slice


def coverage_from_volume(volume, hematoma_type, slice_thickness):  # see RSNA_BHDS_explore.ipynb for logarithmic fit
    if hematoma_type == 'EDH':
        #slice_coverage = 13.942*math.log(volume) + 13.449
        z_coverage = 10.231*math.log(volume) + 19.094
    elif hematoma_type == 'SDH':
        #z_coverage = 17.739*math.log(volume) + 17.314
        z_coverage = 10.380*math.log(volume) + 24.480
    elif hematoma_type == 'IPH':
        #z_coverage = 8.7064*math.log(volume) + 18.148  # for now, this is intraparenchymal
        z_coverage = 6.925*math.log(volume) + 17.315
    # unused
    elif hematoma_type == 'SAH':
        #z_coverage = 17.181*math.log(volume) + 27.42
        z_coverage = 5.383*math.log(volume) + 18.237
    elif hematoma_type == 'IVH':
        #z_coverage = 11.341*math.log(volume) + 25.45
        z_coverage = 6.657*math.log(volume) + 20.492
    # convert units from mm to number of slices
    slice_coverage = z_coverage / slice_thickness

    # round to nearest odd number
    slice_coverage = math.ceil(slice_coverage)
    if slice_coverage % 2 == 0:
        slice_coverage = slice_coverage - 1

    return slice_coverage
