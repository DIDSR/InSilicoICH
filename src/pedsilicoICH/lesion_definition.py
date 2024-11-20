"""
Module responsible for lesion definition
"""
import math

import numpy as np
import skimage as ski
import scipy
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


def insert_dural_3D(phantom, desired_volume, hematoma_type,
                    mass_effect, seed=None):

    random = np.random.default_rng(seed)

    num_slices = coverage_from_volume(volume=desired_volume,
                                      hematoma_type=hematoma_type,
                                      slice_thickness=phantom.dz)
    ab = (desired_volume*2000)/(num_slices * phantom.dz)  # using ABC/2 formula (although /2000 for mL and mm)
    # print('AB: ' + str(ab))
    if hematoma_type == 'epidural':
        desired_distance = math.sqrt(4*ab)  # assume that length of epidural hemorrhage is about 4 times the width
    elif hematoma_type == 'subdural':
        desired_distance = math.sqrt(10*ab)

    HU_array = phantom.get_CT_number_phantom()

    # TODO: better logic for hemorrhage starting slice
    init_slice = int(random.choice(np.linspace(0, int(HU_array.shape[0]/3), int(HU_array.shape[0]/3) + 1)))

    dura_map = phantom.get_dura_map()
    skull_map = phantom.get_skull_map()

    new_volume = np.copy(HU_array)

    boundary = dura_map
    distances = [(desired_distance-5)/phantom.spacings[2], (desired_distance+5)/phantom.spacings[2]]

    slice_counter = slice_idx = 0  # will iterate on this
    iter_flag = True

    # desired_thickness = 0.5 # slice thickness in mm
    hemisphere = random.choice(['left', 'right'])  # can either be random or pre-defined

    if hemisphere == 'left':
        boundary[:, :, (int(HU_array.shape[2]/2) - 10):None] = 0.0
    elif hemisphere == 'right':
        boundary[:, :, :(int(HU_array.shape[2]/2) + 10)] = 0.0

    hemorrhage_mask = np.zeros_like(boundary)
    while iter_flag:
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

                    orig_start = start_point
                    orig_end = end_point
                except:
                    count += 1
                    init_slice = int(random.choice(np.linspace(0, int(HU_array.shape[0]/3), int(HU_array.shape[0]/3) + 1)))
                    if count == tol:
                        failure_occured = True
                else:
                    count = tol
            if failure_occured:
                raise RuntimeError(f'lesion insertion failed with requested volume: {desired_volume} mL, try a smaller volume')
            # now that the two starting points for the hemorrhage have been defined, need to connect them
            # process should be the same on any given slice, and this function can be updated with new connection options
            filled_array, boundary_coords, connect_coords =\
                connect_points(start=orig_start,
                               end=orig_end,
                               boundary=temp_boundary,
                               hematoma_type=hematoma_type)

            mass_effect = True
            if mass_effect:
                try:
                    warped_slice = warp_slice(HU_array[init_slice, :],
                                              skull_map[init_slice, :],
                                              boundary_coords, connect_coords, filled_array)
                except ValueError:
                    Warning(f'Failed to perform mass effect insertion for\
                          volume: {desired_volume}, now inserting with mass\
                          effect to 0')
                    new_volume[init_slice] = HU_array[init_slice]
                    phantom.mass_effect = 0
                new_volume[init_slice, :, :] = warped_slice

            hemorrhage_mask[init_slice, :, :] = filled_array

            print(init_slice)
            slice_counter += 1
            iter_flag = False

            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(warped_slice, vmin=-40, vmax=120)
            plt.title('warped slice (no hematoma)')
            plt.show()

            plt.figure()
            plt.imshow(filled_array)
            plt.title('hematoma mask')
            plt.show()

        # # this is a bit messy but it will work for intended purpose:
        # # starting from hemorrhage origin, move down while shrinking distance between start/end point
        # # then, move up from hemorrhage origin while doing the same
        # if slice_counter <= (num_slices-1)/2:  # move down from hemorrhage origin
        #     slice_idx = slice_counter
        # elif slice_counter > (num_slices-1)/2:  # start moving up from hemorrhage origin
        #     slice_idx = -1*(slice_counter - int((num_slices-1)/2))

        # temp_boundary = boundary[init_slice-slice_idx]

        # dura_idx = np.argwhere(temp_boundary == 1.0)

        # if len(dura_idx) != 0:  # need to check that we didn't land on a slice with no remaining dura
        #     # find closest boundary point to previous start
        #     distance_idx = np.zeros((len(dura_idx), 2))

        #     distance_from_start = np.zeros(len(dura_idx))
        #     distance_from_end = np.zeros(len(dura_idx))
        #     for i in range(len(dura_idx)):
        #         distance_from_start[i] = math.sqrt((orig_start[0] - dura_idx[i][0])**2 + (orig_start[1] - dura_idx[i][1])**2)
        #         distance_from_end[i] = math.sqrt((orig_end[0] - dura_idx[i][0])**2 + (orig_end[1] - dura_idx[i][1])**2)

        #     new_start = dura_idx[np.argmin(distance_from_start)]
        #     new_end = dura_idx[np.argmin(distance_from_end)]

        #     filled_array, boundary_coords, connect_coords =\
        #         connect_points(start=new_start,
        #                        end=new_end,
        #                        boundary=temp_boundary,
        #                        hematoma_type=hematoma_type)
        #     if mass_effect:
        #         print('WARPING ' + str(init_slice-slice_idx))
        #         try:
        #             warped_slice = warp_slice(HU_array[init_slice-slice_idx, :],
        #                                       skull_map[init_slice-slice_idx, :],
        #                                       boundary_coords, connect_coords, filled_array)
        #             new_volume[init_slice-slice_idx] = warped_slice
        #         except ValueError:
        #             Warning(f'Failed to perform mass effect insertion for\
        #                   volume: {desired_volume}, now inserting with mass\
        #                   effect to 0')
        #             new_volume[init_slice-slice_idx] =\
        #                 HU_array[init_slice-slice_idx]
        #             phantom.mass_effect = 0

        #     hemorrhage_mask[init_slice-slice_idx] = filled_array

        #     slice_counter += 1

        #     if slice_counter == num_slices:
        #         iter_flag = False

        # else:
        #     slice_counter += 1
        #     if slice_counter == num_slices:
        #         iter_flag = False
                
    return hemorrhage_mask.astype(bool), new_volume


def connect_points(start, end, boundary, hematoma_type):
    '''draw a line connecting start and end points but following existing dura'''
    rows, cols = boundary.shape
    costs = np.where(boundary, 0, 10000)
    path, _ = ski.graph.route_through_array(costs, start=(start[0], start[1]), end=(end[0], end[1]), fully_connected=False)

    indices = np.stack(path, axis=-1)
    boundary_route = np.zeros_like(boundary)
    boundary_route[indices[0], indices[1]] = 1.0

    boundary_coords = np.zeros((len(path), 2))  # we want to take the boundary path and save to numpy array for later
    for idx, coord in enumerate(path):
        boundary_coords[idx, :] = np.array(coord)

    # START Bezier curve (could change this to separate function
    # if we add more methods to connect points)
    if hematoma_type == 'epidural':
        bezier_weight = 0.14  # weight should probably be below 0.2 to avoid ballooning too much, but line breaks if below 0.04....
        bezier_middle = (int(rows/2), int(cols/2))  # center of the image, should probably randomize it somewhere along center later
    elif hematoma_type == 'subdural':
        bezier_weight = 2
        bezier_middle = boundary_coords[round(len(boundary_coords)/2)]  # use the middle point of the dura line
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

    # returned coordinate list isn't ordered from start point to end point
    # below is rudimentary but should order from start point to end point as long as weight < 1
    connect_coords = connect_coords.tolist()
    connect_coords.sort(key=lambda p: math.dist(p, [start[0], start[1]]))
    connect_coords = np.array(connect_coords)
    # END Bezier curve

    filled_array = scipy.ndimage.binary_fill_holes(np.where(np.add(connecting_route, boundary_route) > 0, 1.0, 0)).astype(int)

    return filled_array, boundary_coords, connect_coords


def warp_slice(axial_slice, skull_slice, src, dst, filled_boundary):
    '''perform warp of 2D slice according to hematoma boundary coordinates'''
    # to simulate mass effect, transform will need some skull coordinates to NOT move
    #skull_slice = skull_slice.astype(bool)

    import matplotlib.pyplot as plt
    import sys

    # plt.figure()
    # plt.imshow(skull_slice)
    # plt.title('skull_mask')
    # plt.show()
    flood_mask = ski.segmentation.flood(skull_slice, seed_point=(0, 0))

    plt.figure()
    plt.imshow(flood_mask)
    plt.title('flood mask')
    plt.show()

    skull_slice[flood_mask] = 1

    plt.figure()
    plt.imshow(skull_slice)
    plt.title('new skull slice')
    plt.show()

    brain_voxels = np.where(skull_slice != 1)
    print(brain_voxels)

    skull_idx = np.argwhere(skull_slice == 1.0)
    skull_boundary = ski.segmentation.find_boundaries(skull_slice, mode='inner', background=0)
    # plt.figure()
    # plt.imshow(skull_slice)
    # plt.title('flooded skull mask')
    # plt.show()
    # plt.figure()
    # plt.imshow(skull_boundary)
    # plt.show()
    # sys.exit()

    #skull_sample = np.round(np.linspace(0, len(skull_idx)-1, 500)).astype(int)  # increase from 1000 as memory allows
    skull_sample = np.argwhere(skull_boundary != 0)

    # initialize warp source and destination with skull indices in both (shouldn't move!)
    warp_src = warp_dst = skull_sample
    #warp_src = skull_idx[skull_sample]
    #warp_dst = skull_idx[skull_sample]

    # points_to_use = round(min(len(src), len(dst))/2)
    # print(points_to_use)
    src_subset = src[np.round(np.linspace(0, len(src)-1, 20)).astype(int)]
    dst_subset = dst[np.round(np.linspace(0, len(dst)-1, 20)).astype(int)]

    import matplotlib.pyplot as plt
    test = np.zeros_like(axial_slice)
    plt.imshow(filled_boundary)
    for index in src_subset:
        plt.scatter(x=index[1], y=index[0], marker='x', color='green', s=1)
    for index in dst_subset:
        plt.scatter(x=index[1], y=index[0], marker='x', color='red', s=1)
    for index in warp_src:
        plt.scatter(x=index[1], y=index[0], marker='x', color='yellow', s=1)
    plt.show()

    # sys.exit()

    warp_src = np.insert(warp_src, 0, src_subset, axis=0)
    warp_dst = np.insert(warp_dst, 0, dst_subset, axis=0)

    tps = ski.transform.ThinPlateSplineTransform()
    tps.estimate(np.flip(warp_dst), np.flip(warp_src))
    warped_slice = ski.transform.warp(axial_slice, tps,
                                      preserve_range=True, order=1)
    
    # new code to try to "fix" skull warping into brain
    problem_voxels = np.where((flood_mask != 1) & (warped_slice > 400))
    print('problem voxels:')
    print(len(problem_voxels[0]))
    print(len(problem_voxels[1]))

    #for index in problem_voxels:

    # brain_mask = (~skull_slice) & (axial_slice > -100)
    # error_map = (warped_slice - axial_slice) > axial_slice[brain_mask].mean()
    # warped_slice[error_map] = axial_slice[error_map]

    return warped_slice


def coverage_from_volume(volume, hematoma_type, slice_thickness):  # see RSNA_BHDS_explore.ipynb for logarithmic fit
    if hematoma_type == 'epidural':
        slice_coverage = 13.942*math.log(volume) + 13.449
    elif hematoma_type == 'subdural':
        slice_coverage = 17.739*math.log(volume) + 17.314
    elif hematoma_type == 'sphere':
        slice_coverage = 8.7064*math.log(volume) + 18.148  # for now, this is intraparenchymal
    # unused
    elif hematoma_type == 'subarachnoid':
        slice_coverage = 17.181*math.log(volume) + 27.42
    elif hematoma_type == 'intraventricular':
        slice_coverage = 11.341*math.log(volume) + 25.435
    # convert units from mm to number of slices
    slice_coverage = slice_coverage / slice_thickness

    # round to nearest odd number
    slice_coverage = math.ceil(slice_coverage)
    if slice_coverage % 2 == 0:
        slice_coverage = slice_coverage - 1

    return slice_coverage
