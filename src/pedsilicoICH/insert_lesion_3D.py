from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import sys
import math
import skimage as ski
import scipy
import cv2
import time

# Jayse Weaver 08/26/24
# Currently relies on being called by main.py
#	(home/jayse.weaver/the_lab/lesion_simulator/main.py)
# or can be called in Notebook #3
# Only inserts epidural/subdural lesions, will be merged with other lesion types

# the goal is to re-write this script as a function that can work on any atlas/phantom
def insert_dural_3D(header, volume, boundary, skull, init_slice, hematoma_type, verbose, plot_opt):

    print('starting code')

    start_t = time.perf_counter()

    # options and initial set up
    plot_intermediate = False # true if plotting intermediate steps is desired
    plot_final = plot_opt
    save_output = True
    [dx, dy, dz] = header['pixdim'][1:4] # set resolution, mm
    distances = [50/dx, 75/dx] # length range in mm, will divide by voxel size (assuming isotropic) to convert

    num_slices = 11 # number of slices to have hemorrhage, try to make this odd
    slice_counter = 0 # will iterate on this
    iter_flag = True
    rows, cols, slices = volume.shape

    desired_thickness = 0.5 # slice thickness in mm
    hemisphere = random.choice(['left', 'right']) # can either be random or pre-defined

    if verbose: print("Creating synthetic " + hematoma_type + " hemorrhage in " + hemisphere + " hemisphere")

    # create an all-yellow colormap to use for hemorrhage "segmentation" (make sure mask values are 1)
    yellow, norm = mpl.colors.from_levels_and_colors(levels=[0, 1], colors=['blue', 'yellow'], extend='max')
    red, norm = mpl.colors.from_levels_and_colors(levels=[0, 1], colors=['blue', 'red'], extend='max')

    ## FIX THIS LATER
    # # select a single axial slice to speed up computations
    # if desired_thickness != dz:
    #     volume = volume[:, :, ::(round(desired_thickness/dz))]
    #     boundary = boundary[:, :, ::(round(desired_thickness/dz))]
    #     skull = skull[:, :, ::(round(desired_thickness/dz))]

    # if desired_thickness == round(dz, 1): # need to round, MIDA phantom header lists dz as 0.4999
    #     axial_slice = volume[:, :, init_slice]
    #     dura_map = boundary[:, :, init_slice]
    #     skull_map = skull[:, :, init_slice]
    # elif desired_thickness == 5:
    #     axial_slice = volume[:, :, int(init_slice/(desired_thickness/dz))]
    #     dura_map = boundary[:, :, int(init_slice/(desired_thickness/dz))]
    #     skull_map = skull[:, :, int(init_slice/(desired_thickness/dz))]

    if hemisphere == 'left':
        boundary[:, (int(cols/2)-10):None, :] = 0.0
    elif hemisphere == 'right':
        boundary[:, :(int(cols/2)+10), :] = 0.0

    hemorrhage_mask = np.zeros_like(boundary)
    while iter_flag:
        print('iterating')
        if slice_counter == 0: # need to do the slice in the middle of the hemorrhage, same as before
            print('Processing center slice: ' + str(init_slice))
            temp_boundary = boundary[:, :, init_slice]
            dura_idx = np.argwhere(temp_boundary == 1.0)

            # choose a random start point, and calculate distance bfrom all available boundary voxels to start point
            start_point = dura_idx[random.choice(range(len(dura_idx)))]
            print(start_point)

            distance_idx = np.zeros(len(dura_idx))
            for i in range(len(dura_idx)):
                distance_idx[i] = math.sqrt((start_point[0] - dura_idx[i][0])**2 + (start_point[1] - dura_idx[i][1])**2)

            # create list of possible end points and choose one at random
            close_voxel_list = np.where(np.logical_and(distance_idx > distances[0], distance_idx < distances[1]))
            end_point = dura_idx[random.choice(close_voxel_list[0])]

            orig_start = start_point
            orig_end = end_point

            orig_dist = math.sqrt((orig_start[0] - orig_end[0])**2 + (orig_start[1] - orig_end[1])**2)

            # now that the two starting points for the hemorrhage have been defined, need to connect them
            # process should be the same on any given slice, and this function can be updated with new connection options
            filled_array, boundary_coords, connect_coords = connect_points(start=orig_start, end=orig_end, boundary=temp_boundary, hematoma_type=hematoma_type)
            # eventually will use boundary and connection coordinates to warp hemorrhage volume and original phantom

            hemorrhage_mask[:, :, init_slice] = filled_array

            slice_counter += 1

        if slice_counter <= (num_slices-1)/2:
            # now do other slices
            temp_boundary = boundary[:, :, init_slice-slice_counter]
            print(temp_boundary.shape)
            dura_idx = np.argwhere(temp_boundary == 1.0)
            print('Processing slice: ' + str(init_slice - slice_counter))
            # find closest boundary point to previous start
            distance_idx = np.zeros((len(dura_idx),2))

            # plt.figure(figsize=(10,10))
            # plt.imshow(dura_route)
            # plt.scatter(orig_start[0], orig_start[1], s=100, c='green', marker='x') 
            # plt.scatter(orig_end[0], orig_end[1], s=100, c='red', marker='x')
            # plt.imshow(np.ma.masked_where(temp_boundary != 1.0, temp_boundary), cmap=yellow, norm=norm, alpha=0.5)
            # plt.show()

            distance_from_start = np.zeros(len(dura_idx))
            distance_from_end = np.zeros(len(dura_idx))
            for i in range(len(dura_idx)):
                distance_from_start[i] = math.sqrt((orig_start[0] - dura_idx[i][0])**2 + (orig_start[1] - dura_idx[i][1])**2)
                distance_from_end[i] = math.sqrt((orig_end[0] - dura_idx[i][0])**2 + (orig_end[1] - dura_idx[i][1])**2)

            new_start = dura_idx[np.argmin(distance_from_start)]
            new_end = dura_idx[np.argmin(distance_from_end)]

            filled_array, boundary_coords, connect_coords = connect_points(start=orig_start, end=orig_end, boundary=temp_boundary, hematoma_type=hematoma_type)

            hemorrhage_mask[:, :, init_slice-slice_counter] = filled_array

            slice_counter += 1

            if slice_counter == num_slices:
                iter_flag = False

        elif slice_counter > (num_slices-1)/2:
            # now do slices above origin
            temp_boundary = boundary[:, :, init_slice-int((num_slices-1)/2)+slice_counter]
            dura_idx = np.argwhere(temp_boundary == 1.0)
            print('Processing slice: ' + str(init_slice - slice_counter))
            # find closest boundary point to previous start
            distance_idx = np.zeros((len(dura_idx),2))

            # plt.figure(figsize=(10,10))
            # plt.imshow(dura_route)
            # plt.scatter(orig_start[0], orig_start[1], s=100, c='green', marker='x') 
            # plt.scatter(orig_end[0], orig_end[1], s=100, c='red', marker='x')
            # plt.imshow(np.ma.masked_where(temp_boundary != 1.0, temp_boundary), cmap=yellow, norm=norm, alpha=0.5)
            # plt.show()

            distance_from_start = np.zeros(len(dura_idx))
            distance_from_end = np.zeros(len(dura_idx))
            for i in range(len(dura_idx)):
                distance_from_start[i] = math.sqrt((orig_start[0] - dura_idx[i][0])**2 + (orig_start[1] - dura_idx[i][1])**2)
                distance_from_end[i] = math.sqrt((orig_end[0] - dura_idx[i][0])**2 + (orig_end[1] - dura_idx[i][1])**2)

            new_start = dura_idx[np.argmin(distance_from_start)]
            new_end = dura_idx[np.argmin(distance_from_end)]

            filled_array, boundary_coords, connect_coords = connect_points(start=orig_start, end=orig_end, boundary=temp_boundary, hematoma_type=hematoma_type)

            hemorrhage_mask[:, :, init_slice-int((num_slices-1)/2)+slice_counter] = filled_array
            slice_counter += 1

            if slice_counter == num_slices:
                iter_flag = False

    print('no longer iterating')
    
    return hemorrhage_mask


    # OLD CODE LEAVE FOR COPYING
    # skull_tissue_idx = np.argwhere(skull_map/skull_map == 1.0)

    # # need to subsample the number of skull voxels to use in the warping for performance
    # # more than 2000 on admin laptop causes lag
    # # once on the servers I think we can use every skull voxel as an immovable point!
    # skull_sample = np.round(np.linspace(0, len(skull_tissue_idx)-1, 1000)).astype(int)
    # skull_subset = skull_tissue_idx[skull_sample]
    # #skull_subset = skull_tissue_idx

    # if plot_intermediate:
    #     plt.figure()
    #     plt.imshow(axial_slice, cmap='gray')
    #     plt.imshow(np.ma.masked_where(dura_map != 1.0, dura_map), cmap=yellow, norm=norm, alpha=1.0)
    #     plt.title('Dura Segmentation')
    #     plt.show()

    # # we want to avoid the dura in the longitudinal fissure for the start and end points of the EDH 
    # # also, for simplicity, we want the EDH to only be in one hemisphere
    # # eventually can update this with quadrants or specific lobes



    # if plot_intermediate:
    #     plt.figure()
    #     plt.imshow(axial_slice, cmap='gray')
    #     plt.imshow(np.ma.masked_where(dura_map != 1.0, dura_map), cmap=yellow, norm=norm, alpha=1.0)
    #     plt.title('Hemisphere Selection')
    #     plt.show()

    # # calculate list of available dura voxels
    # dura_idx = np.argwhere(dura_map == 1.0)

    # # choose a random start point, and calculate distance from all available dura voxels to start point
    # start_point = random.choice(range(len(dura_idx)))
    # distance_idx = np.zeros(len(dura_idx))
    # for i in range(len(dura_idx)):
    #     distance_idx[i] = math.sqrt((dura_idx[start_point][1] - dura_idx[i][1])**2 + (dura_idx[start_point][0] - dura_idx[i][0])**2)

    # # create list of possible end points
    # close_voxel_list = np.where(np.logical_and(distance_idx > distances[0], distance_idx < distances[1]))
    # end_point = random.choice(close_voxel_list[0]) # chose end point

    # start = [dura_idx[start_point][1], dura_idx[start_point][0]] # column, row
    # end = [dura_idx[end_point][1], dura_idx[end_point][0]]

    # # draw a line connecting start and end points but following existing dura
    # costs = np.where(dura_map, 0, 10000)
    # path, cost = ski.graph.route_through_array(costs, start=(dura_idx[start_point][0], dura_idx[start_point][1]),
    #                                         end=(dura_idx[end_point][0], dura_idx[end_point][1]), fully_connected=True)

    # indices = np.stack(path, axis=-1)
    # dura_route = np.zeros_like(axial_slice)
    # dura_route[indices[0], indices[1]] = 1.0

    # # plt.figure()
    # # plt.imshow(axial_slice, cmap='gray')
    # # plt.imshow(np.ma.masked_where(dura_route != 1.0, dura_route), cmap=yellow, norm=norm, alpha=0.5)
    # # plt.show()

    # path_coords = np.zeros((len(path), 2))
    # for idx, coord in enumerate(path):
    #     path_coords[idx, :] = np.array(coord)

    # # create straight line directly connecting start and end points
    # # rr, cc = ski.draw.line(start[1], start[0], end[1], end[0]) # old code to just draw straight line, leaving for posterity
    # # connecting_line = np.zeros_like(axial_slice)
    # # connecting_line[rr, cc] = 1.0
    # # line_coords = np.stack((rr, cc), axis=-1)

    # if hematoma_type == 'epidural':
    #     bezier_weight = 0.1 # weight should probably be below 0.5 to avoid ballooning too much
    #     bezier_middle = (int(rows/2), int(cols/2)) # center of the image, should probably randomize it somewhere along center later
    # elif hematoma_type == 'subdural':
    #     bezier_weight = 0.5
    #     bezier_middle = path_coords[round(len(path_coords)/2)] # use the middle point of the dura line 
    #     print(bezier_middle)
    # else:
    #     bezier_weight = 0.0
    #     bezier_middle = (int(rows/2), int(cols/2))


    # rr, cc = ski.draw.bezier_curve(r0=start[1], c0=start[0], 
    #                         r1=int(bezier_middle[0]), c1=int(bezier_middle[1]),
    #                         r2=end[1], c2=end[0],
    #                         weight=bezier_weight)

    # connecting_line = np.zeros_like(axial_slice)
    # connecting_line[rr, cc] = 1.0
    # line_coords = np.stack((rr, cc), axis=-1)

    # # the bezier curve coordinate list isn't ordered from start point to end point
    # # below is rudimentary but should order from start point to end point as long as weight < 1
    # line_coords = line_coords.tolist()
    # line_coords.sort(key=lambda p: math.dist(p, [start[1], start[0]]))
    # line_coords = np.array(line_coords)

    # # get some coordinates from the dura path and drawn line to create affine transform
    # line_sample = np.round(np.linspace(0, len(line_coords)-1, 15)).astype(int)
    # line_subset = line_coords[line_sample]

    # path_sample = np.round(np.linspace(0, len(path_coords)-1, 15)).astype(int)
    # path_subset = path_coords[path_sample]

    # # TEMPORARY PLOT CODE TO VISUALIZE ORDER OF PATH POINTS
    # # plt.figure()
    # # plt.imshow(axial_slice, cmap='gray')
    # # plt.scatter(dura_idx[start_point][1], dura_idx[start_point][0], s=100, c='green', marker='x') 
    # # plt.scatter(dura_idx[end_point][1], dura_idx[end_point][0], s=100, c='red', marker='x')
    # # for i in range(len(path_subset)):
    # #     plt.scatter(path_subset[i][1], path_subset[i][0], s=100, c='green', marker='x')
    # #     plt.pause(0.01)

    # # for i in range(len(line_subset)):
    # #     plt.scatter(line_subset[i][1], line_subset[i][0], s=100, c='red', marker='x')
    # #     plt.pause(0.01)

    # #plt.show()

    # path_subset = np.insert(path_subset, 0, skull_subset, axis=0)
    # src = np.insert(path_subset, 0, [[0, 0], [rows,cols], [0, cols], [rows, 0]], axis=0)
    # line_subset = np.insert(line_subset, 0, skull_subset, axis=0)
    # dst = np.insert(line_subset, 0, [[0, 0], [rows,cols], [0, cols], [rows, 0]], axis=0)

    # if verbose: print("Estimating transform and warping")
    # start_warp = time.perf_counter()
    # tps = ski.transform.ThinPlateSplineTransform()
    # tps.estimate(np.flip(dst), np.flip(src))
    # dst = ski.transform.warp(axial_slice, tps, preserve_range=False, order=0)
    # end_warp = time.perf_counter()
    # if verbose: print("Transform and warp took", str(round(end_warp - start_warp, 3)), "seconds")

    # # combine the two lines into a binary mask and fill holes
    # img_fill_holes = scipy.ndimage.binary_fill_holes(np.where(np.add(connecting_line, dura_route) > 0, 1.0, 0)).astype(int)
    # img_fill_holes[rr, cc] = 0

    # end_t = time.perf_counter()

    # if verbose: print("Total time elapsed:", str(round(end_t - start_t, 3)), "seconds")

    # new_slice = np.copy(axial_slice)
    # new_slice[np.where(img_fill_holes == 1)] = 117 # make the interior of the hemorrhage index 117 (unused by atlas)
    # new_slice[rr, cc] = 1 # make the outline of the hemmorrhage dura

    # dst_new = np.round(np.copy(dst))
    # dst_new[np.where(img_fill_holes == 1)] = 117
    # if hematoma_type == 'epidural':
    #     dst_new[indices[0], indices[1]] = 117 # if epidural, make the old dura pixels be hemorrhage
    #     dst_new[rr, cc] = 1 # and the path route dura
    # elif hematoma_type == 'subdural':
    #     dst_new[indices[0], indices[1]] = 1.0 # if subdural, keep old dura pixels as dura
    #     dst_new[rr, cc] = 117 # and make the path part of the hemorrhage

    # final_hemorrhage = np.where(dst_new == 117)
    # volume_ml = (np.count_nonzero(final_hemorrhage)*(dx*dy*desired_thickness))*0.001  

    # if verbose: print('Hemorrhage volume: ', str(volume_ml), ' mL')

    # if plot_final:
    #     plt.figure()
    #     plt.imshow(axial_slice, cmap='gray')
    #     plt.scatter(dura_idx[start_point][1], dura_idx[start_point][0], s=100, c='green', marker='x') 
    #     plt.scatter(dura_idx[end_point][1], dura_idx[end_point][0], s=100, c='red', marker='x')
    #     plt.imshow(np.ma.masked_where(img_fill_holes != 1.0, img_fill_holes), cmap=yellow, norm=norm, alpha=0.5)
    #     plt.title('Original slice with hemorrhage mask')

    #     plt.figure()
    #     plt.subplot(121)
    #     plt.imshow(axial_slice, cmap='gray', vmin=0, vmax=117)
    #     plt.imshow(np.ma.masked_where(img_fill_holes != 1.0, img_fill_holes), cmap=yellow, norm=norm, alpha=0.5)
    #     plt.title('Input')

    #     plt.subplot(122)
    #     plt.imshow(dst_new, cmap='gray', vmin=0, vmax=117)
    #     plt.title('Output No Mask')

    #     plt.show()

    # new_volume = np.copy(volume)
    # if desired_thickness == 0.5:
    #     new_volume[:, :, 350] = dst_new
    # elif desired_thickness == 5:
    #     new_volume[:, :, 35] = dst_new

def connect_points(start, end, boundary, hematoma_type):
    # draw a line connecting start and end points but following existing dura
    rows, cols = boundary.shape
    costs = np.where(boundary, 0, 10000)
    path, cost = ski.graph.route_through_array(costs, start=(start[0], start[1]), end=(end[0], end[1]), fully_connected=True)

    indices = np.stack(path, axis=-1)
    boundary_route = np.zeros_like(boundary)
    boundary_route[indices[0], indices[1]] = 1.0

    boundary_coords = np.zeros((len(path), 2)) # we want to take the boundary path and save to numpy array for later
    for idx, coord in enumerate(path):
        boundary_coords[idx, :] = np.array(coord)

    # now it's time to add the bezier curve
    if hematoma_type == 'epidural':
        bezier_weight = 0.1 # weight should probably be below 0.5 to avoid ballooning too much
        bezier_middle = (int(rows/2), int(cols/2)) # center of the image, should probably randomize it somewhere along center later
    elif hematoma_type == 'subdural':
        bezier_weight = 0.5
        bezier_middle = boundary_coords[round(len(boundary_coords)/2)] # use the middle point of the dura line 
        print(bezier_middle)
    else:
        bezier_weight = 0.0
        bezier_middle = (int(rows/2), int(cols/2))


    rr, cc = ski.draw.bezier_curve(r0=start[0], c0=start[1], 
                            r1=int(bezier_middle[0]), c1=int(bezier_middle[1]),
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




