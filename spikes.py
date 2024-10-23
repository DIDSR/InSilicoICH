# %%
# https://github.com/DIDSR/MISS-tool/blob/main/hat_rho.m
# https://github.com/DIDSR/MISS-tool/blob/main/mod_spike.m

from pedsilicoICH.ground_truth_definition.phantoms import load_phantom
from pedsilicoICH.lesion_definition import warp_slice, connect_points
from notebooks.utils import ctshow
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import white_tophat
# %%
# phantom = load_phantom(6.5)
# phantom.insert_lesion('epidural', volume=10, mass_effect=True)
# # %%
# ctshow(phantom.get_CT_number_phantom()[phantom._lesion_coords[0][0]], 'brain')
# %%
phantom = load_phantom(6.5)
phantom.insert_lesion('round', volume=10, eccentricity=0, seed=42)
ctshow(phantom.get_CT_number_phantom()[phantom._lesion_coords[0][0]])
# %%
HU_array = phantom.get_CT_number_phantom()[phantom._lesion_coords[0][0]]
skull_map = phantom.get_skull_map()[phantom._lesion_coords[0][0]]
lesion = phantom.get_lesion_mask()[phantom._lesion_coords[0][0]]
# %%
f, axs = plt.subplots(1,3)
ctshow(HU_array, fig=f, ax=axs[0])
axs[1].imshow(skull_map)
axs[2].imshow(lesion)
# %%
# warped = warp_slice(HU_array, skull_map, src, dst)