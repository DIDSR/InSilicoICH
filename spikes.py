# %%
# https://github.com/DIDSR/MISS-tool/blob/main/hat_rho.m
# https://github.com/DIDSR/MISS-tool/blob/main/mod_spike.m

from pedsilicoICH.ground_truth_definition.phantoms import (load_phantom,
                                                           get_transformation_src_dst,
                                                           insert_with_mass_effect)
from pedsilicoICH.lesion_definition import warp_slice
from notebooks.utils import ctshow
import matplotlib.pyplot as plt
import numpy as np

# %%
phantom = load_phantom(6.5)
img_w_lesion, lesion_vol, (z, x, y) =\
    phantom.add_round_lesion(volume=10, intensity=100,
                             eccentricity=0.9, seed=42)
strength = 0.9

src, dst = get_transformation_src_dst(lesion_vol[z], strength)
if strength > 0:
    dst_coords = np.argwhere(dst)
    src_coords = np.argwhere(src)
    warped = warp_slice(phantom.get_CT_number_phantom()[z],
                        phantom.get_skull_map()[z],
                        src_coords, dst_coords)
else:
    warped = img_w_lesion[z].copy()

warped[lesion_vol[z]] = img_w_lesion[z][lesion_vol[z]]
f, axs = plt.subplots(1, 2, dpi=150)
ctshow(img_w_lesion[z], 'brain', fig=f, ax=axs[0])
axs[0].imshow(src, alpha=0.2, cmap='Reds')
axs[0].set_title('src')

ctshow(warped, 'brain', fig=f, ax=axs[1])
axs[1].imshow(src, alpha=0.2, cmap='Reds')
axs[1].imshow(dst, alpha=0.2, cmap='Reds')
axs[1].set_title(f'dst, strength: {strength}')
plt.show()

