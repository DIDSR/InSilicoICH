# %%
import numpy as np
from insilicoICH.lesion_definition import FractureLesion
from VITools import get_available_phantoms
import matplotlib.pyplot as plt
# %%
phantom = get_available_phantoms()['1.0 yr UNC Head']()

fracture = FractureLesion('linear fracture', phantom.get_skull_map(), phantom.spacings)
fracture.generate(fracture_length=100)
phantom.insert_lesion(fracture)

ax=2
plt.imshow(phantom.get_skull_map().sum(axis=ax), cmap='bone')
plt.imshow(np.ma.masked_where(fracture.mask == 0,
                              fracture.mask).max(axis=ax), cmap='gist_rainbow', interpolation='none')
# %%
assert np.all(phantom.get_skull_map()[fracture.mask])
# %%

shape = (64, 128, 128) #  # phantom.shape #
spacings = (4, 2, 2) # #phantom.spacings #
# Create a simple spherical shell for the dura map
dura_map = np.zeros(shape, dtype=bool)
z, y, x = np.ogrid[-shape[0]//2:shape[0]//2, -shape[1]//2:shape[1]//2, -shape[2]//2:shape[2]//2]
shell_mask = (x**2 + y**2 + z**2 < 30**2) & (x**2 + y**2 + z**2 > 28**2)
dura_map[shell_mask] = True
fracture = FractureLesion('linear fracture', dura_map, spacings)
fracture.generate(fracture_length=100)

ax=2
plt.imshow(dura_map.sum(axis=ax), cmap='gray')
plt.imshow(np.ma.masked_where(fracture.mask == 0,
                              fracture.mask).max(axis=ax), cmap='gist_rainbow', interpolation='none')

assert np.all(dura_map[fracture.mask])

# %%
