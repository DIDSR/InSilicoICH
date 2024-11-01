# %%
from pedsilicoICH.ground_truth_definition.phantoms import load_phantom
from notebooks.utils import ctshow, show_lesions
age = 6.5
desired_volume = 5 # in mL
complexity = 1
edema = 0
intensity = 100
eccentricity = 0
seed = 42
phantom = load_phantom(age=age)
phantom.insert_lesion('round', volume=desired_volume, intensity=intensity,
                      edema=edema, eccentricity=eccentricity,
                      complexity=complexity, seed=seed)
show_lesions(phantom)
# %%
intensity = 100
small_phantom = load_phantom(age=age, shape=(128, 128, 100))
small_phantom.insert_lesion('round', volume=desired_volume,
                            intensity=intensity, edema=edema,
                            eccentricity=eccentricity,
                            complexity=complexity, seed=seed)
show_lesions(small_phantom)
# %%
# %%the two phantoms should make lesions of the same volume in mL
# I dont think the lesion_insertion accounts for voxel sizes
print(phantom.shape, small_phantom.shape)
assert (phantom.size == small_phantom.size).all()
assert phantom.shape > small_phantom.shape
assert phantom.dx < small_phantom.dx
assert phantom.dz < small_phantom.dz

# %%
def get_measured_vol(phantom):
    'in mL'
    mask = phantom.get_lesion_mask()
    return mask.sum() * phantom.dx*phantom.dy*phantom.dz / 1000

get_measured_vol(phantom), get_measured_vol(small_phantom)
# %%
