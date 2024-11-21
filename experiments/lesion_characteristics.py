from pathlib import Path
from pedsilicoICH.pipeline import Scanner, load_phantom
import matplotlib.pyplot as plt
import numpy as np
from utils import ctshow, center_crop

age = 15.75

if age == 38:
    startZ = 40
    endZ = 50
else:
    startZ = 0
    endZ = 10

# lesion_types = ['round', 'epidural', 'subdural']
lesion_types = ['epidural', 'subdural']
mass_effect = 1.0

outdir = Path('experiments/lesions')
kernel = 'soft'
outdir.mkdir(exist_ok=True, parents=True)

for lesion_type in lesion_types:
    volumes = np.linspace(0.1, 8, 3) if lesion_type == 'round'\
        else np.linspace(1, 20, 3)
    intensities = np.linspace(70, 50, 3)

    for intensity, volume in zip(intensities, volumes):
        print(f'{intensity} HU, {volume} mL')
        phantom = load_phantom(age, name=lesion_type)
        phantom.insert_lesion(lesion_type=lesion_type, volume=volume,
                              intensity=intensity,
                              mass_effect=mass_effect, seed=336)
        lesion_level_mm = (phantom.get_CT_number_phantom().shape[0]/2 -
                           phantom._lesion_coords[0][0])*phantom.dz
        center = lesion_level_mm
        width = 8

        scanner = Scanner(phantom, output_dir=outdir)
        scanner.run_scan(startZ=center-width//2, endZ=center+width//2,
                         views=100)
        scanner.run_recon(kernel=kernel)
        ctshow(center_crop(scanner.recon[1:-1].mean(axis=0)), 'brain')
        plt.savefig(outdir / f'{volume}mL_{intensity}HU_{lesion_type}.png',
                    dpi=600, bbox_inches='tight')
