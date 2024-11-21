# %%
from pathlib import Path
from pedsilicoICH.pipeline import Study, Scanner, load_phantom
import matplotlib.pyplot as plt
import numpy as np
from utils import ctshow, center_crop

age = 15.75
intensity = 50
volume = 0.1
mass_effect = 1.0

# %% kernel
kernels = ['standard', 'soft', 'bone']
outdir = Path('experiments/kernel')
outdir.mkdir(exist_ok=True)

if age == 38:
    startZ = 40
    endZ = 50
else:
    startZ = 0
    endZ = 10

phantom = load_phantom(age, name='acquisition settings')
phantom.insert_lesion('round', volume=volume,
                      intensity=intensity,
                      mass_effect=mass_effect,
                      seed=336)
lesion_level_mm = (phantom.get_CT_number_phantom().shape[0]/2 -
                   phantom._lesion_coords[0][0])*phantom.dz
center = lesion_level_mm
width = 8

scanner = Scanner(phantom, output_dir=outdir)
scanner.run_scan(startZ=center-width//2, endZ=center+width//2,
                 views=1000)
for kernel in kernels:
    scanner.run_recon(kernel=kernel)
    ctshow(center_crop(scanner.recon[1:-1].mean(axis=0)), 'brain')
    plt.savefig(outdir / f'{kernel}_kernel.png', dpi=600)

# %% mAs, kVp
mAs = [40, 160, 640]
kVps = [80, 100, 120, 140]
series_dict = {'mAs': mAs,
               'kVp': kVps}
for series in series_dict:
    outdir = Path('experiments') / series
    outdir.mkdir(exist_ok=True, parents=True)
    param_vector = series_dict[series]
    for param in param_vector:
        if series == 'mAs':
            mA = param
            kVp = 120
            kernel = 'soft'
        elif series == 'kVp':
            mA = (120/param)**2 * 200
            kVp = param
            kernel = 'soft'
        else:
            raise ValueError(f'{series} not mAs or kVp')

        scanner = Scanner(phantom, output_dir=outdir)
        study = Study(scanner, series)
        study.run_study(series, mA=mA, kVp=kVp,
                        zspan=(center-width//2, center+width//2),
                        views=1000, kernel=kernel)
        ctshow(study.images[1:-1].mean(axis=0), 'brain')
        plt.savefig(outdir / f'{mA}mA_{kVp}kVp_{kernel}.png',
                    dpi=600, bbox_inches='tight')
