# %%
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pedsilicoICH.ground_truth_definition.phantoms import (possible_ages,
                                                           load_phantom)
from pedsilicoICH.image_acquisition import Scanner

logging.basicConfig(filename="newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)

# %%
# https://radiopaedia.org/articles/windowing-ct?lang=us
display_settings = {
    'brain': (80, 40),
    'subdural': (300, 100),
    'stroke': (40, 40),
    'temporal bones': (2800, 600),
    'soft tissues': (400, 50),
    'lung': (1500, -600),
    'liver': (150, 30),
}


def ctshow(img, window='soft tissues', fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    # Define some specific window settings here
    if isinstance(window, str):
        if window not in display_settings:
            raise ValueError(f"{window} not in {display_settings}")
        ww, wl = display_settings[window]
    elif isinstance(window, tuple):
        ww = window[0]
        wl = window[1]
    else:
        ww = 6.0 * img.std()
        wl = img.mean()

    if img.ndim == 3:
        img = img[0].copy()

    ax.imshow(img, cmap='gray', vmin=wl-ww/2, vmax=wl+ww/2)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax.imshow(img, cmap='gray', vmin=wl-ww/2, vmax=wl+ww/2)


# %%
possible_ages
# %% lesions
lesion_types = [None, 'epidural', 'subdural', 'round']
mass_effect = True
intensities = np.linspace(0, 100)
seed = None

max_lesions = 100
lesions = []
rng = np.random.default_rng(seed=seed)
while len(lesions) < max_lesions:
    lesion = dict()
    lesion_type = rng.choice(lesion_types, size=1)[0]
    if lesion_type == 'round':
        lesion['volume'] = rng.choice(np.linspace(0.1, 8),
                                      size=1)[0]
    else:
        lesion['volume'] = rng.choice(np.linspace(1, 20),
                                      size=1)[0]
    lesion['intensity'] = rng.choice(intensities, size=1)[0]
    lesion['lesion_type'] = lesion_type
    lesion['mass_effect'] = mass_effect
    lesions.append(lesion)
lesions
# %% acquisition
max_scans = 100
scans = []

kVps = [80, 90, 100, 110, 120, 130, 140]
mAs = list(range(10, 1510, 10))
views = 100

ref_mA = 200
ref_kVp = 120

while len(scans) < max_scans:
    scan = dict(
        kVp=int(rng.choice(kVps, size=1)[0]),
        mA=int(rng.choice(mAs, size=1)[0]),
        views=views)
    scans.append(scan)
scans
# %% recon
kernels = ['soft', 'standard', 'bone']
slice_t = [1, 5]  # slice thickness in mm

recons = [dict(kernel=k, sliceThickness=s) for k in kernels for s in slice_t]
# %% prep results
outdir = Path(__file__).parent / 'images'
outdir.mkdir(exist_ok=True)
results = dict(uid=[], age=[], lesion_type=[], volume=[], attenuation=[],
               kVp=[], mA=[], kernel=[], slice_thickness=[], file=[], seed=[])
# %% loop
desired_iph = desired_control = desired_edh = desired_sdh = 25
n_control = n_iph = n_sdh = n_edh = 0

uid = 0
while (n_control < desired_control) & (n_iph < desired_iph) &\
      (n_edh < desired_edh) & (n_sdh < desired_sdh):
    print(f'{n_control}/{desired_control} control\n{n_iph}/{desired_iph} IPH\n\
{n_sdh}/{desired_sdh} SDH\n{n_edh}/{desired_edh} EDH')
    lesion = rng.choice(lesions, size=1)[0]
    scan = rng.choice(scans, size=1)[0]
    recon = rng.choice(recons, size=1)[0]
    age = float(rng.choice(possible_ages, size=1)[0])
    phantom = load_phantom(age)
    scan_seed = rng.uniform(0, 1e6, size=1)[0]
    fname = outdir / f"{uid:03d}_{age}_yrs_{lesion['lesion_type']}_{scan['kVp']}kV_{scan['mA']}mA.png"
# try:
    if lesion['lesion_type']:
        phantom.insert_lesion(**lesion, seed=scan_seed)
    scanner = Scanner(phantom)
    if lesion['lesion_type'] is None:
        center = rng.choice(scanner.calculate_start_positions()[1:-1],
                            size=1)[0]
    else:
        center = (phantom.get_CT_number_phantom().shape[0]/2 -
                  phantom._lesion_coords[0][0])*phantom.dz
    n_scans = 1
    scan_width = 7
    width = scan_width*n_scans
    scan['startZ'] = center - width / 2
    scan['endZ'] = center + width / 2

    scanner.run_scan(**scan)
    scanner.run_recon(**recon)
    if recon['sliceThickness'] == 5:
        scanner.recon = scanner.recon.mean(axis=0)
    ctshow(scanner.recon, 'brain')
    plt.savefig(fname, dpi=600, bbox_inches='tight')
    # update metadata
    results['uid'].append(uid)
    results['age'].append(age)
    if lesion['lesion_type'] == 'round':
        results['lesion_type'].append('intraparenchymal')
    else:
        results['lesion_type'].append(lesion['lesion_type'])
    results['volume'].append(np.round(lesion['volume'], decimals=1))
    results['attenuation'].append(np.round(lesion['intensity'], decimals=1))
    results['kVp'].append(scan['kVp'])
    results['mA'].append(scan['mA'])
    results['kernel'].append(recon['kernel'])
    results['slice_thickness'].append(recon['sliceThickness'])
    results['file'] = fname.relative_to(outdir.parent)
    results['seed'].append(scan_seed)
    pd.DataFrame(results).to_csv(outdir.parent / 'metadata.csv',
                                 index=False)
    if lesion['lesion_type'] == 'round':
        n_iph += 1
    elif lesion['lesion_type'] == 'epidural':
        n_edh += 1
    elif lesion['lesion_type'] == 'subdural':
        n_sdh += 1
    elif lesion['lesion_type'] is None:
        n_control += 1
    uid += 1
    # except Exception as err:
    #     logger.error(f"Runtime error occurred at {fname}\n{err}")
    #     continue
# %%
