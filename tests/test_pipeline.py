# %%
from pathlib import Path
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from pedsilicoICH.pipeline import Scanner, load_phantom


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


def center_crop(img, thresh=-800):
    cropped = img[img.mean(axis=1) > thresh, :]
    cropped = cropped[:, img.mean(axis=0) > thresh]
    return cropped


def test_lesion_characteristics(age=15.75, views=100, name='lesions.png'):
    lesion_types = ['epidural', 'subdural', 'round']
    mass_effect = True

    outdir = Path('experiments/lesions')
    kernel = 'soft'
    outdir.mkdir(exist_ok=True, parents=True)

    imgs = []
    for lesion_type in lesion_types:
        volumes = np.linspace(0.1, 8, 3) if lesion_type == 'round'\
            else np.linspace(1, 20, 3)
        intensities = np.linspace(70, 50, 3)

        for intensity, volume in zip(intensities, volumes):
            print(f'{intensity} HU, {volume} mL')
            phantom = load_phantom(age, name=lesion_type, shape=None)
            phantom.insert_lesion(lesion_type=lesion_type, volume=volume,
                                  intensity=intensity,
                                  mass_effect=mass_effect, seed=336)
            lesion_level_mm = (phantom.get_CT_number_phantom().shape[0]/2 -
                               phantom._lesion_coords[0][0])*phantom.dz
            center = lesion_level_mm
            width = 8

            scanner = Scanner(phantom, output_dir=outdir)
            scanner.run_scan(startZ=center-width//2, endZ=center+width//2,
                             views=views)
            scanner.run_recon(kernel=kernel)
            imgs.append(scanner.recon[1:-1].mean(axis=0))

    f, axs = plt.subplots(3, 3, tight_layout=True)
    axs = axs.T
    for ax, img in zip(axs.flatten(), imgs):
        ctshow(center_crop(img), 'brain', fig=f, ax=ax)
    f.suptitle(' | '.join(lesion_types))
    f.savefig(name, dpi=600, bbox_inches='tight')


if __name__ == "__main__":
    parser = ArgumentParser(
        description='Generates test report figure')
    parser.add_argument('-v', '--views', type=int, default=100,
                        help='number of views to generate per series')
    parser.add_argument('-n', '--name', type=str,
                        help='filename to save report', default='lesions.png')
    parser.add_argument('-a', '--age', type=float, default=15.75,
                        help='phantom age to use')
    args = parser.parse_args()
    test_lesion_characteristics(age=args.age, views=args.views, name=args.name)
