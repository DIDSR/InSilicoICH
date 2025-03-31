from pathlib import Path
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from insilicoICH.study import Scanner, load_phantom


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


def center_crop_like(img, ref, thresh=-800):
    cropped = img[ref.mean(axis=1) > thresh, :]
    cropped = cropped[:, ref.mean(axis=0) > thresh]
    return cropped


def plot_montage(imgs, params, fname):
    f, axs = plt.subplots(3, 3, figsize=(6.9, 8),
                          gridspec_kw=dict(wspace=0, hspace=0))
    axs = axs.T
    for idx, ax, img in zip(range(len(imgs)), axs.flatten(), imgs):
        lesion_type, intensity, volume = params[idx]
        if idx < 1:
            disp_img = center_crop(img)
        else:
            disp_img = center_crop_like(img, imgs[0])
        ctshow(disp_img, 'brain', fig=f, ax=ax)
        ax.set_aspect('equal')
        if idx in [0, 3, 6]:
            ax.set_title(lesion_type)
        bbox = dict(boxstyle="round", fc="0.8")
        if idx == 3:
            ax.annotate('', xy=(220, 90), xytext=(170, 150),
                        arrowprops=dict(facecolor='black', shrink=0.05))
        if idx == 4:
            ax.annotate('', xy=(220, 75), xytext=(170, 150),
                        arrowprops=dict(facecolor='black', shrink=0.05))
        if idx == 5:
            ax.annotate('', xy=(270, 200), xytext=(190, 200),
                        arrowprops=dict(facecolor='black', shrink=0.05))
        ax.annotate(f'{volume} mL {intensity} HU', xy=(50, 20), bbox=bbox)
    f.savefig(fname, dpi=600, bbox_inches='tight')


def test_lesion_characteristics(age=15.75, views=100,
                                name='tests/images.png'):
    lesion_types = ['EDH', 'SDH', 'IPH']
    mass_effect = True

    name = Path(name)
    name.parent.mkdir(parents=True, exist_ok=True)

    outdir = name.parent / 'lesions'
    kernel = 'soft'
    outdir.mkdir(exist_ok=True, parents=True)

    phantoms = []
    imgs = []
    params = []
    for lesion_type in lesion_types:
        volumes = np.linspace(0.1, 8, 3) if lesion_type == 'IPH'\
            else np.linspace(1, 20, 3)
        intensities = np.linspace(70, 50, 3)

        for intensity, volume in zip(intensities, volumes):
            print(f'{intensity} HU, {volume} mL')
            phantom = load_phantom(age, name=lesion_type, shape=None)
            phantom.insert_lesion(lesion_type=lesion_type, volume=volume,
                                  intensity=intensity,
                                  mass_effect=mass_effect, seed=336)
            params.append((lesion_type, intensity, volume))
            phantoms.append(phantom.get_CT_number_phantom()[
                phantom._lesion_coords[0][0]])
            lesion_level_mm = (phantom.get_CT_number_phantom().shape[0]/2 -
                               phantom._lesion_coords[0][0])*phantom.dz
            center = lesion_level_mm
            width = 7

            scanner = Scanner(phantom, output_dir=outdir)
            scanner.run_scan(startZ=center-width/2, endZ=center+width/2,
                             views=views)
            scanner.run_recon(kernel=kernel)
            imgs.append(scanner.recon[1:-1].mean(axis=0))

    plot_montage(imgs, params, name)
    plot_montage(phantoms, params, name.parent / 'phantoms.png')


if __name__ == "__main__":
    parser = ArgumentParser(
        description='Generates test report figure')
    parser.add_argument('-v', '--views', type=int, default=100,
                        help='number of views to generate per series')
    parser.add_argument('-n', '--name', type=str,
                        help='filename to save report', default='images.png')
    parser.add_argument('-a', '--age', type=float, default=15.75,
                        help='phantom age to use')
    args = parser.parse_args()
    test_lesion_characteristics(age=args.age, views=args.views, name=args.name)
