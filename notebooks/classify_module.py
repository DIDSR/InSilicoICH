import matplotlib.pyplot as plt
import numpy as np
from pedsilicoICH.image_acquisition import read_dicom
from pedsilicoICH.ground_truth_definition.phantoms import get_transformation_src_dst
from pedsilicoICH.lesion_definition import warp_slice
import nibabel as nib

from ipywidgets import interact, IntSlider

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

def show_study(img_dir, truth_dir, id, slice_idx=0, display='soft tissues', f=None, ax=None):
    patient = id

    img = nib.load(img_dir / patient)
    [dx, dy, dz] = img.header['pixdim'][1:4]
    image = img.get_fdata()

    mask = nib.load(truth_dir / patient).get_fdata()

    ww, wl = display_settings[display]
    minn = wl - ww/2
    maxx = wl + ww/2

    if (f is None) or (ax is None):
        f, ax = plt.subplots()

    if slice_idx < image.shape[2]:
        im = ax.imshow(image[:, :, slice_idx], cmap='gray', vmin=minn, vmax=maxx)
    else:
        im = ax.imshow(image[:, :, image.shape[2]-1], cmap='gray', vmin=minn, vmax=maxx)
    plt.colorbar(im, ax=ax, label=f'HU | {display} [ww: {ww}, wl: {wl}]')
    ax.set_title(patient)

def study_viewer(df, img_dir, truth_dir):
    viewer = lambda **kwargs: show_study(img_dir, truth_dir, **kwargs)
    #slices = list(range(nib.load(img_dir / id).get_fdata().shape[2]+1))
    interact(viewer,
             id=df['Data_ID'].unique(),
             display=display_settings.keys(),
             slice_idx=IntSlider(value=10, min=0, max=50))