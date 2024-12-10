import matplotlib.pyplot as plt
import numpy as np
from pedsilicoICH.image_acquisition import read_dicom
from pedsilicoICH.ground_truth_definition.phantoms import get_transformation_src_dst
from pedsilicoICH.lesion_definition import warp_slice
import nibabel as nib
import skimage as sk

from ipywidgets import * 
import ipywidgets as widgets
from IPython.display import display
from functools import partial

def study_viewer(df, img_dir, truth_dir):
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

    # define functions needed
    def show_study(df, img_dir, truth_dir, id, slice_idx=0, display='soft tissues'):
        f=None
        ax=None
        hemorrhage_label='NONE'
        cluster_threshold = 15
        patient = id

        img = nib.load(img_dir + patient)
        [dx, dy, dz] = img.header['pixdim'][1:4]
        image = img.get_fdata()

        mask = nib.load(truth_dir + patient).get_fdata()

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

        total_hemorrhage_volume = (len(np.argwhere(mask != 0)))*((dx*dy*dz)/1000)
        hemorrhage_count = 0

        if total_hemorrhage_volume != 0: # case has hemorrhage
            mask = np.where(mask != 0, 1, 0) # make binary

            label_mask, num = sk.measure.label(mask, return_num=True, connectivity=1)

            for cluster_idx in range(1,num+1):
                cluster = np.where(label_mask == cluster_idx, 1, 0)
                if np.count_nonzero(cluster) > cluster_threshold:

                    hemorrhage_count += 1

                    num_slices = 0
                    for slice_idx in range(cluster.shape[2]):
                        slice = cluster[:, :, slice_idx]
                        if np.any(slice): # check if hemorrhage
                            num_slices += 1

                    hemorrhage_volume = (len(np.argwhere(cluster == 1)))*((dx*dy*dz)/1000)

                    z_dist = num_slices * dz

                    # calculate mean and median HU
                    lesion_only = np.multiply(image, cluster)
                    lesion_only[lesion_only < -500] = 0
        
        print('Total number of hemorrhages: ' + str(hemorrhage_count))

    def on_save_click(id_dropdown, save_button):
        print('Saved!')
        print(id_dropdown.value)

    id_dropdown = Dropdown(description='id:', options=df['Data_ID'].unique())
    display_dropdown = Dropdown(description='display:', options=display_settings.keys())
    slice_slider = IntSlider(value=10, min=0, max=50)
    label_buttons = RadioButtons(options=['NONE', 'EDH', 'IPH', 'IVH', 'SAH', 'SDH', 'UNKNOWN'],
                                                  default_value='NONE', # defaults to none
                                                  description='Type:',
                                                  disabled=False)
    save_button = Button(description="Save Type Selection")
    
    viewer = lambda **kwargs: show_study(df, img_dir, truth_dir, **kwargs)

    ui = widgets.VBox([id_dropdown, display_dropdown, slice_slider, label_buttons, save_button])
    w = widgets.interactive_output(viewer, {'id': id_dropdown, 
                                            'display': display_dropdown,
                                            'slice_idx': slice_slider})
    
    save_button.on_click(partial(on_save_click, id_dropdown))


    # #slices = list(range(nib.load(img_dir / id).get_fdata().shape[2]+1))
    # w = interactive(viewer,
    #                 id=df['Data_ID'].unique(),
    #                 display=display_settings.keys(),
    #                 slice_idx=IntSlider(value=10, min=0, max=50),
    #                 hemorrhage_label=RadioButtons(options=['NONE', 'EDH', 'IPH', 'IVH', 'SAH', 'SDH', 'UNKNOWN'],
    #                                               default_value='NONE', # defaults to none
    #                                               description='Type:',
    #                                               disabled=False))
    
    display(ui, w)



