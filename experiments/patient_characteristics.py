from pathlib import Path
from pedsilicoICH.pipeline import run_study
import matplotlib.pyplot as plt
from utils import ctshow, center_crop

ages = [38, 6.5, 9.0, 10.5, 11.5, 12.0, 15.75]

imgs = {}
gts = {}
outdir = Path('experiments/ages')
outdir.mkdir(exist_ok=True)
for age in ages:
    if age == 38:
        startZ = 40
        endZ = 50
    else:
        startZ = 0
        endZ = 10
    study = run_study(output_directory='show_and_tell',
                      zspan=(startZ, endZ), age=age, views=1000, kernel='soft',
                      add_positioning_augmentation=False)
    imgs[age] = study.images[study.images.shape[0]//2]

    f, ax = plt.subplots(1, 3)
    ax[0].imshow(study.phantom.get_CT_number_phantom().sum(axis=1), cmap='gray')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    center_slice = int(study.phantom.shape[0]/2 -
                       ((startZ+endZ)/2)/study.phantom.dz)
    ctshow(center_crop(study.phantom.get_CT_number_phantom()[center_slice]),
           'brain', fig=f, ax=ax[1])
    ax[1].set_title(f'{age} years')
    ctshow(center_crop(study.images[1:-1].mean(axis=0)),
           'brain', fig=f, ax=ax[2])
    plt.savefig(outdir / f'{age}_yrs.png', dpi=600, bbox_inches='tight')
