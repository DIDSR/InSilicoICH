"""
Module responsible for CT artifact generation, such as motion artification,
a common problem in pediatric imaging
"""
from copy import deepcopy
import numpy as np

from monai.transforms import RandAffine, Affine


def transform_image_label_pair(transform: RandAffine | Affine,
                               image: np.ndarray,
                               label: np.ndarray, seed: int = None):
    '''
    apply the same transform to the image and label and return the transformed
    outputs
    '''
    label_transform = deepcopy(transform)
    if isinstance(transform, RandAffine):
        seed = seed or np.random.randint(1e6)
        transform.set_random_state(seed=seed)
        label_transform.set_random_state(seed=seed)

    if isinstance(transform, RandAffine):
        return transform(image).numpy(), label_transform(label).numpy()
    return transform(image)[0], label_transform(label)[0]
