"""
Module responsible for CT artifact generation, such as motion artification,
a common problem in pediatric imaging
"""
import numpy as np

from monai.transforms import RandAffine


def transform_image_label_pair(transform: RandAffine, image: np.ndarray,
                               label: np.ndarray, seed: int = None):
    '''
    apply the same transform to the image and label and return the transformed
    outputs
    '''
    seed = seed or np.random.randint(1e6)
    transform.set_random_state(seed=seed)
    img_transform = transform(image)

    transform.set_random_state(seed=seed)
    lesion_transform = transform(label)
    return img_transform.numpy(), lesion_transform.numpy()
