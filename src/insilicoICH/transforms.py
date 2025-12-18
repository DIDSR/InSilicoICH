
import numpy as np
import scipy.ndimage
from typing import Union, Sequence, Optional

def resize(phantom: np.ndarray, shape: tuple | int, **kwargs) -> np.ndarray:
    """Resizes a phantom to a new shape while maintaining aspect ratio.

    This function uses scipy.ndimage.zoom to resize a 2D or 3D phantom
    array. This implements the equivalent of MONAI's `size_mode='longest'`
    which scales the longest dimension to match the corresponding dimension
    in `shape`, and scales other dimensions proportionally.

    mode = 'nearest' is useful for downsizing without interpolation errors
    (mapped to scipy order=0).

    Args:
        phantom (np.ndarray): The phantom image array to resize.
        shape (tuple | int): The target shape for the phantom. If a tuple,
            max(shape) is used as the target size for the longest dimension.
        **kwargs: Additional keyword arguments.
            Supported:
            - mode: 'nearest' (use order=0), otherwise defaults to order=1 (linear).
            - order: directly passed to scipy.ndimage.zoom.

    Returns:
        np.ndarray: The resized phantom array.
    """
    if isinstance(shape, (tuple, list)):
        target_max = max(shape)
    else:
        target_max = shape

    current_max = max(phantom.shape)
    scale = target_max / current_max

    # Handle interpolation mode
    mode = kwargs.get('mode', None)
    order = kwargs.get('order', 1) # Default to linear (1) for safety
    if mode == 'nearest':
        order = 0

    # Calculate zoom factors for each dimension
    zoom = [scale] * phantom.ndim

    resized = scipy.ndimage.zoom(phantom, zoom, order=order)
    return resized


class ResizeWithPadOrCrop:
    def __init__(self, spatial_size: Sequence[int], mode: str = 'constant', **kwargs):
        self.spatial_size = np.array(spatial_size)
        self.mode = mode
        self.kwargs = kwargs

    def __call__(self, img: np.ndarray) -> np.ndarray:
        # Handle generic dimensionality where we pad/crop the spatial dims.
        # We will operate on the last N dims where N is len(spatial_size).

        # Determine strict spatial dims
        rank = len(self.spatial_size)
        img_spatial_shape = np.array(img.shape[-rank:])

        diff = self.spatial_size - img_spatial_shape

        # If shapes match, return
        if np.all(diff == 0):
            return img

        # Calculate padding/cropping
        # positive diff -> pad
        # negative diff -> crop

        slices = [slice(None)] * img.ndim
        pads = [(0, 0)] * img.ndim

        for i in range(rank):
            dim_idx = i - rank # -3, -2, -1 for 3D
            d = diff[i]
            if d > 0: # Pad
                pad_left = d // 2
                pad_right = d - pad_left
                pads[dim_idx] = (pad_left, pad_right)
            elif d < 0: # Crop
                crop_left = abs(d) // 2
                crop_right = abs(d) - crop_left
                # End index is shape - crop_right
                start = crop_left
                end = img.shape[dim_idx] - crop_right
                slices[dim_idx] = slice(start, end)

        # Apply Crop
        cropped = img[tuple(slices)]

        # Apply Pad
        if any(p != (0, 0) for p in pads):
            # map mode to numpy mode
            np_mode = 'constant'
            if self.mode == 'edge': np_mode = 'edge'
            elif self.mode == 'reflect': np_mode = 'reflect'

            padded = np.pad(cropped, pads, mode=np_mode)
            return padded

        return cropped


class Affine:
    """
    Affine transform implementation using scipy.ndimage.affine_transform.
    Supports basic affine parameters or explicit matrix.
    """
    def __init__(self, rotate_params=None, shear_params=None, translate_params=None, scale_params=None,
                 affine=None, spatial_size=None, mode='bilinear', padding_mode='zeros', **kwargs):
        self.rotate_params = rotate_params
        self.shear_params = shear_params
        self.translate_params = translate_params
        self.scale_params = scale_params
        self.affine = affine
        self.spatial_size = spatial_size
        self.mode = mode
        self.padding_mode = padding_mode
        self.kwargs = kwargs

    def __call__(self, img: np.ndarray, mode: str = None, padding_mode: str = None) -> np.ndarray:
        # Detect spatial dims
        ndim = img.ndim
        if ndim == 4: # C, Z, Y, X
            spatial_dims = 3
            is_channel_first = True
        elif ndim == 3: # Z, Y, X
            spatial_dims = 3
            is_channel_first = False
        else:
            # Fallback for 2D
            spatial_dims = ndim
            is_channel_first = False

        if is_channel_first:
            spatial_shape = img.shape[1:]
        else:
            spatial_shape = img.shape

        center = (np.array(spatial_shape) - 1.0) / 2.0

        # Construct affine matrix if not provided
        if self.affine is None:
            # Defaults
            angle = self.rotate_params if self.rotate_params is not None else np.zeros(spatial_dims)
            scale = self.scale_params if self.scale_params is not None else np.ones(spatial_dims)
            translate = self.translate_params if self.translate_params is not None else np.zeros(spatial_dims)
            # Ignoring shear for simplicity unless needed, complex to implement generic ND shear without clearer requirements

            # Build matrices
            # 1. Scale
            S = np.diag(list(scale) + [1])

            # 2. Rotate (assuming Euler angles for 3D)
            R = np.eye(spatial_dims + 1)
            if spatial_dims == 3:
                # Ensure angle has 3 components
                if np.isscalar(angle): angle = [angle] * 3
                rx, ry, rz = angle

                Rx = np.eye(4)
                Rx[1, 1] = np.cos(rx)
                Rx[1, 2] = -np.sin(rx)
                Rx[2, 1] = np.sin(rx)
                Rx[2, 2] = np.cos(rx)

                Ry = np.eye(4)
                Ry[0, 0] = np.cos(ry)
                Ry[0, 2] = np.sin(ry)
                Ry[2, 0] = -np.sin(ry)
                Ry[2, 2] = np.cos(ry)

                Rz = np.eye(4)
                Rz[0, 0] = np.cos(rz)
                Rz[0, 1] = -np.sin(rz)
                Rz[1, 0] = np.sin(rz)
                Rz[1, 1] = np.cos(rz)

                R = Rz @ Ry @ Rx
            elif spatial_dims == 2:
                if not np.isscalar(angle): angle = angle[0]
                theta = angle
                R = np.eye(3)
                R[0, 0] = np.cos(theta)
                R[0, 1] = -np.sin(theta)
                R[1, 0] = np.sin(theta)
                R[1, 1] = np.cos(theta)

            M_fwd = R @ S
        else:
            M_fwd = self.affine
            translate = np.zeros(spatial_dims) # If matrix provided, translation usually included or separate?
            # MONAI Affine: affine is 4x4. If provided, ignores other params.
            # Assuming affine includes translation if it's 4x4.
            # But here translate_params is separate in init.
            # If self.affine is given, we assume it's the full transform?
            # Let's assume M_fwd is linear part if 3x3, or full if 4x4.
            if M_fwd.shape == (spatial_dims + 1, spatial_dims + 1):
                # Extract linear and translation
                translate = M_fwd[:spatial_dims, spatial_dims]
                M_fwd_linear = M_fwd.copy()
                M_fwd_linear[:spatial_dims, spatial_dims] = 0
                M_fwd = M_fwd_linear # Treat linear part separately for center logic
            else:
                translate = np.zeros(spatial_dims)

        # Common logic for application (shared with RandAffine mostly)
        # We need Inverse for scipy (Output -> Input)

        # Linear part
        T2 = M_fwd

        # Translation matrices
        T1 = np.eye(spatial_dims + 1)
        T1[:spatial_dims, spatial_dims] = -center

        T3 = np.eye(spatial_dims + 1)
        T3[:spatial_dims, spatial_dims] = translate

        T4 = np.eye(spatial_dims + 1)
        T4[:spatial_dims, spatial_dims] = center

        # Composite: Center -> Transform -> Uncenter
        # Note: Order matters.
        # Rotate/Scale around center: T4 @ (R@S) @ T1
        # Then translate: T3 @ (T4 @ (R@S) @ T1) ??
        # MONAI: "The order of operations is: shear -> rotate -> scale -> translate".
        # And transforms are around the center usually?
        # Let's stick to the logic derived in RandAffine:
        # x_out = (R * S) * (x_in - center) + translation + center

        T_fwd = T4 @ T3 @ T2 @ T1
        T_inv = np.linalg.inv(T_fwd)

        scipy_matrix = T_inv[:spatial_dims, :spatial_dims]
        scipy_offset = T_inv[:spatial_dims, spatial_dims]

        # Mode handling
        effective_mode = mode or self.mode
        effective_padding_mode = padding_mode or self.padding_mode

        order = 1
        if effective_mode == 'nearest': order = 0
        elif effective_mode == 'bilinear': order = 1
        elif effective_mode == 'bicubic': order = 3

        scipy_mode = 'constant'
        cval = 0.0
        if effective_padding_mode == 'border': scipy_mode = 'nearest'
        elif effective_padding_mode == 'zeros': scipy_mode = 'constant'; cval = 0.0
        elif effective_padding_mode == 'reflection': scipy_mode = 'reflect'

        if is_channel_first:
            out = np.zeros_like(img)
            for c in range(img.shape[0]):
                out[c] = scipy.ndimage.affine_transform(
                    img[c],
                    matrix=scipy_matrix,
                    offset=scipy_offset,
                    order=order,
                    mode=scipy_mode,
                    cval=cval
                )
            return out
        else:
            return scipy.ndimage.affine_transform(
                img,
                matrix=scipy_matrix,
                offset=scipy_offset,
                order=order,
                mode=scipy_mode,
                cval=cval
            )


class RandAffine(Affine):
    def __init__(self, prob: float = 0.1,
                 rotate_range: Union[Sequence[float], float] = None,
                 translate_range: Union[Sequence[float], float] = None,
                 scale_range: Union[Sequence[float], float] = None,
                 shear_range: Union[Sequence[float], float] = None,
                 spatial_size: Union[Sequence[int], int] = None,
                 mode: str = 'bilinear',
                 padding_mode: str = 'zeros',
                 **kwargs):
        super().__init__(mode=mode, padding_mode=padding_mode, spatial_size=spatial_size, **kwargs)
        self.prob = prob
        self.rotate_range = rotate_range
        self.translate_range = translate_range
        self.scale_range = scale_range
        self.shear_range = shear_range
        self.rng = np.random.default_rng()

    def set_random_state(self, seed: int = None):
        self.rng = np.random.default_rng(seed)

    def _get_rand_param(self, param_range, dim, default_val=0.0):
        if param_range is None:
            if default_val == 1.0: # Scale
                return np.ones(dim)
            return np.zeros(dim)

        if np.isscalar(param_range):
            return self.rng.uniform(-param_range, param_range, size=dim)

        if len(param_range) == dim:
             return np.array([self.rng.uniform(-r, r) for r in param_range])
        elif len(param_range) == 2 and np.isscalar(param_range[0]):
             return np.array([self.rng.uniform(-r, r) for r in param_range])

        return np.zeros(dim)

    def _get_scale_param(self, param_range, dim):
        if param_range is None:
            return np.ones(dim)

        if hasattr(param_range, '__len__') and len(param_range) == dim:
             return np.array([self.rng.uniform(1 - r, 1 + r) for r in param_range])

        if np.isscalar(param_range):
            return self.rng.uniform(1 - param_range, 1 + param_range, size=dim)

        return np.ones(dim)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.rng.random() > self.prob:
            return img

        ndim = img.ndim
        if ndim == 4: spatial_dims = 3
        elif ndim == 3: spatial_dims = 3
        else: spatial_dims = ndim

        # Generate params
        self.rotate_params = self._get_rand_param(self.rotate_range, spatial_dims)
        self.translate_params = self._get_rand_param(self.translate_range, spatial_dims)
        self.scale_params = self._get_scale_param(self.scale_range, spatial_dims)
        self.affine = None # Ensure we calculate fresh matrix

        # Call parent Affine.__call__ which does the matrix construction and application
        return super().__call__(img)
