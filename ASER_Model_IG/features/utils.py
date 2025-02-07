'''import numpy as np


def downsample_with_max_pooling(array, factor=(1, 4)):
    if np.all(np.array(factor, int) == 1):
        return array

    sections = []

    for offset in np.ndindex(factor):
        part = array[tuple(np.s_[o::f] for o, f in zip(offset, factor))]
        sections.append(part)

    output = sections[0].copy()

    for section in sections[1:]:
        if output.shape == section.shape:
            np.maximum(output, section, output)
        else:
            if output.shape[0] != section.shape[0]:
                c = output.shape[0] - section.shape[0]
                pad = np.zeros((c, output.shape[1]))
                s = np.vstack((section, pad))
                np.maximum(output, s, output)
            if output.shape[1] != section.shape[1]:
                c = output.shape[1] - section.shape[1]
                pad = np.zeros((output.shape[0], c))
                s = np.hstack((section, pad))
                np.maximum(output, s, output)

    return output
'''

import numpy as np
import scipy.ndimage

def downsample_with_max_pooling(array, factor=(1, 4), stride=None, padding='valid'):
    """
    Downsample an array using max pooling.

    Parameters:
    -----------
    array : numpy.ndarray
        Input array to downsample.
    factor : tuple of int
        Pooling factor for each dimension (must be positive integers).
    stride : tuple of int, optional
        Stride for pooling windows. Defaults to `factor` (non-overlapping pooling).
    padding : str, optional
        Padding mode, either 'valid' (no padding) or 'same' (zero-padding to keep output size).

    Returns:
    --------
    numpy.ndarray
        Downsampled array.
    """
    # Validate parameters
    if not isinstance(factor, (tuple, list)) or len(factor) != array.ndim:
        raise ValueError(f"Pooling factor must match the number of dimensions in the array ({array.ndim}).")
    if stride is None:
        stride = factor
    if not isinstance(stride, (tuple, list)) or len(stride) != array.ndim:
        raise ValueError(f"Stride must match the number of dimensions in the array ({array.ndim}).")
    if padding not in ['valid', 'same']:
        raise ValueError("Padding must be either 'valid' or 'same'.")

    # Apply padding if necessary
    if padding == 'same':
        pad_width = []
        for f in factor:
            pad_width.append((0, f - 1))  # Pad at the end
        array = np.pad(array, pad_width, mode='constant', constant_values=0)

    # Define the output shape
    output_shape = [
        (array.shape[i] - factor[i]) // stride[i] + 1 if padding == 'valid'
        else (array.shape[i] + stride[i] - 1) // stride[i]
        for i in range(array.ndim)
    ]

    # Create an array of maximum values
    downsampled = np.zeros(output_shape, dtype=array.dtype)

    # Sliding window approach using maximum filter
    for index in np.ndindex(*output_shape):
        slices = tuple(
            slice(index[i] * stride[i], index[i] * stride[i] + factor[i])
            for i in range(array.ndim)
        )
        downsampled[index] = np.max(array[slices])

    return downsampled
