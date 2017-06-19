import numpy as np


def crop_patch(image, center, shape):
    """
    crop patch from an image

    Parameters
    ----------
    image : (xlen, ylen, zlen, n_channels) np.ndarray
        input image to extract patch from
    center : [x, y, z] iterable
        center index of a patch
    shape : iterable
        shape of patch

    Returns
    -------
    patch : (n_channels, xlen, ylen, zlen) np.ndarray
        extracted patch
    """
    slices = [
        np.clip(range(c - len_ / 2, c + len_ / 2), 0, img_len - 1)
        for c, len_, img_len in zip(center, shape, image.shape)
    ]
    slices = np.meshgrid(*slices, indexing="ij")
    patch = image[slices]
    patch = patch.transpose(3, 0, 1, 2)
    return patch
