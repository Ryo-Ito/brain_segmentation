import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def load_nifti(filename, with_affine=False):
    """
    load image from NIFTI file
    Parameters
    ----------
    filename : str
        filename of NIFTI file
    with_affine : bool
        if True, returns affine parameters

    Returns
    -------
    data : np.ndarray
        image data
    """
    img = nib.load(filename)
    data = img.get_data()
    data = np.copy(data, order="C")
    if with_affine:
        return data, img.affine
    return data


def sample(df, n, shape, scale):
    """
    randomly sample patch images from DataFrame

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing name of image files
    n : int
        number of patches to extract
    shape : list
        shape of patches to extract
    scale : int
        scaling factor

    Returns
    -------
    images : (n, n_channels, shape[0], shape[1], ...) ndarray
        input patches
    labels : (n, shape[0], shape[1], ...) ndarray
        label patches
    """
    N = len(df)
    assert N >= n, "n should be smaller than or equal to " + str(N)
    indices = np.random.choice(N, n, replace=False)
    image_files = df["image"][indices]
    label_files = df["label"][indices]
    images = []
    labels = []
    for image_file, label_file in zip(image_files, label_files):
        image = load_nifti(image_file)
        label = load_nifti(label_file).astype(np.int32)
        mask = np.int32(label > 0)
        indices = np.where(mask > 0)
        i = np.random.choice(len(indices[0]))
        slices = [
            np.clip(range(index[i] - len_ / 2, index[i] + len_ / 2), 0, max_)
            for index, len_, max_ in zip(indices, shape, image.shape)
        ]
        slices = np.meshgrid(*slices, indexing="ij")
        image_patch = image[slices]
        label_patch = label[slices]
        image_patch = image_patch.transpose(3, 0, 1, 2)
        if scale > 1:
            slices = [
                np.clip(range(index[i] - len_ * scale / 2, index[i] + len_ * scale / 2), 0, max_)
                for index, len_, max_ in zip(indices, shape, image.shape)
            ]
            slices = np.meshgrid(*slices, indexing="ij")
            image_patch_ = zoom(image[slices], (1. / scale,) * 3 + (1,))
            image_patch_ = image_patch_.transpose(3, 0, 1, 2)
            image_patch = np.concatenate((image_patch, image_patch_), axis=0)
        images.append(image_patch)
        labels.append(label_patch)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels
