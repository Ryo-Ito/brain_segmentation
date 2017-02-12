import nibabel as nib
import numpy as np


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
    data = np.squeeze(data)
    data = np.copy(data, order="C")
    if with_affine:
        return data, img.affine
    return data


def sample(df, n, shape):
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
    Returns
    -------
    scalars : (n, n_channels, shape[0], shape[1], ...) ndarray
        scalar patches
    labels : (n, shape[0], shape[1], ...) ndarray
        label patches
    """
    N = len(df)
    assert N >= n
    indices = np.random.choice(N, n, replace=False)
    scalar_files = df["preprocessed"][indices]
    label_files = df["segTRI"][indices]
    mask_files = df["mask"][indices]
    scalars = []
    labels = []
    for scalar_file, label_file, mask_file in zip(scalar_files, label_files, mask_files):
        scalar_img = load_nifti(scalar_file)
        if scalar_img.ndim == 3:
            scalar_img = (scalar_img - np.mean(scalar_img)) / np.std(scalar_img)
            scalar_img = scalar_img[:, :, :, None].astype(np.float32)
        label_img = load_nifti(label_file).astype(np.int32)
        mask_img = load_nifti(mask_file)
        slices = [slice(len_ / 2, -len_ / 2) for len_ in shape]
        mask_img[slices] *= 2
        indices = np.where(mask_img > 1.5)
        i = np.random.choice(len(indices[0]))
        slices = [slice(index[i] - len_ / 2, index[i] + len_ / 2) for index, len_ in zip(indices, shape)]
        scalar_patch = scalar_img[slices]
        label_patch = label_img[slices]
        scalar_patch = scalar_patch.transpose(3, 0, 1, 2)
        scalars.append(scalar_patch)
        labels.append(label_patch)
    scalars = np.array(scalars)
    labels = np.array(labels)
    return scalars, labels
