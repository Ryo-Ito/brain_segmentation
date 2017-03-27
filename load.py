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
        mask = np.float32(label > 0)
        slices = [slice(len_ / 2, -len_ / 2) for len_ in shape]
        mask[slices] *= 2
        indices = np.where(mask > 1.5)
        i = np.random.choice(len(indices[0]))
        slices = [
            slice(index[i] - len_ / 2, index[i] + len_ / 2)
            for index, len_ in zip(indices, shape)]
        image_patch = image[slices]
        label_patch = label[slices]
        image_patch = image_patch.transpose(3, 0, 1, 2)
        images.append(image_patch)
        labels.append(label_patch)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels
