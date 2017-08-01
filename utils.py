import numpy as np
import nibabel as nib


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


def load_sample(df, n, input_shape, output_shape):
    """
    randomly sample patch images from DataFrame

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing name of image files
    n : int
        number of patches to extract
    input_shape : list
        shape of input patches to extract
    output_shape : list
        shape of output patches to extract

    Returns
    -------
    images : (n, n_channels, input_shape[0], input_shape[1], ...) ndarray
        input patches
    labels : (n, output_shape[0], output_shape[1], ...) ndarray
        label patches
    """
    N = len(df)
    if "weight" in list(df):
        weights = np.asarray(df["weight"])
        weights /= np.sum(weights)
        indices = np.random.choice(N, n, replace=True, p=weights)
    else:
        indices = np.random.choice(N, n, replace=True)
    image_files = df["image"][indices]
    label_files = df["label"][indices]
    images = []
    labels = []
    for image_file, label_file in zip(image_files, label_files):
        image = load_nifti(image_file)
        label = load_nifti(label_file).astype(np.int32)
        mask = np.int32(label > 0)
        slices = [slice(len_ / 2, -len_ / 2) for len_ in input_shape]
        mask[slices] *= 2
        indices = np.where(mask > 1.5)
        i = np.random.choice(len(indices[0]))
        input_slices = [
            slice(index[i] - len_ / 2, index[i] + len_ / 2)
            for index, len_ in zip(indices, input_shape)
        ]
        output_slices = [
            slice(index[i] - len_ / 2, index[i] + len_ / 2)
            for index, len_ in zip(indices, output_shape)
        ]
        image_patch = image[input_slices]
        label_patch = label[output_slices]
        image_patch = image_patch.transpose(3, 0, 1, 2)
        images.append(image_patch)
        labels.append(label_patch)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


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
    mini = [c - len_ / 2 for c, len_ in zip(center, shape)]
    maxi = [c + len_ / 2 for c, len_ in zip(center, shape)]
    if all(m >= 0 for m in mini) and all(m < img_len for m, img_len in zip(maxi, image.shape)):
        slices = [slice(mi, ma) for mi, ma in zip(mini, maxi)]
    else:
        slices = [
            np.clip(range(mi, ma), 0, img_len - 1)
            for mi, ma, img_len in zip(mini, maxi, image.shape)
        ]
        slices = np.meshgrid(*slices, indexing="ij")
    patch = image[slices]
    patch = patch.transpose(3, 0, 1, 2)
    return patch


def dice_coefficients(label1, label2, labels=None):
    if labels is None:
        labels = np.unique(np.hstack((np.unique(label1), np.unique(label2))))
    dice_coefs = []
    for label in labels:
        match1 = (label1 == label)
        match2 = (label2 == label)
        denominator = 0.5 * (np.sum(match1.astype(np.float)) + np.sum(match2.astype(np.float)))
        numerator = np.sum(np.logical_and(match1, match2).astype(np.float))
        if denominator == 0:
            dice_coefs.append(0.)
        else:
            dice_coefs.append(numerator / denominator)
    return dice_coefs
