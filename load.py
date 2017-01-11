import random
import nibabel as nib
import numpy as np
import chainer


class DatasetFromFiles(chainer.dataset.DatasetMixin):

    def __init__(self, df, n_classes, shape=[80, 80, 80]):
        """
        construct training data object

        Parameters
        ----------
        df : DataFrame
            names of image files
        n_classes : int
            number of classes
        shape : array_like
            shape of patches to extract
        """
        self.df = df
        self.n_classes = n_classes
        self.shape = np.array(shape)

    def __len__(self):
        return len(self.df)

    def get_example(self, i):
        path_scalar_img = self.df["scalar"][i]
        path_label_img = self.df["label"][i]
        scalar_img = load_scalar(path_scalar_img)
        label_img = load_label(path_label_img, self.n_classes)
        p0 = np.array([random.randint(0, len_max - len_) for len_max, len_ in zip(scalar_img.shape, self.shape)])
        p1 = p0 + self.shape

        scalar_patch = extract_patch(scalar_img, p0, p1)
        label_patch = extract_patch(label_img, p0, p1)

        return np.expand_dims(scalar_patch, 0), label_patch.transpose(3, 0, 1, 2)


def load_nifti(filename):
    """
    load image from NIFTI file
    Parameters
    ----------
    filename : str
        filename of NIFTI file
    Returns
    -------
    img : np.ndarray
        image data
    """
    img = nib.load(filename).get_data()
    img = np.squeeze(img)
    img = np.copy(img, order='C')
    return img


def one_hot_encode(label, n_classes):
    return np.eye(n_classes, dtype=np.int32)[label]


def load_scalar(filename):
    return load_nifti(filename).astype(np.float32) / 255


def load_label(filename, n_classes):
    return one_hot_encode(load_nifti(filename).astype(np.int), n_classes)


def extract_patch(data, starts, ends):
    return data[starts[0]: ends[0], starts[1]: ends[1], starts[2]: ends[2]]


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
    batch : list
        pairs of image and label
    """
    N = len(df)
    assert N >= n
    indices = np.random.choice(N, n, replace=False)
    img_files = df["image"][indices]
    label_files = df["label"][indices]
    imgs = []
    labels = []
    for img_file, label_file in zip(img_files, label_files):
        img = load_img(img_file)
        label = load_label(label_file)
        ps = np.array([random.randint(0, len_max - len_) for len_, len_max in zip(shape, img.shape)])
        imgs.append(extract_patch(img, ps, [p + len_ for p, len_ in zip(ps, shape)]))
        labels.append(extract_patch(label, ps, [p + len_ for p, len_ in zip(ps, shape)]))
    imgs = np.array(imgs)
    labels = np.array(labels)
    return np.expand_dims(imgs, -1), labels


if __name__ == '__main__':
    import pandas as pd
    filename = "files.csv"
    df = pd.read_csv(filename)
    img, label = sample(df, 2, [20, 20, 20])
    print img.shape
    print label.shape
