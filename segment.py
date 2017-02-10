import argparse
import itertools
import numpy as np
import nibabel as nib

import chainer
from chainer import cuda, Variable

from load import load_nifti
from model import VoxResNet


def main():
    parser = argparse.ArgumentParser(description="perform segmentation using trained VoxResNet")
    parser.add_argument(
        "--input", "-i", type=str,
        help="brain image to segment")
    parser.add_argument(
        "--in_channels", type=int, default=2,
        help="number of input image channels, default=2")
    parser.add_argument(
        "--n_classes", "-c", type=int, default=4,
        help="number of classes, default=4")
    parser.add_argument(
        "--output", "-o", type=str, default="out.nii.gz",
        help="result of the segmentation, default=out.nii.gz")
    parser.add_argument(
        "--model", "-m", type=str,
        help="a file containing parameters of trained VoxResNet")
    parser.add_argument(
        "--shape", type=int, nargs="*", action="store",
        default=[80, 80, 80],
        help="patch shape to be passed to VoxResNet, default=[80, 80, 80]")
    parser.add_argument(
        "--gpu", "-g", default=-1, type=int,
        help="negative value indicates no gpu, default=-1")
    args = parser.parse_args()
    print(args)

    vrn = VoxResNet(in_channels=args.in_channels, n_classes=args.n_classes)
    chainer.serializers.load_npz(args.model, vrn)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        vrn.to_gpu()
    xp = cuda.cupy if args.gpu >= 0 else np

    scalar_img, affine = load_nifti(args.input, with_affine=True)
    if scalar_img.ndim == 3:
        scalar_img = np.expand_dims(scalar_img, 0)
    elif scalar_img.ndim == 4:
        scalar_img = scalar_img.transpose(3, 0, 1, 2)
    assert scalar_img.ndim == 4, scalar_img.ndim
    slices = [[], [], []]
    for img_len, patch_len, slice_ in zip(scalar_img.shape[1:], args.shape, slices):
        assert img_len > patch_len
        stride = int((img_len - patch_len) / int(img_len / patch_len))
        for i in xrange(int(img_len / patch_len)):
            slice_.append(slice(i * stride, i * stride + patch_len))
        slice_.append(slice(img_len - patch_len, img_len))
    output = np.zeros((args.n_classes,) + scalar_img.shape[1:])
    for xslice, yslice, zslice in itertools.product(*slices):
        scalar_patch = scalar_img[slice(None), xslice, yslice, zslice]
        scalar_patch = np.expand_dims(scalar_patch, 0)
        x = Variable(xp.asarray(scalar_patch))
        output[slice(None), xslice, yslice, zslice] += cuda.to_cpu(chainer.functions.softmax(vrn(x)).data[0])
    y = np.argmax(output, axis=0)
    nib.save(nib.Nifti1Image(y.astype(np.int32), affine), args.output)


if __name__ == '__main__':
    main()
