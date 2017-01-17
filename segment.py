import argparse
import numpy as np
import nibabel as nib

import chainer
from chainer import cuda, Variable

from load import load_scalar, extract_patch
from model import VoxResNet


def main():
    parser = argparse.ArgumentParser(description="perform segmentation using trained VoxResNet")
    parser.add_argument(
        "--input", "-i", type=str,
        help="brain image to segment")
    parser.add_argument(
        "--n_classes", "-c", type=int,
        help="number of classes")
    parser.add_argument(
        "--output", "-o", type=str,
        help="result of the segmentation")
    parser.add_argument(
        "--model", "-m", type=str,
        help="a file containing parameters of trained VoxResNet")
    parser.add_argument(
        "--shape", type=int, nargs="*", action="store",
        default=[80, 80, 80],
        help="patch shape to be passed to VoxResNet")
    parser.add_argument(
        "--gpu", "-g", default=-1, type=int,
        help="negative value indicates no gpu")
    args = parser.parse_args()
    print(args)

    vrn = VoxResNet(n_classes=args.n_classes)
    chainer.serializers.load_npz(args.model, vrn)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        vrn.to_gpu()
    xp = cuda.cupy if args.gpu >= 0 else np

    scalar_img = load_scalar(args.input)
    nifti_img = nib.load(args.input)
    start = [img_len / 2 - patch_len / 2 for img_len, patch_len in zip(scalar_img.shape, args.shape)]
    end = [patch_len + s for patch_len, s in zip(args.shape, start)]
    scalar_patch = extract_patch(scalar_img, start, end)
    scalar_patch = np.reshape(scalar_patch, [1, 1] + args.shape)

    x = Variable(xp.asarray(scalar_patch))
    output = vrn(x)
    y = np.argmax(output[-1].data, axis=1)

    nib.save(nib.Nifti1Image(np.pad(cuda.to_cpu(y[0]), ((88,), (24,), (88,)), mode="constant"), nifti_img.affine), args.output)


if __name__ == '__main__':
    main()
