import argparse
import itertools
import json
import os

import chainer
import numpy as np
import nibabel as nib
import pandas as pd

from load import load_nifti
from model import VoxResNet
from utils import crop_patch


def main():
    parser = argparse.ArgumentParser(description="segment with VoxResNet")
    parser.add_argument(
        "--input_file", "-i", type=str,
        help="input json file of test dataset")
    parser.add_argument(
        "--output_suffix", "-o", type=str, default="_segTRI_predict.nii.gz",
        help="result of the segmentation, default=_segTRI_predict.nii.gz")
    parser.add_argument(
        "--model", "-m", type=str,
        help="a file containing parameters of trained VoxResNet")
    parser.add_argument(
        "--input_shape", type=int, nargs="*", action="store",
        default=[80, 80, 80],
        help="input patch shape of VoxResNet, default=[80, 80, 80]")
    parser.add_argument(
        "--output_shape", type=int, nargs="*", action="store",
        default=[60, 60, 60],
        help="output patch shape of VoxResNet, default=[60, 60, 60]")
    parser.add_argument(
        "--gpu", "-g", default=-1, type=int,
        help="negative value indicates no gpu, default=-1")
    parser.add_argument(
        "--n_tiles", type=int, nargs="*", action="store",
        default=[5, 5, 5],
        help="number of tiles along each axis")
    args = parser.parse_args()
    print(args)

    with open(args.input_file) as f:
        dataset = json.load(f)
    test_df = pd.DataFrame(dataset["data"])

    vrn = VoxResNet(dataset["in_channels"], dataset["n_classes"])
    chainer.serializers.load_npz(args.model, vrn)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        vrn.to_gpu()
        xp = chainer.cuda.cupy
    else:
        xp = np

    for image_path, subject in zip(test_df["image"], test_df["subject"]):
        image, affine = load_nifti(image_path, with_affine=True)
        centers = [[], [], []]
        for img_len, len_out, center, n_tile in zip(image.shape, args.output_shape, centers, args.n_tiles):
            assert img_len < len_out * n_tile, "{} must be smaller than {} x {}".format(img_len, len_out, n_tile)
            stride = int((img_len - len_out) / (n_tile - 1))
            center.append(len_out / 2)
            for i in range(n_tile - 2):
                center.append(center[-1] + stride)
            center.append(img_len - len_out / 2)
        output = np.zeros((dataset["n_classes"],) + image.shape[:-1])
        for x, y, z in itertools.product(*centers):
            patch = crop_patch(image, [x, y, z], args.input_shape)
            patch = np.expand_dims(patch, 0)
            patch = xp.asarray(patch)
            slices_out = [slice(center - len_out / 2, center + len_out / 2) for len_out, center in zip(args.output_shape, [x, y, z])]
            slices_in = [slice((len_in - len_out) / 2, len_in - (len_in - len_out) / 2) for len_out, len_in, in zip(args.output_shape, args.input_shape)]
            output[slice(None), slices_out[0], slices_out[1], slices_out[2]] += chainer.cuda.to_cpu(
                vrn(patch).data[0, slice(None), slices_in[0], slices_in[1], slices_in[2]])
        y = np.argmax(output, axis=0)
        nib.save(
            nib.Nifti1Image(np.int32(y), affine),
            os.path.join(
                os.path.dirname(image_path),
                subject + args.output_suffix))


if __name__ == '__main__':
    main()
