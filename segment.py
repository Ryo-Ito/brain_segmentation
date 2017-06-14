import argparse
import itertools
import json
import os

import chainer
import numpy as np
import nibabel as nib
import pandas as pd

from load import load_nifti, crop_patch
from model import VoxResNet


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
        "--shape", type=int, nargs="*", action="store",
        default=[80, 80, 80],
        help="input patch shape of VoxResNet, default=[80, 80, 80]")
    parser.add_argument(
        "--gpu", "-g", default=-1, type=int,
        help="negative value indicates no gpu, default=-1")
    parser.add_argument(
        "--n_tiles", type=int, nargs="*", action="store",
        default=[4, 4, 4],
        help="number of tiles along each axis")
    parser.add_argument(
        "--scale", type=int, default=1,
        help="scaling factor, default=1")
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
        for img_len, patch_len, center, n_tile in zip(image.shape[1:], args.shape, centers, args.n_tiles):
            assert img_len > patch_len, (img_len, patch_len)
            assert img_len < patch_len * n_tile, "{} must be smaller than {} x {}".format(img_len, patch_len, n_tile)
            stride = int((img_len - patch_len) / (n_tile - 1))
            for i in range(n_tile - 1):
                center.append(patch_len / 2 + i * stride)
            center.append(img_len - patch_len / 2)
        output = np.zeros((dataset["n_classes"],) + image.shape[:-1])
        for x, y, z in itertools.product(*centers):
            patch, slices = crop_patch(image, [x, y, z], args.shape, args.scale, return_slices=True)
            patch = np.expand_dims(patch, 0)
            x = xp.asarray(patch)
            output[:, slices[0], slices[1], slices[2]] += chainer.cuda.to_cpu(vrn(x).data[0])
        y = np.argmax(output, axis=0)
        nib.save(
            nib.Nifti1Image(np.int32(y), affine),
            os.path.join(
                os.path.dirname(image_path),
                subject + args.output_suffix))


if __name__ == '__main__':
    main()
