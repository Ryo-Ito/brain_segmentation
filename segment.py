import argparse
import itertools
import json
import os

import chainer
import numpy as np
import nibabel as nib
import pandas as pd

from model import VoxResNet
from utils import crop_patch, load_nifti, feedforward


def main():
    parser = argparse.ArgumentParser(description="segment with VoxResNet")
    parser.add_argument(
        "--input_file", "-i", type=str,
        help="input json file of test dataset")
    parser.add_argument(
        "--output_suffix", "-o", type=str,
        help="result of the segmentation")
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
        help="number of tiles along each axis, default=[5, 5, 5]")
    args = parser.parse_args()
    print(args)

    with open(args.input_file) as f:
        dataset = json.load(f)
    test_df = pd.DataFrame(dataset["data"])

    vrn = VoxResNet(dataset["in_channels"], dataset["n_classes"])
    chainer.serializers.load_npz(args.model, vrn)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        vrn.to_gpu()

    for image_path, subject in zip(test_df["image"], test_df["subject"]):
        image, affine = load_nifti(image_path, with_affine=True)
        output = feedforward(
            vrn,
            image,
            args.input_shape,
            args.output_shape,
            args.n_tiles,
            dataset["n_classes"]
        )
        y = np.argmax(output, axis=0)
        nib.save(
            nib.Nifti1Image(np.int32(y), affine),
            os.path.join(
                os.path.dirname(image_path),
                subject + args.output_suffix))


if __name__ == '__main__':
    main()
