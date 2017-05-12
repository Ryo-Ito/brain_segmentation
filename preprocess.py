import argparse
import json
import os

import cv2
import nibabel as nib
import numpy as np
import pandas as pd


def preprocess(inputfile, outputfile, order=0, df=None, slice_axis=0):
    img = nib.load(inputfile)
    data = img.get_data()
    affine = img.affine
    data = np.squeeze(data)
    if order == 0:
        if df is not None:
            for target, raw in zip(df["preprocessed"], df["raw"]):
                data[np.where(data == raw)] = target
        data = np.int32(data)
        assert data.ndim == 3, data.ndim
    else:
        data = np.swapaxes(data, 0, slice_axis)
        data_original = np.copy(data)
        data = np.stack((data, data_original), axis=-1)
        assert data.shape == data_original.shape + (2,), data.shape
        clahe = cv2.createCLAHE(clipLimit=2., tileGridSize=(16, 16))
        for i, slice_ in enumerate(data_original):
            slice_ = (slice_ - slice_.mean()) / slice_.std()
            slice_ = np.nan_to_num(slice_)
            slice_sub = slice_ - cv2.GaussianBlur(slice_, (31, 31), 5)
            slice_sub = (slice_sub - np.min(slice_sub)) / (np.max(slice_sub) - np.min(slice_sub))
            slice_sub = np.array(slice_sub * 255, dtype=np.uint8)
            slice_clahe = clahe.apply(slice_sub).astype(np.float)
            slice_clahe = (slice_clahe - slice_clahe.mean()) / slice_clahe.std()
            slice_clahe = np.nan_to_num(slice_clahe)
            data[i, :, :, 0] = slice_clahe
            data[i, :, :, 1] = slice_
        data = np.swapaxes(data, 0, slice_axis)
        assert data.ndim == 4, data.ndim
        data = np.float32(data)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, outputfile)


def main():
    parser = argparse.ArgumentParser(description="preprocess dataset")
    parser.add_argument(
        "--input_directory", "-i", type=str,
        help="directory of original dataset")
    parser.add_argument(
        "--subjects", "-s", type=str, nargs="*", action="store",
        help="subjects to be preprocessed")
    parser.add_argument(
        "--image_suffix", type=str, default="_ana_strip.nii.gz",
        help="suffix of images, default=_ana_strip.nii.gz")
    parser.add_argument(
        "--label_suffix", type=str, default="_segTRI_ana.nii.gz",
        help="suffix of labels, default=_segTRI_ana.nii.gz")
    parser.add_argument(
        "--output_directory", "-o", type=str,
        help="directory of preprocessed dataset")
    parser.add_argument(
        "--output_file", "-f", type=str, default="dataset.json",
        help="json file of preprocessed dataset, default=dataset.json")
    parser.add_argument(
        "--label_file", "-l", type=str, default=None,
        help="csv file with label translation rule, default=None")
    parser.add_argument(
        "--n_classes", type=int, default=4,
        help="number of classes to classify")
    parser.add_argument(
        "--slice_axis", type=int, default=1,
        help="index of axis to perform preprocessing")
    args = parser.parse_args()

    if args.label_file is None:
        df = None
    else:
        df = pd.read_csv(args.label_file)

    dataset = {"in_channels": 2, "n_classes": args.n_classes}
    dataset_list = []

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)
    for subject in args.subjects:
        output_folder = os.path.join(args.output_directory, subject)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        filedict = {"subject": subject}

        filename = subject + args.image_suffix
        outputfile = os.path.join(output_folder, filename)
        filedict["image"] = outputfile
        preprocess(
            os.path.join(args.input_directory, subject, filename),
            outputfile,
            order=1,
            slice_axis=args.slice_axis)

        filename = subject + args.label_suffix
        outputfile = os.path.join(output_folder, filename)
        filedict["label"] = outputfile
        preprocess(
            os.path.join(args.input_directory, subject, filename),
            outputfile,
            order=0,
            df=df)
        dataset_list.append(filedict)
    dataset["data"] = dataset_list

    with open(args.output_file, "w") as f:
        json.dump(dataset, f)


if __name__ == '__main__':
    main()
