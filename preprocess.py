import argparse
import json
import os

from dipy.align.reslice import reslice
import nibabel as nib
import numpy as np
import SimpleITK as sitk


def clahe(img):
    pass


def preprocess(inputfile, outputfile, order=0):
    img = nib.load(inputfile)
    data = img.get_data()
    affine = img.affine
    zoom = img.header.get_zooms()[:3]
    data, affine = reslice(data, affine, zoom, (1., 1., 1.), order)
    if order == 0:
        data = np.int32(data)
        data = np.squeeze(data)
        assert data.ndim == 3, data.ndim
    else:
        img = sitk.GetImageFromArray(np.squeeze(data))
        img = sitk.AdaptiveHistogramEqualization(img)
        data_clahe = sitk.GetArrayFromImage(img)[:, :, :, None]
        data = np.concatenate((data_clahe, data), 3)
        data = (data - np.mean(data, (0, 1, 2))) / np.std(data, (0, 1, 2))
        assert data.ndim == 4, data.ndim
        assert np.allclose(np.mean(data, (0, 1, 2)), 0.), np.mean(data, (0, 1, 2))
        assert np.allclose(np.std(data, (0, 1, 2)), 1.), np.std(data, (0, 1, 2))
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
        "--output_json", type=str, default="dataset.json",
        help="json file of preprocessed dataset, default=dataset.json")
    args = parser.parse_args()

    dataset = {"in_channels": 2, "n_classes": 4}
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
            order=1)

        filename = subject + args.label_suffix
        outputfile = os.path.join(output_folder, filename)
        filedict["label"] = outputfile
        preprocess(
            os.path.join(args.input_directory, subject, filename),
            outputfile,
            order=0)
        dataset_list.append(filedict)
    dataset["data"] = dataset_list

    with open(args.output_json, "w") as f:
        json.dump(dataset, f)


if __name__ == '__main__':
    main()
