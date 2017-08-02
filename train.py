import argparse
import itertools
import json

import chainer
import chainer.functions as F
import numpy as np
import pandas as pd

from model import VoxResNet
from utils import load_sample, load_nifti, crop_patch, dice_coefficients, feedforward


def validate(model, df, input_shape, output_shape, n_tiles, n_classes):
    dice_coefs = []
    for image_path, label_path in zip(df["image"], df["label"]):
        image = load_nifti(image_path)
        label = load_nifti(label_path)
        output = feedforward(
            model,
            image,
            input_shape,
            output_shape,
            n_tiles,
            n_classes)
        y = np.int32(np.argmax(output, axis=0))
        dice_coefs.append(
            dice_coefficients(y, label, labels=range(dataset["n_classes"]))
        )
    dice_coefs = np.array(dice_coefs)
    return np.mean(dice_coefs, axis=0)


def main():
    parser = argparse.ArgumentParser(description="train VoxResNet")
    parser.add_argument(
        "--iteration", "-i", default=10000, type=int,
        help="number of iterations, default=10000")
    parser.add_argument(
        "--monitor_step", "-s", default=1000, type=int,
        help="number of steps to monitor, default=1000")
    parser.add_argument(
        "--gpu", "-g", default=-1, type=int,
        help="negative value indicates no gpu, default=-1")
    parser.add_argument(
        "--input_file", "-f", type=str, default="dataset.json",
        help="json file of traininig dataset, default=dataset.json")
    parser.add_argument(
        "--validation_file", "-v", type=str,
        help="json file for validation dataset")
    parser.add_argument(
        "--n_batch", type=int, default=1,
        help="batch size, default=1")
    parser.add_argument(
        "--input_shape", type=int, nargs='*', action="store",
        default=[80, 80, 80],
        help="shape of input for the network, default=[80, 80, 80]")
    parser.add_argument(
        "--output_shape", type=int, nargs="*", action="store",
        default=[60, 60, 60],
        help="shape of output of the network, default=[60, 60, 60]")
    parser.add_argument(
        "--n_tiles", type=int, nargs="*", action="store",
        default=[5, 5, 5],
        help="number of tiles along each axis, default=[5, 5, 5]")
    parser.add_argument(
        '--out', '-o', default='vrn.npz', type=str,
        help='parameters of trained model, default=vrn.npz')
    parser.add_argument(
        "--learning_rate", "-r", default=1e-3, type=float,
        help="update rate, default=1e-3")
    parser.add_argument(
        "--weight_decay", "-w", default=0.0005, type=float,
        help="coefficient of l2norm weight penalty, default=0.0005")
    args = parser.parse_args()
    print(args)

    with open(args.input_file) as f:
        dataset = json.load(f)
    train_df = pd.DataFrame(dataset["data"])
    if args.validation_file is not None:
        with open(args.validation_file) as f:
            dataset_val = json.load(f)
        df_val = pd.DataFrame(dataset_val["data"])
        val_score = 0

    vrn = VoxResNet(dataset["in_channels"], dataset["n_classes"])
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        vrn.to_gpu()

    optimizer = chainer.optimizers.Adam(alpha=args.learning_rate)
    optimizer.use_cleargrads()
    optimizer.setup(vrn)
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))
    slices_in = [slice(None), slice(None)] + [
        slice((len_in - len_out) / 2, len_in - (len_in - len_out) / 2)
        for len_out, len_in in zip(args.output_shape, args.input_shape)
    ]
    for i in range(args.iteration):
        vrn.cleargrads()
        image, label = load_sample(
            train_df,
            args.n_batch,
            args.input_shape,
            args.output_shape
        )
        x_train = chainer.Variable(image)
        y_train = chainer.Variable(label)
        logits = vrn(x_train, train=True)
        loss = F.softmax_cross_entropy(logits[-1], y_train)
        for logit in logits[:-1]:
            loss += F.softmax_cross_entropy(logit, y_train)
        loss.backward()
        optimizer.update()
        if i % args.monitor_step == 0:
            accuracy = float(F.accuracy(logits[-1], y_train))
            print(
                f"step {i:5d}, accuracy {accuracy:.02f}, loss {float(loss.data):g}"
            )

            if args.validation_file is not None:
                dice_coefs = validate(
                    vrn,
                    df_val,
                    args.input_shape,
                    args.output_sape,
                    args.n_tiles,
                    args.n_classes
                )
                mean_dice_coefs = np.mean(dice_coefs)
                if mean_dice_coefs > val_score:
                    chainer.serializers.save_npz(args.out, vrn)
                    print(f"step {i:5d}, saved model")
                    val_score = mean_dice_coefs
                print(
                    f"step {i:5d}",
                    f"val/dice {mean_dice_coefs:.02f}",
                    *[f"val/dice{j} {dice:.02f}" for j, dice in enumerate(dice_coefs)],
                    sep=", "
                )

    if args.validation_file is not None:
        dice_coefs = validate(
            vrn,
            df_val,
            args.input_shape,
            args.output_shape,
            args.n_tiles,
            args.n_classes
        )
        mean_dice_coefs = np.mean(dice_coefs)
        if mean_dice_coefs > val_score:
            chainer.serializers.save_npz(args.out, vrn)
            print(f"step {args.iteration:5d}, saved model")
        print(
            f"step {i:5d}",
            f"val/dice {mean_dice_coefs:.02f}",
            *[f"val/dice{j} {dice:.02f}" for j, dice in enumerate(dice_coefs)],
            sep=", "
        )
    else:
        chainer.serializers.save_npz(args.out, vrn)
        print(f"step {args.iteration:5d}, saved model")


if __name__ == '__main__':
    main()
