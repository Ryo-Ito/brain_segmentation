from __future__ import print_function
import argparse
import itertools
import json

import chainer
import chainer.functions as F
import numpy as np
import pandas as pd

from model import VoxResNet
from utils import load_sample, load_nifti, crop_patch, dice_coefficients


parser = argparse.ArgumentParser(description="train VoxResNet")
parser.add_argument(
    "--iteration", "-i", default=10000, type=int,
    help="number of iterations, default=10000")
parser.add_argument(
    "--display_step", "-s", default=1000, type=int,
    help="number of steps to display, default=1000")
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
    chainer.cuda.get_device(args.gpu).use()
    vrn.to_gpu()
    xp = chainer.cuda.cupy
else:
    xp = np


def validate(model):
    dice_coefs = []
    for image_path, label_path in zip(df_val["image"], df_val["label"]):
        image = load_nifti(image_path)
        label = load_nifti(label_path)
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
            output[slice(None), slices_out[0], slices_out[1], slices_out[2]] += chainer.cuda.to_cpu(model(patch).data[0, slice(None), slices_in[0], slices_in[1], slices_in[2]])
        y = np.argmax(output, axis=0).astype(np.int32)
        dice_coefs.append(dice_coefficients(y, label))
    dice_coefs = np.array(dice_coefs)
    return np.mean(dice_coefs, axis=0)


optimizer = chainer.optimizers.Adam(alpha=args.learning_rate)
optimizer.use_cleargrads()
optimizer.setup(vrn)
optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))
slices_in = [slice(None), slice(None)] + [slice((len_in - len_out) / 2, len_in - (len_in - len_out) / 2) for len_out, len_in, in zip(args.output_shape, args.input_shape)]
for i in range(args.iteration):
    vrn.cleargrads()
    image, label = load_sample(train_df, args.n_batch, args.input_shape, args.output_shape)
    x_train = xp.asarray(image)
    y_train = xp.asarray(label)
    logits = vrn(x_train, train=True)
    logits = [logit[slices_in] for logit in logits]
    loss = F.softmax_cross_entropy(logits[-1], y_train)
    for logit in logits[:-1]:
        loss += F.softmax_cross_entropy(logit, y_train)
    loss.backward()
    optimizer.update()
    if i % args.display_step == 0:
        accuracy = [float(F.accuracy(logit, y_train).data) for logit in logits]
        print("step {0:5d}, acc_c1 {1[0]:.02f}, acc_c2 {1[1]:.02f}, acc_c3 {1[2]:.02f}, acc_c4 {1[3]:.02f}, acc {1[4]:.02f}, loss {2:g}".format(i, accuracy, float(loss.data)))
        if args.validation_file is not None:
            dice_coefs = validate(vrn)
            mean_dice_coefs = np.mean(dice_coefs)
            if mean_dice_coefs > val_score:
                chainer.serializers.save_npz(args.out, vrn)
                print("step {:5d}, saved model".format(i))
                val_score = mean_dice_coefs
            print("step {:5d}".format(i), "val/dice {:.02f}".format(mean_dice_coefs), *["val/dice{} {:.02f}".format(j, dice) for j, dice in enumerate(dice_coefs)], sep=", ")

if args.validation_file is not None:
    dice_coefs = validate(vrn)
    mean_dice_coefs = np.mean(dice_coefs)
    if mean_dice_coefs > val_score:
        chainer.serializers.save_npz(args.out, vrn)
        print("step {:5d}, saved model".format(args.iteration))
    print("step {:5d}".format(args.iteration), "val/dice {:.02f}".format(mean_dice_coefs), *["val/dice{} {:.02f}".format(j, dice) for j, dice in enumerate(dice_coefs)], sep=", ")
else:
    chainer.serializers.save_npz(args.out, vrn)
    print("step {:5d}, saved model".format(args.iteration))
