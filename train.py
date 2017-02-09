from __future__ import print_function
import argparse
import pandas as pd
import numpy as np

import chainer
from chainer import cuda, Variable
import chainer.functions as F

import load
from model import VoxResNet


def main():
    parser = argparse.ArgumentParser(description="train VoxResNet")
    parser.add_argument(
        "--iteration", "-i", default=100000, type=int,
        help="number of iterations, default=100000")
    parser.add_argument(
        "--display_step", "-s", default=1000, type=int,
        help="number of steps to display, default=1000")
    parser.add_argument(
        "--gpu", "-g", default=-1, type=int,
        help="negative value indicates no gpu, default=-1")
    parser.add_argument(
        "--input_file", "-f", type=str, default="dataset_train.csv",
        help="csv file containing filename of image and its label, default=dataset_train.csv")
    parser.add_argument(
        "--n_batch", type=int, default=1,
        help="batch size, default=1")
    parser.add_argument(
        "--shape", type=int, nargs='*', action="store",
        default=[80, 80, 80],
        help="shape of input for the network, default=[80, 80, 80]")
    parser.add_argument(
        '--out', '-o', default='vrn.npz', type=str,
        help='trained model, default=vrn.npz')
    parser.add_argument(
        "--learning_rate", "-r", default=1e-3, type=float,
        help="update rate, default=1e-3")
    parser.add_argument(
        "--weight_decay", "-w", default=0.0005, type=float,
        help="coefficient of l2norm weight penalty, default=0.0005")

    args = parser.parse_args()
    print(args)
    train_df = pd.read_csv(args.input_file)

    vrn = VoxResNet(in_channels=2, n_classes=4)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        vrn.to_gpu()
    xp = cuda.cupy if args.gpu >= 0 else np

    optimizer = chainer.optimizers.Adam(alpha=args.learning_rate)
    optimizer.use_cleargrads()
    optimizer.setup(vrn)
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

    for i in range(args.iteration):
        vrn.cleargrads()
        scalar_img, label_img = load.sample(train_df, args.n_batch, args.shape)
        assert label_img.shape == (args.n_batch,) + tuple(args.shape), label_img.shape
        x_train = Variable(xp.asarray(scalar_img))
        y_train = Variable(xp.asarray(label_img))
        outputs = vrn(x_train, train=True)
        loss = 0
        for out in outputs:
            loss += F.softmax_cross_entropy(out, y_train)
        accuracy_c1 = F.accuracy(outputs[0], y_train)
        accuracy = F.accuracy(outputs[-1], y_train)
        loss.backward()
        optimizer.update()
        if i % args.display_step == 0:
            print("step %5d, accuracy_c1 %.02f, accuracy %.02f, cost %g" % (i, accuracy_c1.data, accuracy.data, loss.data))

    vrn.to_cpu()
    chainer.serializers.save_npz(args.out, vrn)


if __name__ == '__main__':
    main()
