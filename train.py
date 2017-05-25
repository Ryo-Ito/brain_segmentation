import argparse
import chainer
import chainer.functions as F
import json
import numpy as np
import pandas as pd

import load
from model import VoxResNet


def main():
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
        "--n_batch", type=int, default=1,
        help="batch size, default=1")
    parser.add_argument(
        "--shape", type=int, nargs='*', action="store",
        default=[80, 80, 80],
        help="shape of input for the network, default=[80, 80, 80]")
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

    vrn = VoxResNet(dataset["in_channels"], dataset["n_classes"])
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        vrn.to_gpu()
        xp = chainer.cuda.cupy
    else:
        xp = np

    optimizer = chainer.optimizers.Adam(alpha=args.learning_rate)
    optimizer.use_cleargrads()
    optimizer.setup(vrn)
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

    for i in range(args.iteration):
        vrn.cleargrads()
        image, label = load.sample(train_df, args.n_batch, args.shape)
        x_train = xp.asarray(image)
        y_train = xp.asarray(label)
        logits = vrn(x_train, train=True)
        loss = 0
        for logit in logits:
            loss += F.softmax_cross_entropy(logit, y_train)
        loss.backward()
        optimizer.update()
        if i % args.display_step == 0:
            accuracy = [float(F.accuracy(logit, y_train).data) for logit in logits]
            print("step {0:5d}, acc_c1 {1[0]:.02f}, acc_c2 {1[1]:.02f}, acc_c3 {1[2]:.02f}, acc_c4 {1[3]:.02f}, acc {1[4]:.02f}".format(i, accuracy))

    vrn.to_cpu()
    chainer.serializers.save_npz(args.out, vrn)


if __name__ == '__main__':
    main()
