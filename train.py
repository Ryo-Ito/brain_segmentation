from __future__ import print_function
import argparse
import pandas as pd

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions

import load
from model import VoxResNet


def main():
    parser = argparse.ArgumentParser(description="train VoxResNet")
    parser.add_argument(
        "--epoch", "-e", default=int(1e2),
        help="number of sweeps over the dataset to train")
    parser.add_argument(
        "--gpu", "-g", default=-1,
        help="negative value indicates no gpu")
    parser.add_argument(
        "--input", type=str, default="dataset_train.csv",
        help="csv file containing filename of image and its label")
    parser.add_argument("--n_batch", type=int, default=1, help="batch size")
    parser.add_argument(
        "--input_shape", type=int, nargs='*', action="store",
        default=[80, 80, 80],
        help="shape of input for the network")
    parser.add_argument('--out', '-o', default='result',
        help='Directory to output the result')

    args = parser.parse_args()
    print(args)
    train_df = pd.read_csv(args.input)

    vrn = VoxResNet(n_classes=4)
    model = L.Classifier(vrn)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train = load.DatasetFromFiles(train_df, 4, args.input_shape)

    train_iter = chainer.iterators.SerialIterator(train, args.n_batch)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss']))
    trainer.run()


if __name__ == '__main__':
    main()
