from __future__ import print_function
import argparse

import chainer
from chainer import training
from chainer.training import extensions


parser = argparse.ArgumentParser(description="train VoxResNet")
parser.add_argument(
    "--input", type=str,
    help="csv file containing filename of image and its label")
parser.add_argument("--n_batch", type=int, default=1, help="batch size")
parser.add_argument(
    "--input_shape", type=int, nargs='*', action="store",
    default=[100, 100, 100],
    help="shape of input for the network")
parser.add_argument(
    "--l2_penalty", type=float, default=1.,
    help="coefficient of l2 regularization of the weights")
parser.add_argument("--output", type=str, help="trained model")
parser.add_argument(
    "--learning_rate", type=float, default=1e-4,
    help="learning rate of gradient methods")
parser.add_argument(
    "--n_iter", type=int, default=10000,
    help="number of iterations of gradient methods")
parser.add_argument(
    "--display_step", type=int, default=1000,
    help="number of steps to show the progress")

args = parser.parse_args()


def main():
    train_df = pd.read_csv(args.input)
