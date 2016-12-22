import chainer
import chainer.functions as F
import chainer.links as L


class VoxResNet(chainer.chain):
    """Voxel Residual Network"""
    def __init__(self, n_class=4):
        self.n_class = n_class
        super(VoxResNet, self).__init__(
            conv1a=L.ConvolutionND(
                ndim=3,
                in_channels=1,
                out_channels=32,
                ksize=3,
                stride=1,
                pad=1),
            bnorm1a=L.BatchNormalization(32),
            conv1b=L.ConvolutionND(
                ndim=3,
                in_channels=32,
                out_channels=32,
                ksize=3,
                stride=1,
                pad=1),
            bnorm1b=L.BatchNormalization(32),
            conv1c=L.ConvolutionND(
                ndim=3,
                in_channels=32,
                out_channels=64,
                ksize=3,
                stride=2,
                pad=1)
        )
        self.train = False

    def __call__(self, x):
        """
        calculate output of VoxResNet given input x

        Parameters
        ----------
        x
            image to perform semantic segmentation output

        Returns
        -------
        logit
            logit to be passed to softmax activation
        """
        h = F.relu(self.conv1a(x))
        h = self.bnorm1a(h, test=not self.train)
        return F.relu(self.conv1b(h))
