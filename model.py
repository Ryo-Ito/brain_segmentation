import chainer
import chainer.functions as F
import chainer.links as L


class VoxResModule(chainer.Chain):
    """
    Voxel Residual Module
    input
    BatchNormalization, ReLU
    Conv 64, 3x3x3
    BatchNormalization, ReLU
    Conv 64, 3x3x3
    output
    """
    def __init__(self, in_channels):
        super(VoxResModule, self).__init__(
            bnorm1=L.BatchNormalization(size=in_channels),
            conv1=L.ConvolutionND(3, in_channels, 64, 3, pad=1),
            bnorm2=L.BatchNormalization(size=64),
            conv2=L.ConvolutionND(3, 64, 64, 3, pad=1))

    def __call__(self, x, train):
        h = F.relu(self.bnorm1(x, test=not train))
        h = self.conv1(h)
        h = F.relu(self.bnorm2(x, test=not train))
        h = self.conv2(h)
        return h + x


class VoxResNet(chainer.Chain):
    """Voxel Residual Network"""
    def __init__(self, n_classes=4):
        self.n_classes = n_classes
        super(VoxResNet, self).__init__(
            conv1a=L.ConvolutionND(3, 1, 32, 3, pad=1),
            bnorm1a=L.BatchNormalization(32),
            conv1b=L.ConvolutionND(3, 32, 32, 3, pad=1),
            bnorm1b=L.BatchNormalization(32),
            conv1c=L.ConvolutionND(3, 32, 64, 3, stride=2, pad=1),
            voxres2=VoxResModule(64),
            voxres3=VoxResModule(64),
            bnorm3=L.BatchNormalization(64),
            conv4=L.ConvolutionND(3, 64, 64, 3, stride=2, pad=1),
            voxres5=VoxResModule(64),
            voxres6=VoxResModule(64),
            bnorm6=L.BatchNormalization(64),
            conv7=L.ConvolutionND(3, 64, 64, 3, stride=2, pad=1),
            voxres8=VoxResModule(64),
            voxres9=VoxResModule(64),
            deconv1=L.DeconvolutionND(3, 32, 4, 3),
            deconv2=L.DeconvolutionND(3, 64, 4, 3),
            deconv4=L.DeconvolutionND(3, 64, 4, 3),
            deconv8=L.DeconvolutionND(3, 64, 4, 3)
        )
        self.train = False

    def __call__(self, x):
        """
        calculate output of VoxResNet given input x

        Parameters
        ----------
        x : [sample_size, 1, xlen, ylen, zlen]
            image to perform semantic segmentation

        Returns
        -------
        logit [sample_size, n_classes, xlen, ylen, zlen]
            logit to be passed to softmax activation
        """
        h = self.conv1a(x)
        h = F.relu(self.bnorm1a(h, test=not self.train))
        h = self.conv1b(h)
        c1 = self.deconv1(h)
        # h = F.relu(self.bnorm1b(h1, test=not self.train))
        # h = self.conv1c(h)
        # h = self.voxres2(h, self.train)
        # h2 = self.voxres3(h, self.train)
        # h = F.relu(self.bnorm3(h2, test=not self.train))
        # h = self.conv4(h)
        # h = self.voxres5(h, self.train)
        # h3 = self.voxres6(h, self.train)
        # h = F.relu(self.bnorm6(h3, test=not self.train))
        # h = self.conv7(h)
        # h = self.voxres8(h, self.train)
        # h4 = self.voxres9(h, self.train)
        return c1
        # return F.reshape(F.transpose(c1, (0, 2, 3, 4, 1)), (-1, self.n_classes))
