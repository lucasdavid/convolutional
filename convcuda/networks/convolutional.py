import numpy as np
from sklearn.base import TransformerMixin

from .. import operators as op


class Convolutional(TransformerMixin):
    def __init__(self, kernels=()):
        self.kernels = kernels
        self.n_layers = len(kernels)

        self.weights = self.biases = None

    def add_conv_layer(self, tk, stride=(1, 1), padding=(1, 1)):
        self.kernels.append((tk, stride, padding))

    def random_weights(self):
        self.weights = [op.scale(1 / np.sqrt(k[0] * k[1]), np.random.randn(*k))
                        for k, stride, padding in self.kernels]
        self.biases = [np.random.randn(k[2], 1)
                       for k, stride, padding in self.kernels]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        y = a

        for k, W, b in zip(self.kernels, self.weights, self.biases):
            y = op.add_bias(op.conv(y, W, stride=k[1], padding=k[2]), b)

        return y
