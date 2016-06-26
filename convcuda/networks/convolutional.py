import numpy as np
from sklearn.base import TransformerMixin

from .base import NetworkBase
from .. import op


class Convolutional(NetworkBase, TransformerMixin):
    """Convolutional Neural Networks.

    :param layers: list of tensors, where each element represents a tuple
        containing a kernel shape, the stride taken and the padding.

    Examples:
        >>> cnn = Convolutional([
        ...     [3, 3, 10], [2, 2], [4, 4],
        ...     [3, 3, 100], [2, 2], [4, 4],
        ... ])
        >>> cnn.fit_transform(X)
    """

    def __init__(self, layers, epochs=20, n_batch=20, eta=.2,
                 regularization=0.0, verbose=False):
        super(Convolutional, self).__init__(
            layers, epochs=epochs, n_batch=n_batch, eta=eta,
            regularization=regularization, verbose=verbose)
        self.initialize_random_weights()

    def add_conv_layer(self, tk, stride=(1, 1), padding=(1, 1)):
        self.layers.append((tk, stride, padding))
        self.n_layers = len(self.layers)

        return self

    def initialize_random_weights(self):
        self.weights = [op.scale(1 / np.sqrt(k[0] * k[1]), np.random.randn(*k))
                        for k, stride, padding in self.layers]
        self.biases = [np.random.randn(k[2], 1)
                       for k, stride, padding in self.layers]

    def feed_forward(self, X):
        """Return the output of the network if `a` is input."""
        y = X

        for k, W, b in zip(self.layers, self.weights, self.biases):
            y = op.add_bias(op.conv(y, W, stride=k[1], padding=k[2]), b)

        return y

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y=y, **fit_params).transform(X)

    def transform(self, X, y=None):
        return self.feed_forward(X)

    def back_propagation(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``.
        """

        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # feedforward
        output = x.reshape(28, 28)
        os = [output]

        for k, W, b in zip(self.layers, self.weights, self.biases):
            z = op.conv(output, W, stride=k[1], padding=k[2])
            z = op.add_bias(z, b)
            output = z
            os.append(output)

        # backward pass
        delta = self.cost.delta(os[-1], os[-1], y)
        self.loss_ += np.sum(np.abs(delta))

        nabla_b[-1] = delta
        nabla_w[-1] = op.dot(delta, os[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.n_layers):
            o = os[-l]
            sp = DERIVATIVES['logistic'](o)
            d = op.dot(self.weights[-l + 1].transpose(), delta)
            delta = op.hadamard(sp, d)
            nabla_b[-l] = delta
            nabla_w[-l] = op.dot(delta, os[-l - 1].transpose())
        return nabla_b, nabla_w
