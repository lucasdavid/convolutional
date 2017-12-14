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
                 regularization=0.0, incremental=False, verbose=False):
        super(Convolutional, self).__init__(
            layers, epochs=epochs, n_batch=n_batch, eta=eta,
            regularization=regularization, incremental=incremental,
            verbose=verbose)
        self.initialize_random_weights()

        self.output_delta = None

    def add_conv_layer(self, tk, stride=(1, 1), padding=(1, 1)):
        self.layers.append((tk, stride, padding))
        self.n_layers = len(self.layers)

        return self

    def initialize_random_weights(self):
        self.weights = [op.scale(1 / np.sqrt(k[0] * k[1]), np.random.randn(*k))
                        for k, stride, padding in self.layers]
        self.biases = [np.random.randn(k[2], 1)
                       for k, stride, padding in self.layers]

        return self

    def feed_forward(self, X):
        """Return the output of the network if `a` is input."""
        y = X.reshape((-1, 28, 28, 1))

        transformed = []

        for sample in y:
            for k, W, b in zip(self.layers, self.weights, self.biases):
                sample = op.conv(sample, W, stride=k[1], padding=k[2])
                sample = op.add_bias(sample, b)
            transformed.append(sample)

        return np.array(transformed)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y=y, **fit_params).transform(X)

    def transform(self, X, y=None):
        return self.feed_forward(X)

    def fit(self, X, y=None, **fit_params):
        n_epochs = 1 if self.incremental else self.epochs
        n_samples = X.shape[0]

        for j in range(n_epochs):
            p = np.random.permutation(n_samples)
            X_batch, y_batch = X[p][:self.n_batch], y[p][:self.n_batch]

            self.SGD(X_batch, y_batch, n_samples)

        return self

    def back_propagation(self, x, y):
        """Propagate errors, computing the gradients of the weights and biases.
        """

        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # feedforward
        output = x.reshape(28, 28, -1)
        os = [output]

        for k, W, b in zip(self.layers, self.weights, self.biases):
            output = op.add_bias(
                op.conv(output, W, stride=k[1], padding=k[2]), b)
            os.append(output)

        # Backward error propagation.
        # Restore tensor to kernels shape.
        delta = self.output_delta.reshape((28, 28, -1))

        nabla_b[-1] = op.sum(op.sum(delta, axis=0), axis=0)
        nabla_w[-1] = np.rot90(op.conv(np.rot90(os[-2], k=2), delta), k=2)

        for l in range(2, self.n_layers + 1):
            w = self.weights[-l]
            delta = op.conv(delta, np.rot90(w, k=2))

            nabla_b[-l] = op.sum(op.sum(delta, axis=0), axis=0)
            nabla_w[-l] = np.rot90(op.conv(np.rot90(os[-l - 1], k=2), delta),
                                   k=2)

        return nabla_b, nabla_w, delta
