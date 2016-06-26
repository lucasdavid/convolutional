"""Network.

This is an adapted version of the code originally implemented by
Michael Nielsen, in 2016, and it is still available at this
[github](https://github.com/mnielsen/neural-networks-and-deep-learning)
repository.

"""

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.neural_network.multilayer_perceptron import (ACTIVATIONS,
                                                          DERIVATIVES)

from .base import NetworkBase
from .utils import costs
from .. import operators as op


class FullyConnected(NetworkBase, ClassifierMixin):
    def __init__(self, layers, epochs=20, n_batch=20, eta=.2,
                 regularization=0.0, cost=costs.CrossEntropyCost,
                 verbose=False):
        super(FullyConnected, self).__init__(
            layers, epochs=epochs, n_batch=n_batch, eta=eta,
            regularization=regularization, verbose=verbose)

        self.cost = cost
        self.initialize_random_weights()

    def initialize_random_weights(self):
        self.biases = [np.random.randn(y, 1) for y in self.layers[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.layers[:-1], self.layers[1:])]

    def predict(self, X):
        a = np.atleast_2d(X).transpose()
        for b, w in zip(self.biases, self.weights):
            a = ACTIVATIONS['logistic'](op.add(op.dot(w, a), b))

        return op.argmax(a, axis=0)

    def back_propagation(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        output = x.reshape(-1, 1)
        os = [output]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = op.add(op.dot(w, output), b)
            zs.append(z)
            output = ACTIVATIONS['logistic'](z)
            os.append(output)

        # backward pass
        delta = self.cost.delta(zs[-1], os[-1], y)
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
