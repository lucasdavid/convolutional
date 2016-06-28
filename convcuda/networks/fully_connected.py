"""Fully Connected Network.

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
    """Fully Connected Multilayer Perceptron."""

    def __init__(self, layers, epochs=20, n_batch=20, eta=.2,
                 regularization=0.0, incremental=False,
                 cost=costs.CrossEntropyCost,
                 verbose=False):
        super(FullyConnected, self).__init__(
            layers, epochs=epochs, n_batch=n_batch, eta=eta,
            regularization=regularization, incremental=incremental,
            verbose=verbose)

        self.cost = cost
        self.initialize_random_weights()

    def initialize_random_weights(self):
        self.biases = [np.random.randn(y, 1) for y in self.layers[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.layers[:-1], self.layers[1:])]

    def fit(self, X, y=None, **fit_params):
        n_epochs = 1 if self.incremental else self.epochs
        n_samples = X.shape[0]
        first_layer_delta = np.zeros((self.layers[0], 1))

        for j in range(n_epochs):
            p = np.random.permutation(n_samples)
            X_batch, y_batch = X[p][:self.n_batch], y[p][:self.n_batch]

            delta = self.SGD(X_batch, y_batch, n_samples)
            op.add(first_layer_delta, delta)

            if 'validation_data' in fit_params:
                self.calculate_score(*fit_params['validation_data'])

                if (self.verbose and
                    (j % max(1, self.epochs // 10) == 0 or
                     j == self.epochs - 1)):

                    self.calculate_score(*fit_params['validation_data'])
                    print("[%i], score: %.2f" % (j, self.score_))

        first_layer_delta = op.scale(1 / (n_epochs * self.n_batch),
                                     first_layer_delta)
        self.input_delta_ = first_layer_delta

        return self

    def calculate_score(self, X, y):
        """Compute loss as defined by the cost function."""
        self.score_ = self.score(X, y)
        self.score_history_.append(self.score_)

        return self.score_

    def feed_forward(self, X):
        os = []

        for x in X:
            x = x.reshape(-1, 1)
            for b, w in zip(self.biases, self.weights):
                x = ACTIVATIONS['logistic'](op.add(op.dot(w, x), b))

            os.append(x)

        return np.array(os)

    def predict(self, X):
        A = self.feed_forward(X).reshape(X.shape[0], -1)
        return op.argmax(A, axis=1)

    def back_propagation(self, x, y):
        """Propagate errors, computing the gradients of the weights and biases.

        :return: tuple `(nabla_b, nabla_w, delta)`

        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feed-forward inputs.
        output = x.reshape(-1, 1)
        os = [output]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = op.add(op.dot(w, output), b)
            zs.append(z)
            output = ACTIVATIONS['logistic'](z)
            os.append(output)

        # Backward error propagation itself.
        delta = self.cost.delta(os[-1], y)

        nabla_b[-1] = delta
        nabla_w[-1] = op.dot(delta, os[-2].transpose())

        for l in range(2, self.n_layers):
            o = os[-l]
            sp = DERIVATIVES['logistic'](o)
            d = op.dot(self.weights[-l + 1].transpose(), delta)
            delta = op.hadamard(sp, d)
            nabla_b[-l] = delta
            a = os[-l - 1].transpose()
            nabla_w[-l] = op.dot(delta, a)

        return nabla_b, nabla_w, op.dot(self.weights[0].transpose(), delta)
