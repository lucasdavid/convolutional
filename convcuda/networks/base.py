import json

import numpy as np
from sklearn.base import BaseEstimator

from .. import op


class NetworkBase(BaseEstimator):
    def __init__(self, layers, epochs=20, n_batch=20, eta=.2,
                 regularization=0.0, verbose=False):
        self.layers = layers
        self.n_layers = len(layers)

        self.epochs = epochs
        self.n_batch = n_batch
        self.eta = eta
        self.regularization = regularization
        self.verbose = verbose

        self.weights = []
        self.biases = []

        self.loss_ = None

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {
            'layers': self.layers,
            'epochs': self.epochs,
            'n_batch': self.n_batch,
            'eta': self.eta,
            'regularization': self.regularization,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
        }

        with open(filename, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filename):
        """Load a neural network from the file ``filename`` and return an
        instance built from those parameters.
        """
        with open(filename) as f:
            data = json.load(f)

        nn = cls(**data)
        nn.weights = [np.array(w) for w in data["weights"]]
        nn.biases = [np.array(b) for b in data["biases"]]
        return nn

    def fit(self, X, y=None, **fit_params):
        n_samples = X.shape[0]

        for j in range(self.epochs):
            self.loss_ = 0
            p = np.random.permutation(n_samples)
            X_batch, y_batch = X[p][:self.n_batch], y[p][:self.n_batch]

            self.SGD(X_batch, y_batch, n_samples)

            if self.verbose and (j % (self.epochs // 10) == 0 or
                                         j == self.epochs - 1):
                # If verbose and epoch is dividable by 10 or
                # if it's the last one.
                print("[%i], loss: %.2f" % (j, self.loss_ / self.n_batch))

        return self

    def SGD(self, X, labels, n_samples):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.
        """
        batch_size = X.shape[0]

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in zip(X, labels):
            delta_nabla_b, delta_nabla_w = self.back_propagation(x, y)
            nabla_b = [op.add(nb, dnb) for nb, dnb in
                       zip(nabla_b, delta_nabla_b)]
            nabla_w = [op.add(nw, dnw) for nw, dnw in
                       zip(nabla_w, delta_nabla_w)]
        self.weights = [
            (op.scale((1 - self.eta * (self.regularization / n_samples)), w) -
             op.scale(self.eta / batch_size, nw))
            for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - op.scale(self.eta / batch_size, nb)
                       for b, nb in zip(self.biases, nabla_b)]
