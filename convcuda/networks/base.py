import json

import numpy as np
from sklearn.base import BaseEstimator

from .. import op


class NetworkBase(BaseEstimator):
    def __init__(self, layers, epochs=20, n_batch=20, eta=.2,
                 regularization=0.0, incremental=False, verbose=False):
        self.layers = layers
        self.n_layers = len(layers)

        self.epochs = epochs
        self.n_batch = n_batch
        self.eta = eta
        self.regularization = regularization
        self.incremental = incremental
        self.verbose = verbose

        self.weights = []
        self.biases = []

        self.score_ = None
        self.score_history_ = []

        self.input_delta_ = None

    def SGD(self, X, labels, n_samples):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.
        """
        batch_size = X.shape[0]

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        delta = np.zeros((np.prod(X.shape) // X.shape[0], 1))

        for x, y in zip(X, labels):
            delta_nabla_b, delta_nabla_w, _delta = self.back_propagation(x, y)
            nabla_b = [op.add(nb, dnb) for nb, dnb in
                       zip(nabla_b, delta_nabla_b)]
            nabla_w = [op.add(nw, dnw) for nw, dnw in
                       zip(nabla_w, delta_nabla_w)]

            delta = op.add(delta, _delta)

        self.weights = [
            (op.scale((1 - self.eta * (self.regularization / n_samples)), w) -
             op.scale(self.eta / batch_size, nw))
            for w, nw in zip(self.weights, nabla_w)]
        self.biases = [op.sub(b, op.scale(self.eta / batch_size, nb))
                       for b, nb in zip(self.biases, nabla_b)]

        return delta

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
