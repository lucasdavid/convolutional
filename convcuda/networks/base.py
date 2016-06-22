import json

import numpy as np
from sklearn.base import BaseEstimator


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
