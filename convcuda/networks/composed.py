import numpy as np
from sklearn.base import ClassifierMixin

from .base import NetworkBase
from .convolutional import Convolutional
from .fully_connected import FullyConnected


class Composed(NetworkBase, ClassifierMixin):
    def __init__(self, convolutional_layers=(), connected_layers=(), epochs=20,
                 n_batch=20, eta=.2, regularization=0.0, verbose=False):
        super(Composed, self).__init__(
            layers=convolutional_layers + connected_layers, epochs=epochs,
            n_batch=n_batch, eta=eta, regularization=regularization,
            verbose=verbose)

        self.networks = (
            Convolutional(convolutional_layers, epochs=epochs, n_batch=n_batch,
                          eta=eta, regularization=regularization,
                          verbose=False, incremental=True),
            FullyConnected(connected_layers, epochs=epochs, n_batch=n_batch,
                           eta=eta, regularization=regularization,
                           verbose=False, incremental=True)
        )

        self.score_ = None
        self.score_history_ = []

    def fit(self, X, y=None, **fit_params):
        n_epochs = 1 if self.incremental else self.epochs
        n_samples = X.shape[0]
        cnn, fc = self.networks

        for j in range(n_epochs):
            self.score_history_ = 0
            p = np.random.permutation(n_samples)
            X_batch, y_batch = X[p][:self.n_batch], y[p][:self.n_batch]

            X_batch = cnn.transform(X_batch, y_batch)
            fc.fit(X_batch, y_batch)
            cnn.output_delta = fc.input_delta_
            cnn.fit(X_batch, y_batch)

            if self.verbose and (j % min(1, self.epochs // 10) == 0 or
                                         j == self.epochs - 1):
                # If verbose and epoch is dividable by 10 or
                # if it's the last one.
                print("[%i], loss: %.2f" % (j, self.score_ / self.n_batch))

        return self

    def predict(self, X):
        for nn in self.networks[:-1]:
            X = nn.transform(X)

        return self.networks[-1].predict(X)

    def calculate_score(self, X, y):
        """Compute loss as defined by the cost function."""
        self.score_ = self.score(X, y)
        self.score_history_.append(self.score_)

        return self.score_
