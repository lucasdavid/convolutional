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
            Convolutional(connected_layers),
            FullyConnected(FullyConnected)
        )

    def fit(self, X, y=None, **fit_params):
        raise NotImplementedError

    def predict(self, X):
        for nn in self.networks[:-1]:
            X = nn.transform(X)

        return self.networks[-1].predict(X)
