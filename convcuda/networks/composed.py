from .base import NetworkBase



class Composed(NetworkBase):
    def __init__(self, *networks):
        super(Composed, self).__init__([])
        self.networks = networks

    def feedforward(self, X):
        y = X

        for network in self.networks:
            y = network.feedforward(y)

        return y

    def fit(self, training_data, epochs, mini_batch_size, eta,
            lmbda=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        raise NotImplementedError
