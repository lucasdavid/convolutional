import numpy as np


def one_hot_encoding(y, n_classes=None):
    """Return a n_classes-dimensional unit vector with a 1.0 in the
    j'th position and zeroes elsewhere.
    This is used to convert a digit (0...9) into a corresponding desired
    output from the neural network.

    :param y: 1-ranked tensor.
        The labels associated to each sample.

    :param n_classes: int.
        The counting of all possible classes in the data set.

    """
    if n_classes is None: n_classes = np.max(y) + 1
    return np.eye(n_classes)[y].reshape(-1, 1)
