"""Network Costs

Defines cost functions for optimizers used in NN trainining procedures.
"""

import numpy as np
from .one_hot import one_hot_encoding

from sklearn.neural_network.multilayer_perceptron import DERIVATIVES


class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``."""
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(a, y):
        """Return the error delta from the output layer."""
        return (a - y) * DERIVATIVES['logistic'](a)


class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        encoded_y = one_hot_encoding(y, n_classes=10)
        return np.sum(
            np.nan_to_num(-y * np.log(a) - (1 - encoded_y) * np.log(1 - a)))

    @staticmethod
    def delta(a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        encoded = one_hot_encoding(y, n_classes=10)
        b = a - encoded
        return b
