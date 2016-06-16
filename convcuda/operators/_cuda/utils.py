import math

from . import settings


def distributed_grid(shape):
    """Compute Standard Grid for Matrices with Defined Shape.

    :param shape: the shape of your matrix.
    :return: grid tuple such as (4, 2, 1).
    """
    b = settings.block
    return math.ceil(shape[0] / b[0]), math.ceil(shape[1] / b[1]), 1
