import math

from ... import settings


def distributed_grid(shape):
    """Calculate a simple distribution of a matrix's elements over a grid.

    :param shape: the shape of your matrix.
    :return: grid tuple such as (4, 2, 1).
    """
    b = settings.block
    return math.ceil(shape[1] / b[1]), math.ceil(shape[0] / b[0]), 1


def distributed_vector(n_elements):
    """Calculate a simple distribution of a vector's elements over a flatten
    grid.

    :param n_elements: the number of elements contained in the vector.
    :return: grid tuple such as (4, 1, 1).
    """
    b = settings.block[0] * settings.block[1] * settings.block[2]
    return math.ceil(n_elements / b), 1, 1
