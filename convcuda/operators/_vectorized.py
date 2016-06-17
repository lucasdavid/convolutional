import numpy as np


def hadamard(a, b, out=None):
    if out:
        out[:] = a * b
    else:
        out = a * b
    return out


def scale(alpha, a, out=None):
    if out:
        out[:] = alpha * a
    else:
        out = alpha * a
    return out


operations = {
    'add': np.add,
    'sub': np.subtract,
    'dot': np.dot,
    'hadamard': hadamard,
    'scale': scale,
    'sum': np.sum,
}
