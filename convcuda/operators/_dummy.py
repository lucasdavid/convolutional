import numpy as np


def _pairwise_operation(op, a, b, out=None):
    assert a.shape == b.shape, ('Cannot apply operate on matrices with '
                                'incompatible shapes: %s and %s.' %
                                (a.shape, b.shape))
    out = out or np.empty(a.shape)
    shape = a.shape

    for i in range(shape[0]):
        for j in range(shape[1]):
            out[i][j] = op(a[i][j], b[i][j])

    return out


def add(a, b, out=None):
    return _pairwise_operation(lambda x, y: x + y, a, b, out=out)


def sub(a, b, out=None):
    return _pairwise_operation(lambda x, y: x - y, a, b, out=out)


def hadamard(a, b, out=None):
    return _pairwise_operation(lambda x, y: x * y, a, b, out=out)


def dot(a, b, out=None):
    assert a.shape[1] == b.shape[0], \
        ('Cannot apply dot operator on matrices. '
         'Incompatible shapes: %s and %s.' % (a.shape, b.shape))

    shape = a.shape[0], b.shape[1]
    out = out or np.zeros(shape)

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(a.shape[1]):
                out += a[i][k] * b[k][j]

    return out


def scale(alpha, a, out=None):
    if not out:
        out = np.empty(a.shape)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            out[i][j] = alpha * a[i][j]

    return out


operations = {
    'add': add,
    'sub': sub,
    'hadamard': hadamard,
    'dot': dot,
    'scale': scale,
}
