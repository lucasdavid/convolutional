import numpy as np


def _pairwise_operation(op, a, b, out=None):
    assert a.shape == b.shape, ('Cannot apply operate on matrices with '
                                'incompatible shapes: %s and %s.' %
                                (a.shape, b.shape))
    if out is None:
        out = np.empty(a.shape)
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
    if out is None: out = np.zeros(shape)

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(a.shape[1]):
                out[i, j] += a[i, k] * b[k, j]

    return out


def scale(alpha, a, out=None):
    if not out:
        out = np.empty(a.shape)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            out[i][j] = alpha * a[i][j]

    return out


def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    if axis is None:
        out = 0

        for n in a.ravel():
            out += n

        return out

    if not -1 < axis < 2: raise ValueError('\'axis\' entry is out of bounds')

    n_elements = a.shape[1 - axis]
    if out is None: out = np.zeros(n_elements)
    assert out.shape in (n_elements, (n_elements,)), \
        'Invalid size for out. It should have exactly %i' % n_elements

    for i in range(n_elements):
        for j in range(a.shape[axis]):
            out[i] += a[j, i] if axis == 0 else a[i, j]

    return out if dtype is None else out.astype(dtype)


def conv(t, tk, stride=(1, 1), padding=(1, 1), out=None):
    n_channels = t.shape[2]
    n_kernels = tk.shape[2]

    if out is None:
        out = np.empty((t.shape[0], t.shape[1], n_kernels))

    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            for k in range(n_kernels):
                convolution = 0
                _i, _j = i - tk.shape[0] // 2, j - tk.shape[1] // 2

                for m in range(tk.shape[0]):
                    for n in range(tk.shape[1]):
                        for l in range(n_channels):
                            if -1 < _i + m < t.shape[0] and -1 < _j + n < \
                                    t.shape[1]:
                                convolution += t[_i + m, _j + n, l] * tk[
                                    m, n, k]

                out[i, j, k] = convolution
    return out


def add_bias(a, bias, out=None):
    a_3d = np.atleast_3d(a)
    bias = np.atleast_1d(bias)

    assert a_3d.shape[2] == bias.shape[0]

    if out is None: out = np.empty(a_3d.shape)

    for k in range(a_3d.shape[2]):
        # For each kernel
        for i in range(a_3d.shape[0]):
            for j in range(a_3d.shape[1]):
                out[i, j, k] = a_3d[i, j, k] + bias[k]

    return out.reshape(a.shape).astype(a.dtype)


def transpose(a, axes=None):
    if axes is not None:
        raise NotImplementedError

    out = np.empty(a.shape[1], a.shape[0])

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            out[j, i] = a[i, j]

    return out


operations = {
    'add': add,
    'sub': sub,
    'hadamard': hadamard,
    'dot': dot,
    'scale': scale,
    'sum': sum,
    'conv': conv,
    'add_bias': add_bias,
    'transpose': transpose,
}
