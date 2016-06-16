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


def _cut(t, interval_x, interval_y, channel):
    out = np.empty(t.shape[:2])
    for i in range(*interval_x):
        for j in range(*interval_y):
            if i < 0 or i >= t.shape[0] or j < 0 or j >= t.shape[1]:
                out[i][j] = 0
            else:
                out[i][j] = t[i][j][channel]
    return out


def sum(a):
    out = 0
    for n in a.ravel():
        out += n
    return out


def conv(t, tk, stride=(1, 1), padding=(1, 1)):
    n_kernels = tk.shape[3]
    n_channels = t.shape[2]
    kernel_size = tk.shape[:2]
    activations = np.empty(t.shape[:2])
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            for k in range(n_kernels):
                kernel = tk[:, :, :, k]
                kernel = kernel.reshape(kernel.shape[:2])
                kernel_sum = 0
                for l in range(n_channels):
                    st = _cut(t,
                              (i - (kernel_size[0] - 1) // 2,
                               i + (kernel_size[0] - 1) // 2),
                              (j - (kernel_size[1] - 1) // 2,
                               j + (kernel_size[1] - 1) // 2), l)
                    kernel_sum += sum(hadamard(st, kernel, st))
                activations[i][j] = kernel_sum
    return activations


operations = {
    'add': add,
    'sub': sub,
    'hadamard': hadamard,
    'dot': dot,
    'scale': scale,
    'conv': conv,
    'sum': sum,
}
