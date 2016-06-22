import numpy as np


def hadamard(a, b, out=None):
    if out:
        out[:] = a * b
    else:
        out = a * b
    return out


def scale(alpha, a, out=None):
    return hadamard(alpha, a, out=out)


def add_bias(a, bias, out=None):
    return a + bias


def conv(t, tk, stride=(1, 1), padding=(1, 1), out=None):
    n_channels = t.shape[2]
    n_kernels = tk.shape[2]

    if out is None: out = np.empty((t.shape[0], t.shape[1], n_kernels))

    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            for k in range(n_kernels):
                _i, _j = i - tk.shape[0] // 2, j - tk.shape[1] // 2

                out[i, j, k] = sum(t[_i + m, _j + n, l] * tk[m, n, k]
                                   for n in range(tk.shape[1])
                                       for m in range(tk.shape[0])
                                       if -1 < _i + m < t.shape[0] and -1 < _j + n < t.shape[1]
                                           for l in range(n_channels))
    return out


operations = {
    'add': np.add,
    'sub': np.subtract,
    'dot': np.dot,
    'hadamard': hadamard,
    'scale': scale,
    'sum': np.sum,
    'conv': conv,
    'add_bias': add_bias,
    'transpose': np.transpose,
}
