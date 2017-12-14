import os

import numpy as np
from pycuda import autoinit
from pycuda import compiler
from pycuda.driver import In, Out

from . import utils
from ... import settings

# Auto-initiates pyCUDA driver.
autoinit


def _load_kernels():
    """Load All Kernels.

    :return: a list of compiled kernels.
    """
    kernel_map = {}
    kernels_folder = os.path.dirname(os.path.realpath(__file__))
    kernels_folder = os.path.join(kernels_folder, 'kernels')

    for k_name in os.listdir(kernels_folder):
        kernel_file_path = os.path.join(kernels_folder, k_name)

        with open(kernel_file_path, 'r') as f:
            kernel = f.read()

        # Replace templates with defined settings.
        kernel = kernel % {
            'N_THREADS_0': settings.block[0],
            'N_THREADS_1': settings.block[1],
            'N_THREADS_2': settings.block[2],
        }

        # Compile kernel and add it to the kernel map.
        kernel_map[k_name] = compiler.SourceModule(kernel)

    return kernel_map


kernels = _load_kernels()


def _pairwise_operation(k, a, b, out=None):
    """Perform a kernel assuming its a pairwise operation."""
    original_type = a.dtype
    original_shape = a.shape

    a, b = a.ravel(), b.ravel()

    assert a.shape == b.shape, \
        ('Cannot apply %s on matrices. Incompatible shapes: %s and %s.' %
         (k, a.shape, b.shape))

    a, b = a.astype(np.float32), b.astype(np.float32)

    if not out: out = np.empty(a.shape, dtype=np.float32)

    op = kernels[k + '.cu'].get_function(k)
    op(In(a), In(b), Out(out),
       np.int32(a.shape[0]),
       grid=utils.distributed_flatten_grid(a.shape), block=settings.block)

    return out.astype(original_type).reshape(original_shape)


def add(a, b, out=None):
    """Compute the Addition of Two Matrices."""
    return _pairwise_operation('mat_add', a, b, out=out)


def sub(a, b, out=None):
    """Compute the Subtraction of Two Matrices."""
    return _pairwise_operation('mat_sub', a, b, out=out)


def hadamard(a, b, out=None):
    """Compute the Hadamard Product of Two Matrices."""
    return _pairwise_operation('mat_hadamard', a, b, out=out)


def dot(a, b, out=None):
    """Compute the Dot Product Between Two Matrices."""
    original_type = a.dtype

    # Convert arrays (10,) -> (10, 1).
    a = np.atleast_2d(a.astype(np.float32))
    b = np.atleast_2d(b.astype(np.float32))

    assert a.shape[1] == b.shape[0], ('Cannot apply dot operator on matrices.'
                                      ' Incompatible shapes: %s and %s.'
                                      % (a.shape, b.shape))
    shape = (a.shape[0], b.shape[1])
    if not out: out = np.empty(shape).astype(np.float32)

    op = kernels['mat_dot.cu'].get_function('mat_dot')
    op(In(a), In(b), Out(out),
       np.int32(a.shape[0]), np.int32(a.shape[1]),
       np.int32(b.shape[0]), np.int32(b.shape[1]),
       grid=utils.distributed_grid(shape), block=settings.block)

    return out.astype(original_type)


def scale(alpha, a, out=None):
    original_shape = a.shape

    if not out: out = np.empty(a.shape, dtype=np.float32)

    if len(a.shape) == 1 or a.shape[0] == 1 or a.shape[1] == 1:
        shape = (1, max(a.shape), 1)
        g = utils.distributed_flatten_grid(shape[1])

        max_threads = settings.block[0] * settings.block[1] * settings.block[2]
        b = (min(shape[1], max_threads), 1, 1)
    else:
        a = np.atleast_3d(a)
        shape = a.shape
        g = utils.distributed_grid(a.shape)
        b = settings.block

    op = kernels['mat_scale.cu'].get_function('mat_scale')
    op(np.float32(alpha), In(a.astype(np.float32)), Out(out),
       np.int32(shape[0]), np.int32(shape[1]), np.int32(shape[2]),
       grid=g, block=b)

    return out.reshape(original_shape).astype(a.dtype)


def add_bias(a, bias, out=None):
    a_3d = np.atleast_3d(a).astype(np.float32)
    bias = np.atleast_1d(bias).astype(np.float32)

    # Assert biases count is equal to kernels count.
    assert bias.shape[0] == a_3d.shape[2]

    if not out: out = np.empty(a_3d.shape, dtype=np.float32)

    op = kernels['add_bias.cu'].get_function('add_bias')
    g = utils.distributed_grid(a_3d.shape)
    op(In(a_3d), In(bias), Out(out),
       np.int32(a_3d.shape[0]), np.int32(a_3d.shape[1]),
       np.int32(a_3d.shape[2]),
       grid=g, block=settings.block)

    return out.reshape(a.shape).astype(a.dtype)


def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    if axis is not None:
        raise NotImplementedError

    # TODO: Find what's causing the incorrect sum of the values.
    b = a.ravel()
    if out is None: out = np.zeros((1,), dtype=np.float32)

    g = utils.distributed_flatten_grid(b.shape[0])
    block = utils.distributed_flatten_block(b.shape[0])

    op = kernels['t_sum.cu'].get_function('t_sum')
    op(In(b), Out(out), np.int32(b.shape[0]),
       grid=g, block=block)

    return out.astype(a.dtype)[0]


def conv(t, tk, stride=(1, 1), padding=(1, 1), out=None):
    """Compute the convolution between two tensors of rank 3."""
    t, tk = np.atleast_3d(t), np.atleast_3d(tk)

    # Output has the entering tensor `t`'s width and height,
    # and channel count equal to the number of kernels in `tk`.
    shape = t.shape[:2] + tk.shape[-1:]

    if not out: out = np.empty(shape).astype(np.float32)
    assert out.shape == shape

    op = kernels['conv.cu'].get_function('conv')
    op(In(t.astype(np.float32)), In(tk.astype(np.float32)), Out(out),
       *np.int32(t.shape), *np.int32(tk.shape),
       grid=utils.distributed_grid(shape), block=settings.block)

    return out.astype(t.dtype)


def transpose(a, axes=None):
    if axes is not None:
        raise NotImplementedError

    out = np.empty(a.shape[1], a.shape[0])

    op = kernels['mat_transpose.cu'].get_function('mat_transpose')
    op(In(a), Out(out),
       np.int32(a.shape[0]), np.int32(a.shape[1]),
       grid=utils.distributed_grid(a.shape), block=settings.block)

    return out


operations = {
    'add': add,
    'sub': sub,
    'dot': dot,
    'hadamard': hadamard,
    'scale': scale,
    'sum': sum,
    'conv': conv,
    'add_bias': add_bias,
    'transpose': transpose,
}
