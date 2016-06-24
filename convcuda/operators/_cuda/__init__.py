import os

import numpy as np
import pycuda.autoinit
from pycuda import compiler
from pycuda.driver import In, Out
from . import utils
from ... import settings


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
    assert a.shape == b.shape, \
        ('Cannot apply %s on matrices. Incompatible shapes: %s and %s.' %
         (k, a.shape, b.shape))

    original_type = a.dtype
    a, b = a.astype(np.float32), b.astype(np.float32)

    if not out: out = np.empty(a.shape, dtype=np.float32)

    shape = out.shape
    if len(shape) == 1: shape = (1, shape[0])

    op = kernels[k + '.cu'].get_function(k)
    op(In(a), In(b), Out(out),
       np.int32(shape[0]), np.int32(shape[1]),
       grid=utils.distributed_grid(shape), block=settings.block)

    return out.astype(original_type)


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
    if not out: out = np.empty(a.shape, dtype=np.float32)

    if len(a.shape) == 1 or a.shape[0] == 1 or a.shape[1] == 1:
        shape = (1, max(a.shape))
        g = utils.distributed_vector(shape[1])

        max_threads = settings.block[0] * settings.block[1] * settings.block[2]
        b = (min(shape[1], max_threads), 1, 1)
    else:
        shape = a.shape
        g = utils.distributed_grid(shape)
        b = settings.block

    op = kernels['mat_scale.cu'].get_function('mat_scale')
    op(np.float32(alpha), In(a.astype(np.float32)), Out(out),
       np.int32(shape[0]), np.int32(shape[1]),
       grid=g, block=b)

    return out.reshape(a.shape).astype(a.dtype)


def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    raise NotImplementedError


def add_bias(a, bias, out=None):
    original_type = a.dtype
    a = a.astype(np.float32)

    if not out: out = np.empty(a.shape, dtype=np.float32)

    shape = a.shape
    if len(shape) == 1: shape = (shape[0], 1)

    op = kernels['mat_add_bias.cu'].get_function('mat_add_bias')
    op(In(a), In(bias), Out(out),
       *np.int32(shape),
       np.int32(bias.shape[0]),
       grid=utils.distributed_grid(shape), block=settings.block)

    return out.astype(original_type)


def conv(t, tk, stride=(1, 1), padding=(1, 1), out=None):
    """Compute the convolution between two tensor of rank 3."""
    assert len(t.shape) == len(tk.shape) == 3

    # Output has the entering tensor `t`'s width and height,
    # and channel count equal to the number of kernels in `tk`.
    shape = t.shape[:2] + tk.shape[-1:]

    if not out: out = np.empty(shape).astype(np.float32)
    assert out.shape == shape

    op = kernels['conv.cu'].get_function('conv')
    op(In(t.astype(np.float32)), In(tk.astype(np.float32)), Out(out),

       np.int32(t.shape[0]), np.int32(t.shape[1]), np.int32(t.shape[2]),
       np.int32(tk.shape[0]), np.int32(tk.shape[1]), np.int32(tk.shape[2]),

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
    'transpose': transpose,
}
