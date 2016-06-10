import pycuda.autoinit

import numpy as np
from pycuda import compiler
from pycuda.driver import In, Out

from . import settings

sources = """
__global__ void k_mat_hadamard(float *a, float *b, float *c,
                               int size_x, int size_y) {
    const int i = threadIdx.x,
              j = threadIdx.y;
    const int real_pos = i * size_y + j;

    if (i < size_x && j < size_y)
        c[real_pos] = a[real_pos] * b[real_pos];
}
"""


def hadamard(a, b, out=None):
    assert a.shape == b.shape, \
        ('Cannot apply hadamard operator onto matrices. Incompatible shapes: '
         '%s and %s.' % (a.shape, b.shape))

    # Convert to float32 array.
    original_dtype = a.dtype
    a, b = a.astype(np.float32), b.astype(np.float32)

    if out is None: out = np.zeros_like(a)

    # Convert shapes such as (100,) to (100, 1).
    shape = out.shape
    if len(shape) == 1: shape = (shape[0], 1)
    shape = np.array(shape, dtype=np.int32)

    source = compiler.SourceModule(sources)

    op = source.get_function('k_mat_hadamard')
    op(In(a), In(b), Out(out), shape[0], shape[1],
       grid=settings.grid, block=settings.block)

    return out.astype(original_dtype)


def add(a, b, out=None):
    assert a.shape == b.shape, \
        ('Cannot add matrices. Incompatible shapes: %s and %s.' %
         (a.shape, b.shape))

    if not out: out = np.zeros(a.shape)
    sources.get_function('k_mat_add')(In(a), In(b), Out(out),
                                      In(out.shape[0]), In(out.shape[1]),
                                      block=settings.grid)
    return out


def dot(a, b, out=None):
    """Compute the Dot Product Between Two Matrices."""
    a = np.atleast_2d(a.astype(np.float32))
    b = np.atleast_2d(b.astype(np.float32))

    # Two vectors, should compute the inner product instead.
    if a.shape[0] == b.shape[0] == 1 or a.shape[1] == b.shape[1] == 1: b = b.T

    assert a.shape[1] == b.shape[0], \
        ('Cannot multiply matrices. Incompatible shapes: %s and %s.'
         % (a.shape, b.shape))

    if not out: out = np.zeros(a.shape[0], b.shape[1])
    sources.get_function('k_mat_mul')(In(a), In(b), Out(out),
                                      block=settings.grid)
    return out


ops = {
    'add': add,
    'dot': dot,
    'hadamard': hadamard
}
