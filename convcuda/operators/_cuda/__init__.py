import os

import numpy as np
import pycuda.autoinit
from pycuda import compiler
from pycuda.driver import In, Out

from . import settings


def _load_kernels():
    """Load All Kernels.

    :return: a list of compiled kernels.
    """
    kernels = {}
    kernels_folder = os.path.dirname(os.path.realpath(__file__))
    kernels_folder = os.path.join(kernels_folder, 'kernels')

    for k_name in os.listdir(kernels_folder):
        k_path = os.path.join(kernels_folder, k_name)

        with open(k_path, 'r') as f:
            kernel = f.read()
            kernels[k_name] = compiler.SourceModule(kernel)

    return kernels


kernels = _load_kernels()


def add(a, b, out=None):
    assert a.shape == b.shape, \
        ('Cannot add matrices. Incompatible shapes: %s and %s.' %
         (a.shape, b.shape))

    original_type = a.dtype
    a, b = a.astype(np.float32), b.astype(np.float32)

    if not out: out = np.zeros(a.shape).astype(np.float32)

    shape = out.shape
    if len(shape) == 1: shape = (shape[0], 1)
    shape = np.array(shape, dtype=np.int32)

    op = kernels['mat_add_k.cu'].get_function('mat_add')
    op(In(a), In(b), Out(out), shape[0], shape[1],
       grid=settings.grid, block=settings.block)

    return out.astype(original_type)


def dot(a, b, out=None):
    """Compute the Dot Product Between Two Matrices."""
    original_type = a.dtype

    # Convert arrays (10,) -> (10, 1).
    a = np.atleast_2d(a.astype(np.float32))
    b = np.atleast_2d(b.astype(np.float32))

    assert a.shape[1] == b.shape[0], \
        ('Cannot apply dot operator on matrices. '
         'Incompatible shapes: %s and %s.' % (a.shape, b.shape))

    if not out: out = np.zeros((a.shape[0], b.shape[1])).astype(np.float32)

    op = kernels['mat_dot_k.cu'].get_function('mat_dot')
    op(In(a), In(b), Out(out),
       np.int32(a.shape[0]), np.int32(a.shape[1]),
       np.int32(b.shape[0]), np.int32(b.shape[1]),
       grid=settings.grid, block=settings.block)

    return out.astype(original_type)


def hadamard(a, b, out=None):
    assert a.shape == b.shape, \
        ('Cannot apply hadamard operator onto matrices. Incompatible shapes: '
         '%s and %s.' % (a.shape, b.shape))

    # Convert to float32 array, but keep the original
    # type so we can return a consistent answer.
    original_type = a.dtype
    a, b = a.astype(np.float32), b.astype(np.float32)

    if out is None: out = np.zeros_like(a)

    # Convert shapes such as (100,) to (100, 1).
    shape = out.shape
    if len(shape) == 1: shape = (shape[0], 1)
    shape = np.array(shape, dtype=np.int32)

    # Retrieve Hadamard kernel and execute it.
    op = kernels['mat_hadamard_k.cu'].get_function('mat_hadamard')
    op(In(a), In(b), Out(out), shape[0], shape[1],
       grid=settings.grid, block=settings.block)

    return out.astype(original_type)


ops = {
    'add': add,
    'dot': dot,
    'hadamard': hadamard
}
