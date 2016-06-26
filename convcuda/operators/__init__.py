import numpy as np

from . import _sequential, _vectorized, _cuda
from .. import settings

# Operators map.
_operations = None


def set_mode(device):
    """Set The Operation Mode by Switching The Map of Operators.

    :param device: str, ('default', 'sequential', 'vectorized' or 'gpu'):
        which device should perform the operations.

        Options are:
        * 'default' and 'vectorized' will execute with numpy.
        * 'sequential' implements the operations as they would usually be
          on the programming language C.
        * 'gpu' uses the CUDA interface to make the
          computations in the GPU device.
    """
    global _operations
    assert device in ('default', 'sequential', 'vectorized', 'gpu')

    if device in ('default', 'vectorized'):
        _operations = _vectorized.operations
    elif device == 'sequential':
        _operations = _sequential.operations
    elif device == 'gpu':
        _operations = _cuda.operations


def get_mode():
    """Get the mode in which the operations are currently set.

    :return str: 'vectorized', 'sequential' or 'gpu'.
    """
    global _operations
    if _operations == _vectorized.operations:
        return 'vectorized'
    if _operations == _sequential.operations:
        return 'sequential'
    if _operations == _cuda.operations:
        return 'gpu'


# Set operation mode to the default.
set_mode(settings.DEFAULT_OPERATION_MODE)


# Operator Wrappers.
def _run_operator(op, *args, **kwargs):
    """Run An Operator.

    This is a private method and should not be invoked directly.
    Instead, use one of the wrappers bellow.

    :param op: str, name of the operator in `_ops` map.
    :param args: positional arguments for the operation.
    :param kwargs: key arguments for the operation.
    :return: the operation result.
    """
    if op not in _operations:
        raise ValueError('%s operator is not defined' % op)

    return _operations[op](*args, **kwargs)


def dot(a, b, out=None):
    return _run_operator('dot', a, b, out=out)


def add(a, b, out=None):
    return _run_operator('add', a, b, out=out)


def sub(a, b, out=None):
    return _run_operator('sub', a, b, out=out)


def scale(alpha, a, out=None):
    return _run_operator('scale', alpha, a, out=out)


def hadamard(a, b, out=None):
    return _run_operator('hadamard', a, b, out=out)


def conv(t, tk, stride=(1, 1), padding=(1, 1), out=None):
    return _run_operator('conv', t, tk,
                         stride=stride, padding=padding, out=out)


def add_bias(a, bias, out=None):
    return _run_operator('add_bias', a, bias, out=out)


def transpose(a, axes=None):
    return _run_operator('transpose', a, axes=axes)


def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    return np.sum(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def argmax(a, axis=None, out=None):
    return np.argmax(a, axis=axis, out=out)


class Device(object):
    """Helper class for clean scope setting.

    Example:
        >>> # Default device is 'vectorized'
        >>> with Device('gpu') as s:
        >>>    ... # Do some work with the GPU.
        >>> # Vectorized device is in use once again.
    """

    def __init__(self, device_name):
        self.device_name = device_name
        self.previous_device = None

    def __enter__(self):
        self.previous_device = get_mode()
        set_mode(self.device_name)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_mode(self.previous_device)
