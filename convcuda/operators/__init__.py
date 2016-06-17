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


def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    return _run_operator('sum', a,
                         axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def conv(t, tk, stride=(1, 1), padding=(1, 1), out=None):
    return _run_operator('conv', t, tk,
                         stride=stride, padding=padding, out=out)
