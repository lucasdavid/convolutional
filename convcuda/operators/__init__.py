from . import _default, _cuda

# Map of Operators.
_ops = _default.ops


def set_mode(device='default'):
    """Set operation mode by switching the map of operators.

    :param device: str, ('default', 'cpu' or 'gpu'):
        which device should perform the operations. 'default' and 'cpu' will
        execute with numpy.
    """
    global _ops
    assert device in ('default', 'cpu', 'gpu')

    if device in ('default', 'cpu'):
        _ops = _default.ops
    if device == 'gpu':
        _ops = _cuda.ops


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
    if op not in _ops:
        raise ValueError('Cannot find %s operator' % op)

    return _ops[op](*args, **kwargs)


def dot(a, b, out=None):
    return _run_operator('dot', a, b, out=out)


def add(a, b, out=None):
    return _run_operator('add', a, b, out=out)


def hadamard(a, b, out=None):
    return _run_operator('hadamard', a, b, out=out)
