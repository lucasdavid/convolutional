from . import _default, _cuda, _dummy

# Map of Operators.
_operations = _default.operations


def set_mode(device='default'):
    """Set operation mode by switching the map of operators.

    :param device: str, ('default', 'cpu' or 'gpu'):
        which device should perform the operations. 'default' and 'cpu' will
        execute with numpy.
    """
    global _operations
    assert device in ('default', 'dummy', 'cpu', 'gpu')

    if device in ('default', 'cpu'):
        _operations = _default.operations
    if device == 'dummy':
        _operations = _dummy.operations
    if device == 'gpu':
        _operations = _cuda.operations


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
        raise ValueError('Cannot find %s operator' % op)

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
