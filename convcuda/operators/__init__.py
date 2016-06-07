from . import _default, _cuda

# Map of operators.
_ops = _default.ops


def _run_operator(op, *args, **kwargs):
    if op not in _ops:
        raise ValueError('Cannot find %s operator' % op)

    return _ops[op](*args, **kwargs)


def dot(a, b, out=None):
    return _run_operator('dot', a, b, out=None)


def set_mode(device='default'):
    global _ops
    assert device in ('default', 'cpu', 'gpu')

    if device in ('default', 'cpu'):
        _ops = _default.ops
    if device == 'gpu':
        _ops = _cuda.ops
