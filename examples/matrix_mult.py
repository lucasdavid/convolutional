import numpy as np
from convcuda import operators as op
from numpy.testing import assert_array_equal

a = np.array([[1, 2], [3, 4]])
b = np.array([[3, 2], [3, 4]])
c = op.dot(a, b)
assert_array_equal(c, [[9, 10], [21, 22]])
op.set_mode('gpu')
c = op.dot(a, b)
assert_array_equal(c, [[9, 10], [21, 22]])

print('ok')
