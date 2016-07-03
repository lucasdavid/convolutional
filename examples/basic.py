from numpy.testing import assert_array_almost_equal
import numpy as np
from convcuda import op, Device


def main():
    X, W = np.random.rand(10, 50), np.random.rand(50, 100)

    with Device('vectorized') as d:
        print('y = X dot W on %s' % d.device_name)
        y_vectorized = op.dot(X, W)

    with Device('gpu') as d:
        print('y = X dot W on %s' % d.device_name)
        y_gpu = op.dot(X, W)

        print(y_gpu)

    assert_array_almost_equal(y_vectorized, y_gpu, decimal=5)


if __name__ == '__main__':
    main()
