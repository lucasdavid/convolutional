from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal
import convcuda.operators as op


class CudaTest(TestCase):
    def setUp(self):
        np.random.seed(0)
        op.set_mode('gpu')

    def test_add_operator(self):
        a, b = np.random.rand(4, 4), np.random.rand(4, 4)

        expected = a + b
        actual = op.add(a, b)

        assert_array_almost_equal(expected, actual, decimal=6)

    def test_dot_operator(self):
        a, b = np.random.rand(4, 4), np.random.rand(4, 4)

        expected = np.dot(a, b)
        actual = op.dot(a, b)

        assert_array_almost_equal(expected, actual, decimal=6)

    def test_hadamard_operator(self):
        a, b = np.random.randn(32, 32), np.random.randn(32, 32)

        expected = a * b
        actual = op.hadamard(a, b)

        # Almost equal is required because my video card only accepts float32.
        assert_array_almost_equal(actual, expected, decimal=6)
