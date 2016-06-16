from unittest import TestCase

import numpy as np
from nose_parameterized import parameterized
from numpy.testing import assert_array_almost_equal

import convcuda.operators as op


class CudaTest(TestCase):
    def setUp(self):
        np.random.seed(0)
        op.set_mode('gpu')

    @parameterized.expand([
        ((1, 1),),
        ((32, 32),),
        ((124, 43),),
        ((1066, 1024),),
    ])
    def test_add_operator(self, expected_shape):
        a, b = np.random.rand(*expected_shape), np.random.rand(*expected_shape)

        expected = a + b
        actual = op.add(a, b)

        self.assertEqual(actual.shape, expected_shape)
        assert_array_almost_equal(actual, expected, decimal=6)

    @parameterized.expand([
        ((1, 1), (1, 1),),
        ((24, 24), (24, 24),),
        ((243, 45), (45, 67),),
    ])
    def test_dot_operator(self, expected_s_a, espected_s_b):
        a, b = np.random.rand(*expected_s_a), np.random.rand(*espected_s_b)

        expected = np.dot(a, b)
        actual = op.dot(a, b)

        self.assertEqual(actual.shape, (expected_s_a[0], espected_s_b[1]))
        assert_array_almost_equal(actual, expected, decimal=5)

    @parameterized.expand([
        ((1, 1),),
        ((32, 32),),
        ((125, 35),),
        ((4014, 1025),),
        ((4096, 4096),),
    ])
    def test_hadamard_operator(self, shape):
        a, b = np.random.randn(*shape), np.random.randn(*shape)

        expected = a * b
        actual = op.hadamard(a, b)

        c = expected - actual

        self.assertEqual(actual.shape, shape)
        # Almost equal is required because my video card only accepts float32.
        assert_array_almost_equal(actual, expected, decimal=6)
