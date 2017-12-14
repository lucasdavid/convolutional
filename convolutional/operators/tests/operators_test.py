import abc
from unittest import TestCase

import numpy as np
from nose_parameterized import parameterized
from numpy.testing import assert_array_almost_equal

import convolutional.operators as op


class _BaseTest(TestCase, metaclass=abc.ABCMeta):
    def setUp(self):
        np.random.seed(0)

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
        ((1, 1),),
        ((32, 32),),
        ((124, 43),),
        ((1066, 1024),),
    ])
    def test_sub_operator(self, expected_shape):
        a, b = np.random.rand(*expected_shape), np.random.rand(*expected_shape)

        expected = a - b
        actual = op.sub(a, b)

        self.assertEqual(actual.shape, expected_shape)
        assert_array_almost_equal(actual, expected, decimal=6)

    @parameterized.expand([
        ((1, 1), (1, 1),),
        ((2, 4), (4, 2),),
        ((12, 24), (24, 32),),
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
        ((50, 15),),
        ((3, 24),),
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

    @parameterized.expand([
        ((1, 1),),
        ((32, 1),),
        ((1, 32),),
        ((32, 32),),
        ((125, 35),),
        ((4014, 1025),),
        ((4096, 4096),),
    ])
    def test_scale_operator(self, shape):
        alpha, a = np.random.rand(), np.random.randn(*shape)

        expected = alpha * a
        actual = op.scale(alpha, a)

        c = expected - actual

        self.assertEqual(actual.shape, shape)
        assert_array_almost_equal(actual, expected, decimal=6)

    @parameterized.expand([
        ((3, 3, 1), (3, 3, 1), np.array([[[77], [136], [89]],
                                         [[179], [227], [137]],
                                         [[91], [165], [175]]])),
        ((2, 2, 1), (3, 3, 2), np.array([[[112, 128], [105, 176]],
                                         [[113, 115], [115, 161]]])),
    ])
    def test_conv_operator(self, a_shape, b_shape, expected):
        a, k = 10 * np.random.rand(*a_shape), 10 * np.random.rand(*b_shape)
        a, k = a.astype(int), k.astype(int)

        actual = op.conv(a, k)
        assert_array_almost_equal(actual, expected)

    @parameterized.expand([
        ((1, 1),),
        ((10, 10),),
        ((32, 17),),
        ((40,),),
        ((2, 12, 4, 3,),),
    ])
    def test_sum_operator(self, a_shape):
        a = np.random.rand(*a_shape)
        expected = np.sum(a)
        actual = op.sum(a)

        self.assertAlmostEqual(expected, actual, delta=.00001)

    @parameterized.expand([
        ((1, 1, 1), (1,)),
        ((10, 10, 10), (10,)),
        ((10, 10, 1), (1,)),
        ((32, 17, 3), (3,)),
        ((40, 20, 100), (100,)),
        ((30, 1230, 3), (3,)),
    ])
    def test_add_bias_operator(self, a_shape, bias_shape):
        a = np.random.rand(*a_shape)
        bias = np.random.rand(*bias_shape)

        expected = np.array(a, copy=True)
        expected[:, :, range(bias.shape[0])] += bias
        actual = op.add_bias(a, bias)

        assert_array_almost_equal(expected, actual, decimal=6)


class SequentialTest(_BaseTest):
    def setUp(self):
        super().setUp()
        op.set_mode('sequential')


class VectorizedTest(_BaseTest):
    def setUp(self):
        super().setUp()
        op.set_mode('vectorized')


class GpuTest(_BaseTest):
    def setUp(self):
        super().setUp()
        op.set_mode('gpu')
