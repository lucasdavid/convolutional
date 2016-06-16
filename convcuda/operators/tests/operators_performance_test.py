import time
from unittest import TestCase

import numpy as np
from nose_parameterized import parameterized
from convcuda import operators as op


class PerformanceTest(TestCase):
    @parameterized.expand([
        ((4, 4), (4, 4), ('dummy', 'gpu'), 1.2),
        ((4, 4), (4, 4), ('cpu', 'gpu'), 4),

        ((12, 24), (24, 32), ('gpu', 'dummy'), 10),
        ((12, 24), (24, 32), ('cpu', 'dummy'), 64),
        ((12, 24), (24, 32), ('gpu', 'cpu'), 10),

        ((243, 45), (45, 67), ('gpu', 'dummy'), 64),
        ((243, 45), (45, 67), ('cpu', 'dummy'), 64),
        ((243, 45), (45, 67), ('gpu', 'cpu'), 10),

        ((2048, 2048), (2048, 2048), ('gpu', 'cpu'), 2),
    ])
    def test_cuda_takes_less_time(self, expected_s_a, espected_s_b, modes,
                                  expected_speed_up):
        a, b = np.random.rand(*expected_s_a), np.random.rand(*espected_s_b)

        times = []

        for mode in modes:
            times.append(time.time())
            op.set_mode(mode)
            c = op.dot(a, b)
            times[-1] = time.time() - times[-1]

        self.assertLess(times[0], times[1])

        speed_up = times[1] / times[0]
        self.assertGreater(speed_up, expected_speed_up)
