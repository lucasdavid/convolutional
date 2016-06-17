import time
from unittest import TestCase

import numpy as np
from nose_parameterized import parameterized
from convcuda import operators as op


class PerformanceTest(TestCase):
    @parameterized.expand([
        ((4, 4), (4, 4), ('sequential', 'gpu'), 1.2),
        ((4, 4), (4, 4), ('vectorized', 'gpu'), 4),

        ((12, 24), (24, 32), ('gpu', 'sequential'), 10),
        ((12, 24), (24, 32), ('vectorized', 'sequential'), 64),
        ((12, 24), (24, 32), ('gpu', 'vectorized'), 10),

        ((243, 45), (45, 67), ('gpu', 'sequential'), 64),
        ((243, 45), (45, 67), ('vectorized', 'sequential'), 64),
        ((243, 45), (45, 67), ('gpu', 'vectorized'), 10),

        ((2048, 2048), (2048, 2048), ('gpu', 'vectorized'), 2),
    ])
    def test_had_speedup(self, shape_a, shape_b, modes, min_speed_up):
        a, b = np.random.rand(*shape_a), np.random.rand(*shape_b)
        times = []
        for mode in modes:
            times.append(time.time())
            op.set_mode(mode)
            c = op.dot(a, b)
            times[-1] = time.time() - times[-1]

        self.assertLess(times[0], times[1])

        speed_up = times[1] / times[0]
        self.assertGreater(speed_up, min_speed_up)
