import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from identification_numerics import (  # noqa: E402
    base_parameter_transform,
    differentiate_positions,
    scaled_least_squares,
)
from utility_math import quaternion_normalize  # noqa: E402


class IdentificationNumericsTest(unittest.TestCase):
    def test_polynomial_derivatives_have_no_wraparound_outlier(self):
        dt = 0.01
        t = np.arange(101) * dt
        q = np.column_stack((t ** 2, -0.5 * t ** 2))
        qd, qdd = differentiate_positions(q, dt)
        np.testing.assert_allclose(qd[:, 0], 2 * t, atol=1e-10)
        np.testing.assert_allclose(qdd[:, 0], 2.0, atol=1e-9)
        self.assertLess(abs(qdd[0, 0] - 2.0), 1e-9)

    def test_scaled_least_squares_handles_ill_scaled_columns(self):
        rng = np.random.default_rng(4)
        H = np.column_stack((rng.normal(size=200),
                             1e-7 * rng.normal(size=200)))
        expected = np.array([2.0, -3e6])
        y = H @ expected
        actual, _, rank, _ = scaled_least_squares(H, y)
        self.assertEqual(rank, 2)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-8)

    def test_base_parameter_transform_reconstructs_dependent_columns(self):
        rng = np.random.default_rng(7)
        independent = rng.normal(size=(80, 3))
        dependent = independent @ np.array([[2.0], [-1.0], [0.5]])
        Z = np.hstack((independent, dependent))
        Pb, Pd, Kd, rank = base_parameter_transform(Z)
        self.assertEqual(rank, 3)
        np.testing.assert_allclose(Z @ Pd, Z @ Pb @ Kd, atol=1e-11)

    def test_quaternion_normalization_uses_euclidean_norm(self):
        q = quaternion_normalize([2.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            quaternion_normalize([0.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
