import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynid_core.data import load_measurement_csv  # noqa: E402
from dynid_core.estimation import (  # noqa: E402
    estimate_physically_consistent,
    estimate_unconstrained,
)
from dynid_core.pinocchio_backend import PinocchioRegressor  # noqa: E402


class CoreDataTest(unittest.TestCase):
    def test_loader_selects_measured_torque_not_external_torque(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "measurements.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["q_0", "tau_0", "tau_ext_0"])
                for i in range(20):
                    writer.writerow([0.01 * i ** 2, 10 + i, 1000 + i])
            data = load_measurement_csv(path, dt=0.01, trim=5)
            np.testing.assert_allclose(data.torque[:, 0], np.arange(15, 25))

    def test_loader_requires_dt_when_timestamp_is_absent(self):
        with self.assertRaises(ValueError):
            load_measurement_csv(ROOT / "src/test/measurements.csv")


class PinocchioBackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = PinocchioRegressor(ROOT / "urdf/med/med7dock.urdf")

    def test_regressor_times_urdf_parameters_matches_rnea(self):
        rng = np.random.default_rng(12)
        q = rng.uniform(-0.5, 0.5, self.backend.dof)
        qd = rng.uniform(-0.3, 0.3, self.backend.dof)
        qdd = rng.uniform(-0.8, 0.8, self.backend.dof)
        Y = self.backend.sample_regressor(q, qd, qdd)
        parameters = self.backend.nominal_parameters(friction=False)
        predicted = Y @ parameters
        expected = self.backend.pin.rnea(self.backend.model,
                                         self.backend.model.createData(),
                                         q, qd, qdd)
        np.testing.assert_allclose(predicted, expected, atol=1e-10)

    def test_unconstrained_recovers_synthetic_torque(self):
        rng = np.random.default_rng(8)
        blocks = []
        for _ in range(100):
            blocks.append(self.backend.sample_regressor(
                rng.uniform(-1, 1, 7), rng.uniform(-1, 1, 7),
                rng.uniform(-2, 2, 7)))
        H = np.vstack(blocks)
        expected = self.backend.nominal_parameters(friction=False)
        result = estimate_unconstrained(H, H @ expected)
        np.testing.assert_allclose(result.predicted_torque, H @ expected,
                                   atol=1e-9)
        self.assertLess(result.rmse, 1e-10)


class PhysicalEstimatorTest(unittest.TestCase):
    def test_positive_point_mass_solution(self):
        # One link with a valid inertia and two friction parameters.
        nominal = np.array([2.0, 0.2, 0.0, 0.0,
                            0.04, 0.0, 0.06, 0.0, 0.0, 0.06,
                            0.1, 0.2])
        rng = np.random.default_rng(3)
        H = rng.normal(size=(100, len(nominal)))
        result = estimate_physically_consistent(
            H, H @ nominal, link_count=1, nominal=nominal,
            regularization=1e-10, solver="CLARABEL"
        )
        self.assertLess(result.rmse, 1e-5)
        self.assertGreater(result.parameters[0], 0)
        self.assertTrue(np.all(result.parameters[10:] >= -1e-8))


if __name__ == "__main__":
    unittest.main()
