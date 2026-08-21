from __future__ import annotations
from pathlib import Path
import numpy as np

class PinocchioRegressor:
    """URDF-backed torque regressor with no ROS dependency."""

    def __init__(self, urdf_path, *, gravity=(0.0, 0.0, -9.81)):
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise RuntimeError("install the 'pin' package to use Pinocchio") from exc
        self.pin = pin
        self.urdf_path = Path(urdf_path)
        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.model.gravity.linear = np.asarray(gravity, dtype=float)
        self.data = self.model.createData()
        self.parameter_count = 10 * (self.model.njoints - 1)

    @property
    def dof(self):
        return self.model.nv

    @property
    def joint_names(self):
        return tuple(self.model.names[1:])

    def sample_regressor(self, q, qd, qdd):
        return np.asarray(self.pin.computeJointTorqueRegressor(
            self.model, self.data, np.asarray(q), np.asarray(qd), np.asarray(qdd)
        ))

    def stacked_regressor(self, identification_data, *, friction=True):
        if identification_data.position.shape[1] != self.dof:
            raise ValueError("measurement DOF does not match the URDF")
        blocks = []
        for q, qd, qdd in zip(identification_data.position,
                              identification_data.velocity,
                              identification_data.acceleration):
            Y = self.sample_regressor(q, qd, qdd)
            if friction:
                Yf = np.hstack((np.diag(np.sign(qd)), np.diag(qd)))
                Y = np.hstack((Y, Yf))
            blocks.append(Y)
        return np.vstack(blocks), identification_data.torque.reshape(-1)

    def nominal_parameters(self, *, friction=True):
        values = np.concatenate([
            np.asarray(inertia.toDynamicParameters()).reshape(-1)
            for inertia in self.model.inertias[1:]
        ])
        if friction:
            values = np.concatenate((values, np.zeros(2 * self.dof)))
        return values
