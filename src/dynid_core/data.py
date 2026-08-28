from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal


def differentiate_positions(positions, dt, *, window_length=11, polyorder=3):
    q = np.asarray(positions, dtype=float)
    if q.ndim != 2 or q.shape[0] < 2:
        raise ValueError("positions must have shape (samples, joints), samples >= 2")
    if not np.isscalar(dt) or float(dt) <= 0:
        raise ValueError("dt must be a positive scalar")
    n = q.shape[0]
    window = min(int(window_length), n if n % 2 else n - 1)
    if window >= polyorder + 2 and window >= 5:
        qd = signal.savgol_filter(q, window, polyorder, deriv=1, delta=dt,
                                  axis=0, mode="interp")
        qdd = signal.savgol_filter(q, window, polyorder, deriv=2, delta=dt,
                                   axis=0, mode="interp")
    else:
        edge_order = 2 if n >= 3 else 1
        qd = np.gradient(q, dt, axis=0, edge_order=edge_order)
        qdd = np.gradient(qd, dt, axis=0, edge_order=edge_order)
    return qd, qdd


@dataclass(frozen=True)
class IdentificationData:
    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    torque: np.ndarray
    joint_names: tuple[str, ...]

    def __post_init__(self):
        arrays = [self.time, self.position, self.velocity,
                  self.acceleration, self.torque]
        if any(not np.all(np.isfinite(a)) for a in arrays):
            raise ValueError("identification data contains NaN or infinity")
        n, joints = self.position.shape
        if self.time.shape != (n,):
            raise ValueError("time must contain one value per sample")
        if any(a.shape != (n, joints) for a in arrays[2:]):
            raise ValueError("position, velocity, acceleration and torque shapes differ")
        if len(self.joint_names) != joints:
            raise ValueError("joint_names count does not match data")
        if np.any(np.diff(self.time) <= 0):
            raise ValueError("timestamps must be strictly increasing")


def _indexed_columns(fieldnames, prefix):
    columns = [name for name in fieldnames
               if name.startswith(prefix) and name[len(prefix):].isdigit()]
    return sorted(columns, key=lambda name: int(name[len(prefix):]))


def load_measurement_csv(path, *, dt=None, torque_prefix="tau_",
                         time_column="timestamp", trim=5):
    """Load q/tau CSV data and consistently estimate qd/qdd.

    The legacy files have no timestamp, so callers must explicitly provide
    ``dt``.  ``torque_prefix`` prevents accidental use of ``tau_ext_*``.
    """
    path = Path(path)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        q_columns = _indexed_columns(fields, "q_")
        tau_columns = _indexed_columns(fields, torque_prefix)
        if not q_columns or len(q_columns) != len(tau_columns):
            raise ValueError("CSV must contain matching q_i and torque columns")
        rows = list(reader)

    q = np.asarray([[float(row[c]) for c in q_columns] for row in rows])
    tau = np.asarray([[float(row[c]) for c in tau_columns] for row in rows])
    if time_column in fields:
        time = np.asarray([float(row[time_column]) for row in rows])
        sample_dt = float(np.median(np.diff(time)))
    else:
        if dt is None or dt <= 0:
            raise ValueError("CSV has no timestamp; provide a positive dt")
        sample_dt = float(dt)
        time = np.arange(len(rows), dtype=float) * sample_dt

    qd, qdd = differentiate_positions(q, sample_dt)
    if trim:
        if 2 * trim >= len(time):
            raise ValueError("trim removes all samples")
        selection = slice(trim, -trim)
        time, q, qd, qdd, tau = (a[selection] for a in
                                  (time, q, qd, qdd, tau))
    joint_names = tuple(f"joint_{i}" for i in range(q.shape[1]))
    return IdentificationData(time, q, qd, qdd, tau, joint_names)
