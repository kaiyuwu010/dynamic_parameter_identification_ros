from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def scaled_least_squares(regressor, target, *, weights=None, rcond=None):
    H = np.asarray(regressor, dtype=float)
    y = np.asarray(target, dtype=float)
    if H.ndim != 2 or y.shape[0] != H.shape[0]:
        raise ValueError("regressor and target sample counts must match")
    if weights is not None:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.size != H.shape[0] or np.any(w <= 0):
            raise ValueError("weights must be positive and match the row count")
        root_w = np.sqrt(w)
        H = H * root_w[:, None]
        y = y * root_w.reshape((-1,) + (1,) * (y.ndim - 1))
    scale = np.linalg.norm(H, axis=0)
    scale[scale == 0] = 1.0
    solution, residuals, rank, singular_values = np.linalg.lstsq(
        H / scale, y, rcond=rcond
    )
    solution = solution / scale.reshape((-1,) + (1,) * (solution.ndim - 1))
    return solution, residuals, rank, singular_values


@dataclass(frozen=True)
class IdentificationResult:
    parameters: np.ndarray
    predicted_torque: np.ndarray
    residual: np.ndarray
    rmse: float
    rank: int
    singular_values: np.ndarray
    status: str


def estimate_unconstrained(regressor, torque):
    H = np.asarray(regressor, dtype=float)
    y = np.asarray(torque, dtype=float).reshape(-1)
    parameters, _, rank, singular_values = scaled_least_squares(H, y)
    prediction = H @ parameters
    residual = y - prediction
    return IdentificationResult(parameters, prediction, residual,
                                float(np.sqrt(np.mean(residual ** 2))), int(rank),
                                singular_values, "least_squares")


def _pseudo_inertia_constraints(cp, theta, link_count, minimum_mass):
    constraints = []
    for link in range(link_count):
        p = theta[10 * link:10 * (link + 1)]
        mass, h = p[0], p[1:4]
        # Pinocchio order: m, hx, hy, hz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz.
        inertia = cp.bmat([
            [p[4], p[5], p[7]],
            [p[5], p[6], p[8]],
            [p[7], p[8], p[9]],
        ])
        sigma = 0.5 * cp.trace(inertia) * np.eye(3) - inertia
        pseudo = cp.bmat([[sigma, cp.reshape(h, (3, 1), order="F")],
                          [cp.reshape(h, (1, 3), order="F"),
                           cp.reshape(mass, (1, 1), order="F")]])
        constraints.extend([mass >= minimum_mass,
                            pseudo >> minimum_mass * 1e-9 * np.eye(4)])
    return constraints


def estimate_physically_consistent(regressor, torque, *, link_count,
                                   nominal=None, regularization=1e-8,
                                   minimum_mass=1e-6, solver=None):
    """Convex full-parameter fit using pseudo-inertia LMIs."""
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise RuntimeError("install cvxpy for physical consistency constraints") from exc

    H = np.asarray(regressor, dtype=float)
    y = np.asarray(torque, dtype=float).reshape(-1)
    theta = cp.Variable(H.shape[1])
    constraints = _pseudo_inertia_constraints(cp, theta, link_count,
                                              minimum_mass)
    # Remaining columns are Coulomb and viscous friction coefficients.
    if H.shape[1] > 10 * link_count:
        constraints.append(theta[10 * link_count:] >= 0)
    objective = cp.sum_squares(H @ theta - y)
    if nominal is not None and regularization > 0:
        scale = np.maximum(np.abs(np.asarray(nominal)), 1e-3)
        objective += regularization * cp.sum_squares(
            cp.multiply(1.0 / scale, theta - nominal)
        )
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=solver)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"physical identification failed: {problem.status}")
    parameters = np.asarray(theta.value).reshape(-1)
    prediction = H @ parameters
    residual = y - prediction
    singular_values = np.linalg.svd(H, compute_uv=False)
    rank = int(np.linalg.matrix_rank(H))
    return IdentificationResult(parameters, prediction, residual,
                                float(np.sqrt(np.mean(residual ** 2))), rank,
                                singular_values, problem.status)
