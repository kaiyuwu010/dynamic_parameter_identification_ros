from __future__ import annotations
import numpy as np
from scipy import linalg, signal
"""基础数值计算工具模块"""

# 计算速度和加速度的函数
def differentiate_positions(positions, dt, *, window_length=11, polyorder=3):
    """Estimate velocity and acceleration without wrapping the first sample.
    A Savitzky-Golay differentiator is used when enough samples are available;
    otherwise ``numpy.gradient`` provides a safe short-record fallback.
    """
    q = np.asarray(positions, dtype=float)
    if q.ndim != 2 or q.shape[0] < 2:
        raise ValueError("positions must have shape (samples, joints), samples >= 2")
    if not np.isscalar(dt) or float(dt) <= 0:
        raise ValueError("dt must be a positive scalar")
    n = q.shape[0]
    window = min(int(window_length), n if n % 2 else n - 1)
    if window >= polyorder + 2 and window >= 5:
        qd = signal.savgol_filter(q, window, polyorder, deriv=1, delta=dt, axis=0, mode="interp")
        qdd = signal.savgol_filter(q, window, polyorder, deriv=2, delta=dt, axis=0, mode="interp")
    else:
        edge_order = 2 if n >= 3 else 1
        qd = np.gradient(q, dt, axis=0, edge_order=edge_order)
        qdd = np.gradient(qd, dt, axis=0, edge_order=edge_order)
    return qd, qdd

# 计算最小二乘解的函数，在列缩放后，不使用正规方程
def scaled_least_squares(regressor, target, *, weights=None, rcond=None):
    """Solve least squares after column scaling, without normal equations."""
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
    solution, residuals, rank, singular_values = np.linalg.lstsq(H / scale, y, rcond=rcond)
    solution = solution / scale.reshape((-1,) + (1,) * (solution.ndim - 1))
    return solution, residuals, rank, singular_values

# 提取基本动力学参数，通过QR分解的列主元法。(从完整惯性参数中分离可独立辨识列Pb和相关列Pd)
def base_parameter_transform(observation_matrix, *, rtol=None):
    """Compute a deterministic base-parameter column transform using pivoted QR."""
    Z = np.asarray(observation_matrix, dtype=float)
    if Z.ndim != 2:
        raise ValueError("observation_matrix must be two-dimensional")
    _, R, pivots = linalg.qr(Z, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(R))
    if diagonal.size == 0:
        raise ValueError("observation_matrix must not be empty")
    if rtol is None:
        rtol = max(Z.shape) * np.finfo(float).eps
    rank = int(np.count_nonzero(diagonal > rtol * diagonal[0]))
    if rank == 0:
        raise ValueError("observation_matrix has zero numerical rank")

    independent = np.asarray(pivots[:rank], dtype=int)
    dependent = np.asarray(pivots[rank:], dtype=int)
    Pb = np.eye(Z.shape[1])[:, independent]
    Pd = np.eye(Z.shape[1])[:, dependent]
    if dependent.size:
        Kd = linalg.solve_triangular(R[:rank, :rank], R[:rank, rank:])
    else:
        Kd = np.empty((rank, 0))
    return Pb, Pd, Kd, rank
