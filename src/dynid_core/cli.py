from __future__ import annotations

import argparse
import json

from .data import load_measurement_csv
from .estimation import estimate_physically_consistent, estimate_unconstrained
from .pinocchio_backend import PinocchioRegressor


def main():
    parser = argparse.ArgumentParser(description="ROS-independent dynamics identification")
    parser.add_argument("urdf")
    parser.add_argument("measurements")
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--torque-prefix", default="tau_")
    parser.add_argument("--physical", action="store_true")
    args = parser.parse_args()

    measured = load_measurement_csv(args.measurements, dt=args.dt,
                                    torque_prefix=args.torque_prefix)
    backend = PinocchioRegressor(args.urdf)
    H, torque = backend.stacked_regressor(measured)
    if args.physical:
        result = estimate_physically_consistent(
            H, torque, link_count=backend.model.njoints - 1,
            nominal=backend.nominal_parameters()
        )
    else:
        result = estimate_unconstrained(H, torque)
    print(json.dumps({"samples": len(measured.time), "dof": backend.dof,
                      "parameter_count": len(result.parameters),
                      "rank": result.rank, "rmse": result.rmse,
                      "status": result.status}, indent=2))


if __name__ == "__main__":
    main()
