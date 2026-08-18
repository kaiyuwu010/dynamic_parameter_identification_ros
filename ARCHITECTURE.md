# Dynamics identification architecture

The repository is split into four layers:

1. `dynid_core`: ROS-independent identification using Pinocchio, NumPy/SciPy
   and optional CVXPY physical-consistency constraints.
2. `TrajGeneration.py`: optional CasADi/Ipopt excitation-trajectory design.
3. ROS 2 acquisition scripts: record timestamp, joint position and measured
   actuator torque only.
4. `trajsimulation.py`: optional PyBullet simulation and collision validation.

The core can be run without sourcing ROS:

```bash
PYTHONPATH=src python3 -m dynid_core.cli \
  urdf/med/med7dock.urdf src/test/measurements.csv --dt 0.01
```

Add `--physical` to solve the pseudo-inertia semidefinite constraints with
CVXPY. The legacy CSV files do not contain timestamps, so `--dt` is mandatory.
The default torque input is `tau_0 ... tau_n`; `tau_ext_*` is never selected
implicitly.

Pinocchio 4 currently requires NumPy 2. Keep the core in a dedicated virtual
environment if the ROS distribution or plotting stack requires NumPy 1.x.
