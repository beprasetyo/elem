# Changelog v4 — C3 noncircular search and circular-gradient audit

## Implemented

- Added `src/kimlee/stages/c3.py` and integrated stage `"c3"` into the central runner.
- Noncircular surfaces consist of a polyline with N vertices. Its shape vector is N x-coordinates and N-2 interior y-coordinates, total `2N-2`, with entry and exit elevations obtained from the C1 ground profile.
- Progressive vertex growth `4 → 7 → 13`, inserting new vertices at the midpoints of existing segments. Other configurations must follow `N_next = 2*N - 1`.
- Enforced explicit geometry restrictions (entry/exit x zones, strictly increasing x, minimum segment horizontal span, maximum turning angle, admissible C1 area, maximum depth, minimum sliding body area).
- Objective retains the original `bishop_pt` or `bishop_only` formulation from B2, evaluates signed shear stresses and separates available shear resistance.
- SLSQP constrained optimization with finite-difference sensitivities is followed by deterministic feasible-coordinate polishing. The *original* feasible-direction method from Kim & Lee (1997) has **not** been implemented; solver success does not mean faithful paper reproduction.
- Each final N-vertex candidate is independently checked via existing Stage B1 FE-edge segmentation and Stage B2 stress integration. Reports store both screened and FE-edge FS and actual SLSQP termination status.
- Added `run_c2_gradient_audit.py`: finite-difference gradient sensitivity of saved C2 circular BFGS candidates, including invalid geometry perturbations. No new command line options are needed.
- Added six C3 tests; 20 total test cases pass.

## Known limitations

- C2 BFGS precision-loss warnings remain unresolved. The gradient audit shows sensitivity to finite-difference steps and hard C1 feasibility boundaries. C3 is an **additional conditional search**, not a solution proving BFGS convergence.
- C3 uses SLSQP plus a simple feasible-coordinate search, not Kim & Lee's particular feasible-direction algorithm.
- Screened objective integrates a piecewise-linear path with Gauss–Legendre points and material-separated P2 stress recovery. It may be nonsmooth at FEM/material boundaries.
- C1 entry/exit intervals and depth limits are provisional; the result is neither a global minimum nor the actual failure surface.
- Bishop + Pt is an adopted BBM-informed equivalent-Mohr–Coulomb postprocessing strength, not the full BBM yield surface.
- The initial `N=4` piecewise-linear interpolation is not identical to the C2 circular arc. It can have FS higher than the circular reference before further vertices are introduced.

## Upgrade and usage

1. Back up `config/` and `results/` before replacing project source code.
2. Place source and config v4 in the project root; retain original PLAXIS data files in the parent directory.
3. To run all stages, use `python run.py` (Stage C3 follows C2).
4. To run **only C3** using existing A2/C1/C2 outputs, set `analysis.stages` to `["c3"]` in `config/project.json`, then run `python run.py`. Saved `results/<snapshot>/C2/C2_report.json` and `C2_best_circular_surface.csv` must already exist and use the same A2 day/formulation.
5. To diagnose remaining BFGS gradient instability from saved C2 output, run `python run_c2_gradient_audit.py`.
6. `python -m pytest -q` runs 20 regression/unit tests.

Read `METHODOLOGY_AUDIT_C3.md` and `VALIDATION_V4.md` for detailed evidence and scientific limitations.
