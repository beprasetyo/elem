# Design notes — modularization

**Important distinction:** This is a *controlled refactor* of existing working scripts,
not a rewrite of their numerical algorithms into new approximations. Exceptions are
explicitly documented: BBM Pt global P2 recovery; nonnegative Pt evaluation cap;
B2/C2 available strength `(sigma_n_B + Pt)*tan(phi')` as user selected.

**Package-level boundaries**

- `core/shape.py`: shared P2 interpolation polynomial.
- `core/mechanics.py`: effective stress tensor projection and selected equivalent shear strength.
- `stages/a2.py`: validates 15-node connectivity and 12 GP/element, solves L2 field recovery; writes NPZ+report.
- `stages/b1.py`: `BishopField` loads and validates NPZ, samples the curve using FE-edge splitting; saves B1 quadrature.
- `stages/b2.py`: consumes CSV and report, evaluates strength and signed-shear FS with material-specific parameters.
- `stages/c1.py`: allowed material geometries, entry/exit ranges, candidates, domain GeoJSON.
- `stages/c2.py`: circular candidate generation, screening, shortlist FE-edge refinement; writes candidates+best.
- `runner.py`: one central config; determines output paths; sequentially executes selected stages; checks required files; writes manifest.

**Deliberate behavior**

- `strength_formulation` is explicit and recorded for every B2/C2 output.
- Material indices are never merged just because they share text labels.
- A2 does not add a separate saturation correction to PLAXIS Bishop-effective stress.
- Pt is a scalar state field, not a Cartesian stress component.
- Interpolation outside active FE elements is rejected.
- C1 sample counts are independent and validated by C2 against saved output.
- Bad strength parameters/missing Pt are rejected; never silently substituted.
- Signed shear is not converted to absolute shear to force low or positive FS.

**Remaining architectural improvements for later**

- Split heavy C1/C2 visualization code into a standalone `viz/` module; at present plotting routines remain in stage modules to avoid destabilizing regression-tested numerics.
- Implement batch processing of 94 PLAXIS snapshot CSVs with consistent mesh and caching, without overwriting per-day data.
- Move PLAXIS Output A1 extraction into an importer module once its exact source is supplied.
- Store full stage artifact provenance with content hashes and cache/invalidation decisions if the pipeline will be used for repeated scenario runs.

## v2 modules

- `src/kimlee/validation/pt_recovery.py`: pure postprocessing Pt recovery
  diagnostics and fixed-surface FS sensitivity; called by the runner after C2.
  The original recovered field remains unchanged.
- `src/kimlee/core/fast_geometry.py`: optional Numba-compiled C2 numerical
  geometry; `stages/c2.py` uses it only when `use_numba_geometry` is true.
  All shape/discrete-field and physical acceptance checks are unchanged.
- `tests/test_fast_geometry.py` and `tests/test_pt_audit.py`: JIT/NumPy
  agreement and analytical sensitivity fixtures.

Numba compilation belongs inside the optional kernel, not in the public
stage runner. Python orchestration is retained for all file/mesh operations.
