# v2 — Pt recovery audit and optional Numba acceleration (2026-11-22)

## Added

- `src/kimlee/validation/pt_recovery.py` and the optional `pt_audit` stage.
  Audits original vs unconstrained P2 recovered BBM Pt at all valid Gaussian
  points per material and compares fixed-geometry FS sensitivity to Pt
  treatments: raw P2, GP-clipped P2 (current), nodal-clipped P2, and a
  positivity-preserving P1 interpolation using P2-recovered corner values.
- Writes `Pt_audit_report.json`, `Pt_audit_GP_material_statistics.csv`,
  `Pt_audit_fixed_surface_sensitivity.csv`, and `Pt_audit_diagnostics.png`.
  Uses the existing B1 diagnostic and the saved C2 best-refined pointwise
  quadrature, when present. Does NOT reoptimize the slip surface.
- Optional Numba backend for C2's **numerical arc sampling, area, maximum
  depth, and x-monotonicity checks**. Set
  `circular_search.use_numba_geometry=true` in `config/project.json`.
  Install with `pip install -r requirements-numba.txt`.
- Four additional unit tests (11 total). NumPy is still the default backend
  for backward compatibility.

## Unchanged

- Original A2 global P2 L2 stress and Pt recovery; no material parameter changes.
- Bishop + Pt Mohr–Coulomb equivalent `R` formula.
- B1 FE-edge integration, B2, C1, C2 search grid, Shapely checks, TriFinder
  interpolation, shortlisting and FE-edge refinement.
- C2 default refine-top 18. Increasing it is a separate study and may be costly.

## Limitations

- Numba **does not compile** Shapely, Matplotlib, TriFinder, Pandas or SciPy
  sparse solver. Do not extrapolate numeric-only microbenchmarks to total C2.
- Pt clipping is a numerical safeguard, NOT a constitutive model derivation.
- No Kim & Lee (1997) BFGS circular/noncircular optimization yet.
- No validated historical FS(t) yet; snapshot is day 100.41 (post-failure).
