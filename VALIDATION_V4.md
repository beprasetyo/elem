# Numerical validation — Stage C3 v4

Snapshot: `Phase_12_Step_2287.csv`, PLAXIS time 100.410761 days. C1: entry 23–46 m (40 points), exit 60–97 m (95 points); allowed materials index 0 and 5. `bishop_pt` strength: material 0 phi=32°, c=0; material 5 phi=36°, c=0. The snapshot is after the previously identified macroscopic failure time.

## Regression baseline (v3 C2)
- 57,000 circular grid candidates, 28,365 screened feasible, 100 FE-edge refinements.
- Minimum FE-verified C2 circular candidate: **0.9482561679303649**, derived from an unconverged C2 BFGS run; treat as reference only.
- Formal C2 BFGS success: 0 of 3 runs.

## Stage C3 v4 test with objective screening step 0.8 m

| Number of vertices | Shape variables | SLSQP success | Screened FS | Independently FE-edge verified FS |
|---:|---:|---|---:|---:|
| 4 | 6 | Yes | 0.971102 | 0.9710489515 |
| 7 | 12 | Yes | 0.946598 | 0.9469190543 |
| 13 | 24 | Yes | 0.944144 | **0.9448214504** |

The 4-vertex noncircular polyline is not the same geometry as the original C2 circular arc: a higher FS at this resolution is not contradictory.

## Independent repeat using 0.4 m screening step

| Vertices | FE-edge FS after re-optimization |
|---:|---:|
| 4 | 0.9721410419 |
| 7 | 0.9471639881 |
| 13 | **0.9447260195** |

The 13-vertex results differ by about 0.0000954 (0.01% relative); this compares **separately optimized surfaces**, not a fixed-geometry quadrature convergence study. A formal global optimization or mesh convergence claim would be premature.

## BFGS gradient audit

For the best v3 C2 BFGS candidate, central finite differences for `(xc,yc,x1)` yielded:

| FD step (m) | dFS/dxc | dFS/dyc | dFS/dx1 |
|---:|---:|---:|---:|
| 0.050 | -0.000160 | +0.000667 | invalid |
| 0.025 | +0.000966 | -0.001238 | +0.005506 |
| 0.010 | +0.006464 | -0.006634 | -0.001074 |
| 0.005 | -0.001181 | +0.001164 | -0.009982 |
| 0.001 | -0.001197 | +0.001156 | +0.005162 |

The sign and size vary strongly, and some stencil evaluations are outside the feasible region. Such finite-difference sensitivity is one plausible mechanism behind BFGS precision-loss warnings; this is an audit, not proof of a unique cause.

## Tests and caveats

- Unit tests: **20 passed** (`pytest -q`).
- C3 each reported final candidate was verified using existing Stage B1 FE-edge integration and Stage B2 FS.
- SLSQP reports success at 4, 7, and 13 vertices *in this specific run*, but subsequent feasible coordinate polishing may improve the selected solution. This does **not** prove a local or global mathematical optimum.
- The C3 nonlinear optimizer is NOT the specific 1997 feasible-direction solver. See `METHODOLOGY_AUDIT_C3.md`.
