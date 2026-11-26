# Validation V3 — full real-PLAXIS snapshot, Stage C2

Input: `Phase_12_Step_2287.csv` (day **100.410760986931**), A2 Bishop effective stress + separate Pt, C1 entry `[23,46]`, exit `[60,97]`, material indices `[0,5]`, 40 entry samples, 95 exit samples; `bishop_pt` formulation.

## Full grid, FE-edge shortlist, and BFGS

| Metric | Full test result |
| --- | ---: |
| Trial circular surfaces | 57,000 |
| Valid after screening | 28,365 |
| Explicit FE-edge refinements | 100 / 100 successful |
| FE-verified multistart BFGS candidates | 3 / 3 |
| **BFGS SciPy solver convergence** | **0 / 3 (precision-loss warnings)** |
| Best refined **grid** FS | 0.9492013101742783 |
| Best refined **BFGS feasible candidate** FS | **0.9482561679303649** |
| Earlier v2 baseline FS | 0.949201310 (same grid, 18 refinements) |

The best feasible BFGS candidate has entry `(45.327,69.026)` m and exit `(90.900,35.066)` m (rounded), sagitta `5.404` m, radius `77.419` m, and strength/driving integrated forces `R=1651.5583` and `T=1741.6795` kN/m. These are conditional FE stress-integral quantities, **not a global safety factor certified by a converged optimizer**. Geometric entry/exit changed substantially compared with the old x≈80.862 m grid winner; this deserves independent physical interpretation.

### BFGS record details

- Seed ID 55324: screened FS 0.948297, independently FE-refined FS 0.948256, solver iterations 2; solver message `Desired error not necessarily achieved due to precision loss.`
- Seed ID 54979: screened FS 0.948878, FE-refined FS 0.949290; solver message precision loss.
- Seed ID 55054: screened FS 0.949495, FE-refined FS 0.949638; solver message precision loss.

`refinement_status=OK` means the resultant shape and FS passed B1/B2 verification, **not** that BFGS satisfied convergence criteria. The non-smooth FE/material boundary crossings and penalty handling likely contribute to the difficulties, but that requires further diagnosis.

### FE-edge quadrature / circular polygonization convergence

| Maximum chord length (m) | FS |
| ---: | ---: |
| 0.64 | 0.948276 |
| 0.32 | 0.948256 |
| 0.16 | 0.948250 |
| 0.08 | 0.948249 |

The 0.16 → 0.08 m comparison changes FS by roughly 1.5×10⁻⁶. This is a fixed-geometry integration test, **not** convergence of the slip-surface search or stress smoothing.

## Unit and smoke tests

- Fourteen pytest tests passed (including center-entry circle reconstruction, shortlist determinism, prior material/traction tests, Numba–NumPy geometry comparison).
- Real-data 9×13×7 smoke test passed: 819 trial surfaces, 401 screened valid, 25 FE-edge refinements, and 3 BFGS feasible candidate re-integrations.
- 57,000-candidate real-data run passed with NumBa geometry enabled.

## Scientific interpretation and outstanding work

Kim & Lee (1997), *Computers and Geotechnics* 21(4), pp. 295–313, uses the circular shape variables `(xc,yc,x1)` and BFGS with numerical sensitivities (Eqs. 11–12). The v3 program implements those **variable definitions and search family**, but uses a sagitta-grid for seeds, numerical exit-ground intersections, a finite penalty for invalid candidates and the custom Bishop + Pt equivalent shear strength. It is therefore **Kim–Lee-inspired, not an exact reproduction**.

The next C2 methodological task should focus on optimizer smoothness, reliable feasible line search / gradient estimation, and validation against an original benchmark. C3 non-circular feasible-direction optimization remains pending.
