# Validation log — refactor smoke tests

Test environment: Python with numpy/pandas/scipy/matplotlib/shapely; original supplied
`Phase_12_Step_2287.csv` and `connectivity.txt` used **externally** (not included in ZIP).

## Real-data numerical checks

| Check | Result |
|---|---|
| A2 active triangles | 9,258 |
| A2 GP points | 111,096 |
| A2 patch test max error | ~9.78×10⁻¹² |
| A2 Bishop checks (total-effective-PActive) | ≤1.03×10⁻¹¹ |
| A2 Pt/Suction median ratio, materials 0 and 5 | 0.270000 (each) |
| Materials whose GP Pt is missing | 4 and 8 (excluded from search) |
| A2 recovered raw Pt minimum | −1.24048 kPa (projection undershoot) |
| A2 negative Pt GP recovered points | 10,605 (reported, nonnegative eval cap) |
| B1 quadrature points | 708 |
| B1 original diagnostic surface length | 64.978915 m |
| B1 raw Pt minimum | −0.114324 kPa |
| B1 Pt quadrature points capped to zero | 223 / 708 |
| B2 bishop_only diagnostic FS | 1.052570492592733 |
| B2 bishop_pt diagnostic FS | 1.0781538317225494 |
| B2 signed mobilized integral | 2,158.916267 kN/m |

## C2 smoke/regression test

A deliberately **small, non-research** C2 grid (9 entry points × 14 exit points × 5 sagitta ratios = 630 trial circles) was used to test a complete run quickly. Same admissible geometry and material properties in both modes.

| Implementation/mode | Best refined FS |
|---|---:|
| Old standalone C2, bishop_only | 0.9307206290476553 |
| Refactored C2, bishop_only | 0.9307206290476553 |
| Refactored C2, bishop_pt | ~0.952292 |

The `bishop_only` refactor reproduces the old stand-alone C2 **exactly** at the stored precision on this grid. The `bishop_pt` mode was tested separately and changes the strength as expected.

**Not a full scientific validation:** these tests do not establish the correct failure surface, exact BBM strength, equivalence to the Kim–Lee optimization, or convergence of the 40×95×15 research grid. A high-priority future study is replacing the Pt clipping workaround with nonnegative constrained stress-state recovery and examining the effect on FS.
