# v2 Validation — real Jayapura–Wamena snapshot and regression tests

Snapshot: `Phase_12_Step_2287.csv`, day **100.410760986931**.
Friction parameters: material 0 = 32°, material 5 = 36°, c'=0 in both.
Formulation: conditional **Bishop effective + BBM Pt**.

## 1. Pt Gaussian-point recovery

| Material | Pt original GP negative | P2 recovered negative | Recovered min Pt (kPa) | Recovered negative GP fraction |
| --- | ---: | ---: | ---: | ---: |
| 0 (Sandy Clay) | 0 | 711 / 37,260 | -0.188818 | 1.908% |
| 5 (Silty Clay) | 0 | 9,894 / 42,420 | -1.240481 | 23.324% |

Pt is originally nonnegative in both materials. An unconstrained global P2
least-squares projection is **not positivity-preserving**. The count does not
imply the same area or energy fraction, nor proof of geophysical inaccuracy.

## 2. Fixed-geometry FS sensitivity

| Pt rule | B2 diagnostic FS (708 points) | C2 candidate #54949 FS (927 points) |
| --- | ---: | ---: |
| Raw unconstrained P2 | 1.078094327 | 0.949154605 |
| GP-clipped P2 (the current pipeline) | **1.078153832** | **0.949201310** |
| Clip nodal P2 then evaluate and clip | 1.078205234 | 0.949225958 |
| Positive P1 from clipped P2 corner nodes | 1.079590727 | 0.950255600 |

C2 candidate #54949 was reconstructed using the exact 40×95×15 grid
settings from the user's successful run: entry (45.41025641, 69.02307692),
exit (80.86170213, 41.75886525), sagitta 4.23270775 m, radius 61.1844 m.
The reconstructed `bishop_pt` FS and integrated forces match the user's log:
FS 0.949201310, R 1120.00535 kN/m, T 1179.94502 kN/m.

The *fixed C2 candidate* does not establish that a newly optimized critical
surface would have the same ranking or minimum with an alternative Pt rule.

## 3. Numba regression

Test on the actual A2 and C1 geometry, using a **9×13×7 = 819 candidate**
search, with four FE-edge refinements in each run:

- NumPy and Numba screened exactly 401 candidates as valid.
- Rejection categories and counts matched exactly.
- Each candidate's screening FS, resistance, signed driving shear, sliding
  area and maximum depth matched exactly within printed CSV precision.
- Both returned candidate #709 with conditional FS 0.984450.
- `test_fast_geometry.py`: NumPy vs Numba pointwise and area differences
  <= 1e-10 and 1e-9 m² respectively on representative arcs.

**Numeric geometry only** benchmark (no I/O, Shapely, TriFinder, FE stress
integration, plotting or JIT first-call compilation), 57,000 arcs on the
same host: NumPy **2.18 s**, Numba **0.277 s** (~**7.9×** faster for this
subset). These times are *not* the full C2 pipeline speedup and will vary
by machine and version. Numba's first run includes compilation overhead.

## 4. Tests and scientific scope

All **11 tests passed**, including the four added in v2.
No changes were made to the constitutive strength approximation, A2
smoothing algorithm or the existing C2 search strategy. These numerical
checks do not prove equivalence to Kim & Lee (1997) BFGS circular/noncircular
optimization or an exact BBM factor of safety.
