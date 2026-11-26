# Methodology audit of Stage C2 against Kim & Lee (1997)

**Reference**: J. Y. Kim and S. R. Lee, “An Improved Search Strategy for the Critical Slip Surface Using Finite Element Stress Fields,” *Computers and Geotechnics*, 21(4), 295–313, 1997. Original supplied PDF `1-s2.0-S0266352X9700027X-main.pdf`.

| Item | Kim & Lee paper | This v3 implementation | Classification |
|---|---|---|---|
| FS, Eq. (1), p. 297 | Shear strength / mobilized shear, integrated along slip surface | B1/B2 numerical integration of recovered PLAXIS field with signed shear | Core method, with BBM strength adaptation |
| Mohr-Coulomb, Eq. (2), p. 297 | `tau_f=c+sigma_n tan(phi)` | `tau_f=c+(sigma_n^B+Pt)tan(phi)` | **Original research adaptation**, not the authors' constitutive law |
| Global smoothing, Eq. (3), p. 298 | Continuous stress field through nodal interpolation | Material-separated global FE P2 L2 projection | Similar purpose, different numerical implementation to audit |
| FE segment quadrature, Eq. (4)–(6), pp. 298–299 | Segment split at element boundary; numerical stress integration | B1 exact line–edge splitting and three-point Gaussian quadrature | Methodologically aligned in principle |
| Circular optimization, Eq. (11), p. 301 | Minimize `FS(xc,yc,x1)`; center coordinates and start-x | **New in v3**: multistart BFGS in exactly those physical variables | Follows variable choice, algorithm family |
| Approximate sensitivities, Eq. (12), p. 302 | Forward differences for shape variable derivatives | SciPy BFGS with forward numerical finite-difference `eps` | Aligned in principle; difference increment is an implementation choice |
| Initial search, pp. 303–304 | Select initial circles using stress/displacement; BFGS circular search | Deterministic sagitta-grid screening as multistart initializer | **Different**, documented, not claimed to be from Kim & Lee |
| Feasibility | Circular coordinates practically constrained by ground/hard stratum | C1 search polygon, material exclusion, x-monotonicity, moving-mass area and signed shear filters; barriers in objective | **Our additional constraints** |
| Noncircular, Eq. (13)–(16), pp. 302–303 | Feasible-direction method, angle/y constraints | Not implemented | Pending Stage C3 |
| Vertex refinement, steps 1–6, pp. 303–304 | Introduce midpoint vertices successively | Not implemented | Pending Stage C3 |

### Why sagitta is retained

The sagitta grid is only a reproducible source of initial circular shapes and an independent diagnostic baseline. Kim & Lee's circular optimization uses the variables `(xc, yc, x1)`, not a sagitta-grid minimization. Rewriting every grid coordinate from sagitta to radius would not by itself reproduce Eq. (11). The new BFGS phase works with the actual center and initial point; the exit is found where the lower circle branch meets the C1 ground-surface polyline in the allowed exit interval.

### Geometrical and optimizer caveats

The BFGS phase applies a finite rejection penalty for candidates outside C1 or lacking an allowed exit, because SciPy's `BFGS` is unconstrained. Penalties and FE-element transitions make the objective nonsmooth: it is possible to obtain a useful feasible candidate without the solver declaring analytic convergence. This should **not** be misrepresented as a verified reproduction of the original code or a global optimum.

### Remaining work before claiming faithful Kim & Lee reproduction

Reproduce a benchmark case from the paper on homogeneous dry slopes using the **original Mohr–Coulomb formulation** and compare optimum critical geometries; then implement noncircular feasible-direction optimization with vertex constraints and repeated midpoint insertion (their Fig. 4 and steps 1–6). Examine stress-smoothing alternatives and stress-integration convergence separately.
