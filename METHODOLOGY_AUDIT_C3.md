# Methodology audit — Kim & Lee (1997) versus Stage C3 v4

Source: Kim & Lee, *An Improved Search Strategy for the Critical Slip Surface Using Finite Element Stress Fields*, Computers and Geotechnics 21(4), 295–313 (1997), pp. 301–304. The user's original paper was inspected directly.

| Paper | Stage C3 v4 | Status |
|---|---|---|
| Eq. (13): minimize FS of a noncircular surface | `minimize()` objective uses stress-integral FS | Concept matches; adopted Bishop + Pt strength differs |
| Shape vector with N x coordinates and N−2 internal y (2N−2 variables) | `to_variables` / `from_variables` | Implemented |
| Eq. (14): upper/lower bounds controlled by terrain/firm block | C1 polygon and ground/depth rejection; trust boxes | Adaptation; not the same analytic variable bounds |
| Eq. (15): neighboring x-order control | minimum strictly positive `dx` | More restrictive than paper's printed constraint |
| Eq. (16): adjacent-segment angle restrictions | maximum absolute turning-angle control | Related concept, but not the same angle convention/bounds |
| Feasible-direction method with FD gradients | SciPy SLSQP + feasibility-preserving coordinate pattern search | **Different algorithm**; no claim of exact reproduction |
| Progressive vertex insertion | N = 4, 7, 13 by midpoint insertion | Implemented |
| Reuse preceding optimum for next stage | Previous level's selected best feasible candidate is seed | Implemented |
| Multiple initial surfaces (paper Step 6) | Single C2 best surface only in v4 | **Not yet implemented** |
| FEM stress-field line integration | A2 P2 projection, screened GL3, final B1 FE-edge/B2 | Implemented in adapted form; A2 smoothing not verified identical |

## C2 BFGS convergence (outstanding)

The original paper's circular objective uses three physical shape variables `(xc,yc,x1)`, BFGS and finite-difference derivatives (pp. 301–302). C2 v3 uses this shape vector but evaluates a penalized objective with hard domain and material screening. At the best candidate, estimated central-difference gradients vary considerably when the step varies. The C2 BFGS solver does **not** report successful convergence.

The v4 gradient-audit script (`run_c2_gradient_audit.py`) reports valid/invalid stencil perturbations and gradient components. This diagnosis is useful before replacing or reformulating the circular optimizer. Even when a C2 BFGS candidate has lower FE-verified FS, it remains a *feasible candidate*, not a solver-verified circular minimum.

## Interpretation

Stage C3 estimates an **admissible low-FS noncircular surface** for a prescribed stress snapshot under an adapted stress-integral criterion. It does not prove global criticality, reproduce the full BBM yielding physics, or establish true failure time. For research use, additional initial surfaces, pre-failure snapshots, mesh/recovery sensitivity, convergence checks and constitutive benchmarking remain necessary.
