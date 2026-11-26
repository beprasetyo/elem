# v3 — C2 shortlist FE-edge audit + circular BFGS

## Scope

This version changes **only the Stage C2 circular search and central configuration**. It does not change A2 stress recovery, B1 FE-edge integration, B2's selected Bishop + Pt strength formula, C1 geometry, or the separate Pt audit.

1. Increase default FE-edge refinement budget from 18 to **100**.
2. Select candidates using **screening rank plus geometric diversity**: the first 75 are the best screened circles; the remaining budget preferentially samples geometrically separated circles. This is intended to reduce reliance on one narrow local basin.
3. Introduce an optional **multistart circular BFGS** phase based on the shape variables (xc, yc, x1), as in Eq. (11) of Kim & Lee (1997). Calculate derivatives by finite differences, as in Eq. (12). Each optimized candidate is independently checked with Stage B1 and Stage B2.
4. Maintain the existing deterministic sagitta-grid search as the **initialization/baseline**. It is not described as the Kim & Lee algorithm itself.
5. Export `C2_BFGS_results.csv`, include convergence metadata in `C2_report.json`, and maintain old `C2_all_candidates.csv`, `C2_refined_shortlist.csv`, `C2_best_surface_pointwise.csv` and visual diagnostic.

### C2 new config settings

In `config/project.json` inside `circular_search`:

```json
"refine_top_n": 100,
"refine_diverse_fraction": 0.25,
"refine_diverse_pool": 4000,
"refine_entry_separation_m": 1.0,
"refine_exit_separation_m": 1.5,
"refine_ratio_separation": 0.015,
"bfgs": {
  "enabled": true,
  "starts": 3,
  "maxiter": 35,
  "finite_difference_steps_m": [0.025, 0.025, 0.015],
  "seed_entry_separation_m": 1.0,
  "seed_exit_separation_m": 1.5,
  "seed_sagitta_ratio_separation": 0.015
}
```

`bfgs.enabled = false` disables BFGS and leaves a deterministic rank-plus-diversity candidate search. The speedup from Numba remains optional through `use_numba_geometry` and does not affect stress integration routines.

## Crucial research caveats

- **This is not a full reproduction of Kim & Lee.** Eq. (11) shape variables and BFGS / numerical gradients are now represented, but the handling of forbidden domains via penalty objective, the exit intersection selection, initialization, and BFGS tolerance are our implementation decisions.
- Geometry and constitutive constraints can make the FS objective discontinuous; a local optimizer may stop with `solver_success=false` even when it finds a feasible improved point. `refinement_status=OK` and `solver_success=true` are **different statements**.
- C2 FS still uses the Bishop + Pt equivalent MC shear strength. That is a research adaptation, not an exact BBM yield-based safety factor. Current slopes are post-failure day 100.410761.
- The best of 100 refinements (plus BFGS) cannot prove a global minimum among 57,000 trial circles or all feasible circular surfaces.
- C3 noncircular feasible-direction optimization is not implemented.

## Regression and test strategy

Use a frozen A2 field and C1 geometry to compare (1) original 18-only, (2) 100 rank+diverse, and (3) rank+diverse+BFGS. Compare the screened and refined minima, entry/exit, radius, material crossings, and signed shear. Re-run A2 and C1 only when input or configuration changes. Verify full `C2_report.json` and `C2_BFGS_results.csv` for objective/optimizer convergence reports.
