# KimLee FEM–BBM Modular Pipeline — version 3

The v3 release strengthens C2 with **100 candidate FE-edge refinements** and **optional multistart BFGS circular optimization** based on Kim & Lee (1997), Eq. (11). The A2/B1/B2/C1 and Pt audit implementations are retained from v2. **The numerical FS remains conditional and is not a PLAXIS SRM FS.**

**Start:** `python -m pip install -r requirements.txt` and optionally `python -m pip install -r requirements-numba.txt`, then `python run.py`. Inputs `Phase_12_Step_2287.csv` and `connectivity.txt` are in the parent directory when the folder is extracted under `C:\PLAXIS_ELEM\Phase_12_Fredlund`.

**Configuration:** `config/project.json` sets the active stages, FS mode (`bishop_pt` or `bishop_only`) and circular search parameters. `config/materials.json` has friction/cohesion values. `config/search_domain.json` stores the independently sampled entry and exit intervals. No per-stage CLI flags are needed.

**C2 v3:** After the deterministic sagitta grid, C2 takes top-screened AND geometrically distinct candidates for FE-edge re-evaluation, then can locally refine circles using SciPy BFGS on `(xc,yc,x1)`. Every reported BFGS FS undergoes independent B1/B2 integration. See `CHANGELOG_V3.md` and `METHODOLOGY_AUDIT_C2.md` for details and limitations.

**Outputs** under `results/<snapshot>/C2`: `C2_all_candidates.csv`, `C2_refined_shortlist.csv`, `C2_BFGS_results.csv`, `C2_best_circular_surface.csv`, `C2_best_surface_pointwise.csv`, `C2_circular_search_diagnostics.png`, `C2_report.json`.

**Verification:** `python -m pytest -q` (requires `pytest`) executes unit and regression tests. A complete run requires large PLAXIS input files, not included in the project ZIP.

---

# KimLee FEM–BBM | Modular Stress-Integral Slope Analysis

**Status:** Research implementation/prototype, **not a validated reproduction of Kim & Lee (1997)** and **not a PLAXIS shear-strength reduction FS**. It uses the existing Stage A2–C2 calculations, relocated into a Python package, with separate handling of BBM `Pt`.

## 1. Directory structure

```text
KimLee_BBM_Pipeline/
├── run.py                              # ONE entry point; no CLI stage arguments
├── config/
│   ├── project.json                    # input paths, stages, FS mode, C2 search
│   ├── materials.json                  # Material_Index -> c', phi'
│   └── search_domain.json              # Entry, exit, eligible materials, depths
├── data/inputs/
│   └── B1_example_surface_DIAGNOSTIC.csv
├── src/kimlee/
│   ├── runner.py                       # orchestration, output paths, manifest
│   ├── core/
│   │   ├── mechanics.py                # stress transform + MC-equivalent strength
│   │   └── shape.py                    # P2 triangle interpolation
│   └── stages/
│       ├── a2.py                       # global material-wise P2 stress+Pt projection
│       ├── b1.py                       # prescribed curve, stress+Pt quadrature
│       ├── b2.py                       # FS on prescribed curve
│       ├── c1.py                       # entry-exit domain
│       └── c2.py                       # circular grid+refined search
├── tests/test_mechanics.py
├── requirements.txt
└── results/                            # automatically created, by snapshot name
```

**Stage A1** (exporting CSV + `connectivity.txt` from PLAXIS Output) has **not** been integrated: the extractor source file was not part of the checked refactoring inputs. This project starts with the existing exported files. **Stage C3** (Kim & Lee non-circular optimization) remains future work.

## 2. Windows installation

1. Extract the archive to `C:\PLAXIS_ELEM\Phase_12_Fredlund\KimLee_BBM_Pipeline`.
2. Ensure the folder `C:\PLAXIS_ELEM\Phase_12_Fredlund` already contains `Phase_12_Step_2287.csv` and `connectivity.txt`.
3. Open PowerShell in `KimLee_BBM_Pipeline`.
4. Install requirements once (use the Python environment in which SciPy/Shapely work):

   ```powershell
   python -m pip install -r requirements.txt
   ```

5. Review **three** files: `config/project.json`, `config/materials.json`, `config/search_domain.json`.
6. Run:

   ```powershell
   python run.py
   ```

The `input_dir` in `config/project.json` defaults to `..` (the **parent** folder of `KimLee_BBM_Pipeline`). If installed elsewhere, edit that path; absolute Windows paths are accepted on Windows.

**No separate `python stage_... --a2 ... --out ...` commands are needed.**

## 3. Edit just the JSON configuration

`config/project.json` contains:

```json
"analysis": {
  "strength_formulation": "bishop_pt",
  "stages": ["a2", "b1", "b2", "c1", "c2"],
  "make_plots": true
}
```

- `bishop_pt`: **main model**; use `c' + (sigma_n^B + Pt) tan(phi')`.
- `bishop_only`: explicitly retain old method for regression/audit, `c' + sigma_n^B tan(phi')`.
- `stages`: subset in order. For example, after changing only the material parameters and with correctly generated A2/B1/C1, you may run `['b2','c2']`. If A2 or C1 settings have changed, **run their dependent stages again**. Standard recommendation is to run **all** to avoid stale outputs.
- `make_plots`: toggles A2, B1, B2 figures and high-strain overlay; C1 and C2 currently still save their diagnostic figures.

In `config/search_domain.json` you may adjust `entry_x_range_m`, `exit_x_range_m`, `entry_samples`, `exit_samples`, allowed materials, and depth. The default reproduces the latest user-selected provisional entry **23–46 m (40 samples)** and exit **60–97 m (95 samples)**; this is **not** an automatically determined failure surface.

In `config/project.json`, C2's `sagitta_steps` = 15 by default, with `sagitta_ratio_min` = 0.025 and `sagitta_ratio_max` = 0.35. You may adjust `refine_top_n`, `screen_step_m`, and `refinement_step_m`. These are our **current deterministic sampled-circle search** settings, **not** a reproduction of Kim & Lee's circular BFGS algorithm.

The provided `config/materials.json` uses your latest inputs: index 0 `c'=0,phi'=32°` and index 5 `c'=0,phi'=36°`. Check these parameters against the PLAXIS material assignments before research use; there are no hidden default strengths.

## 4. Dataflow and output

For input `Phase_12_Step_2287.csv`, outputs appear under:

```text
results/Phase_12_Step_2287/
├── A2/
│   ├── A2_global_FE_effective_stress.npz
│   ├── A2_global_FE_effective_stress_report.json
│   └── A2_global_FE_effective_stress_diagnostics.png
├── B1/
│   ├── B1_Bishop_quadrature_points.csv
│   ├── B1_Bishop_validation_report.json
│   └── B1_Bishop_effective_diagnostics.png
├── B2/
│   ├── B2_pointwise_strength.csv
│   ├── B2_report.json
│   └── B2_diagnostics.png
├── C1/
│   ├── C1_entry_exit_candidates.csv
│   ├── C1_admissible_domain.geojson
│   ├── C1_ground_surface.csv
│   ├── C1_config_resolved.json
│   └── C1_search_domain.png
├── C2/
│   ├── C2_all_candidates.csv
│   ├── C2_refined_shortlist.csv
│   ├── C2_best_circular_surface.csv
│   ├── C2_best_surface_pointwise.csv
│   ├── C2_report.json
│   └── C2_circular_search_diagnostics.png
└── run_manifest.json
```

`runner.py` consumes actual disk artifacts from upstream stages. It does not just call the previous stage and pass unsaved Python objects: B1 reads A2 NPZ; B2 reads B1 CSV+report; C1 reads A2 NPZ; C2 reads A2 NPZ and C1 CSV+GeoJSON, and uses B1/B2 functions to refine candidates.

The run manifest includes input/config checksums and primary artifact file hashes. An interrupted run retains completed stages. **Do not reuse an earlier C1 output after changing search bounds/sampling:** C2 checks for stale configuration and raises an error.

## 5. Physics: effective saturation and Pt

Your selected BBM-informed equivalent strength (for `c'=0`) is

```text
FS = ∫_Γ [ (sigma_net + s * (Seff + ks)) * tan(phi') ] ds / ∫_Γ tau_mobilized ds
   = ∫_Γ [ (sigma_n^B + Pt) * tan(phi') ] ds / ∫_Γ tau_mobilized ds
```

provided the same suction/sign and Bishop definitions are used, `Pt=ks*s`, and `sigma_n^B` uses compression-positive normal stress.

- PLAXIS `Sigxx_E`, `Sigyy_E`, `Sigxy` provide **Bishop effective stress**. Effective saturation was included during PLAXIS calculation; **never add `Seff*s` again**.
- PLAXIS `Pt` is separately recovered from original Gauss point state, separately by material; it contributes to **strength**, not to the effective stress tensor.
- `Pt` is **missing** in original CSV for materials 4 and 8. Under `bishop_pt`, using them as allowed slip materials is rejected rather than silently inventing Pt.
- **IMPORTANT numerical treatment:** unconstrained P2 L2 projection of nonnegative Gauss `Pt` can yield negative overshoots, especially near `Pt=0`. This implementation saves raw recovered Pt values, **explicitly applies `Pt_used=max(Pt_recovered,0)` at field evaluation**, and reports raw negative values/counts. This is a new numerical approximation; it is **not validated as an exact BBM field recovery**. Consider a positivity-constrained projection for final publication.
- `B2_report.json` records `strength_formulation` and per-material `integrated_pt_kN_per_m`.
- Signed shear is integrated, not absolutized. Results depend on surface movement direction and are **conditional FS**.

## 6. Validation carried out during this refactor

Using the user's `Phase_12_Step_2287.csv` and `connectivity.txt`:

- 9,258 active triangles; 111,096 GP records; quadratic A2 patch error ~9.78e−12.
- 708 B1 quadrature points, 64.979 m prescribed diagnostic curve.
- `bishop_only` B2: **1.0525704926**, consistent with earlier B2.
- `bishop_pt` B2: **1.0781538317** for the **diagnostic surface**, not the critical surface.
- On a *small 9×14×5 test grid*: new C2 `bishop_only` exactly matched previous C2 FS **0.93072062905**; new `bishop_pt` gave **0.952292**. These numbers are only regression/smoke tests, **not** final C2 results for the 40×95 grid.
- The full C2 40×95×15 search has **not** been rerun with `bishop_pt` in this refactor. The previous 0.926611 result used `bishop_only` and must not be re-used as the new model's FS.

Run fast unit tests without PLAXIS data:

```powershell
python -m unittest discover -s tests -v
```

## 7. Important limitations

- The paper by Kim & Lee (1997) uses a global stress recovery and a stress-integral FS. Our global P2 FE recovery is **not proven identical** to its algorithm.
- C2 is a **sagitta-grid** candidate search, not the paper's circular BFGS optimization; C3 non-circular feasible-direction method is not yet implemented.
- `bishop_pt` is a BBM-informed **equivalent Mohr–Coulomb** post-processing formula, **not** exact evaluation of the BBM yield surface or strength-reduction FS.
- Search domain, positive driving-shear constraints, and local negative-shear rejection criteria require physical validation.
- A2 currently operates on **one PLAXIS snapshot per run**. For a different time step, update `snapshot_csv` and rerun. A multi-time-step FS(t) batch workflow remains future work.

**Source:** Kim & Lee (1997), *An Improved Search Strategy for the Critical Slip Surface using Finite Element Stress Fields*, Computers and Geotechnics 21(4), 295–313.

## v2: Pt recovery audit (recommended next validation stage)

v2 introduces a **non-destructive** audit stage named `pt_audit`, run after
C2 when the default `analysis.stages` list is used. Its files are written to
`results/<snapshot>/PT_AUDIT/`.

The stage compares (i) original PLAXIS Gaussian-point `Pt` against global
unconstrained quadratic P2 recovery and (ii) the resulting *conditional* FS
on two **fixed** surfaces: B1 diagnostic and C2's saved best-refined surface.
It uses the identical signed driving shear `T` and material parameters
throughout. The four Pt evaluation rules are:

| Name | Interpretation |
| --- | --- |
| `raw_P2` | Unconstrained nodal P2 projection evaluated without clipping; diagnostic and may be negative |
| `GP_clipped_P2` | `max(0, raw_P2)` at every integration point; **existing code's current rule** |
| `nodal_clipped_P2` | Clip nodal values before P2 interpolation, then clip any remaining P2 undershoot; sensitivity only |
| `positive_P1_corner` | Clip P2 corner nodes then interpolate linearly; nonnegative but lower order; sensitivity only |

No option is asserted to be more physically accurate. The audit reports the
changes **at fixed geometry**, not the change to the minimum after a new
C2 optimization. Original raw Pt fields remain intact in the A2 NPZ.

You can run the audit alone after your existing A2/B1/C2 run by editing
`config/project.json`:

```json
"analysis": {
  "strength_formulation": "bishop_pt",
  "stages": ["pt_audit"],
  "make_plots": true
}
```

Then simply execute `python run.py`. The rest of the project config is
unchanged. To run everything, set `"stages": ["a2", "b1", "b2", "c1", "c2", "pt_audit"]`.

**Note:** if C2 outputs are absent, the Pt audit still works on B1 and flags
that C2 fixed-geometry sensitivity is unavailable. Do not re-label a
small-test surface as your full-grid critical candidate.

## v2: Optional Numba for C2 numerical geometry

Install optionally:

```powershell
python -m pip install -r requirements-numba.txt
```

Set the following in `config/project.json`:

```json
"circular_search": {
  "use_numba_geometry": true
}
```

(The existing parameters in the same object must remain in place.) Numba
runs only the arc-point generation, local ground interpolation, sliding-body
trapezoidal area, maximum depth and x-monotonicity checks. It does **not**
replace Shapely domain tests, NumPy-based stress traction, PLAXIS field
interpolation, FE-edge refinement or the sparse solver in A2.

The default remains `false` because it requires an optional package and the
NumPy backend serves as the numerical reference. See `VALIDATION_V2.md`
for the actual 819-candidate end-to-end regression and 57,000-candidate
numerical-geometry microbenchmark. First call to Numba incurs JIT compile
cost; later calls use its cached machine code.

In C2, the `refine_top_n` default remains 18, which means only 18 shortlisted
surfaces are integrated at FE-element edges. Increasing it requires separate
refinement-selection and convergence checks, not merely enabling Numba.


## Version 4 — Stage C3 noncircular search

Stage `c3` is now integrated into `python run.py` after `c2`. It uses the C2-best surface
as the *single* seed and optimizes 4, then 7, then 13 vertices. Final candidates are
re-evaluated by Stage B1 FE-edge splitting and Stage B2 strength integration.

The C3 optimizer is **SLSQP + feasible-coordinate search**, not the original Kim &
Lee feasible-direction algorithm. `c3` may report a lower candidate FS but cannot
certify global optimality or solve C2 BFGS precision loss.

- Edit `config/project.json` -> `noncircular_search` to set vertex levels, bounds,
  maximum turning angle and optimizer tolerances.
- To run only C3 on saved outputs, set `analysis.stages` to `["c3"]`, keep existing
  `results/<snapshot>/A2`, `C1`, and `C2`, and use `python run.py`.
- Use `python run_c2_gradient_audit.py` to diagnose saved C2 BFGS gradients; no
  terminal arguments are required.
- See `CHANGELOG_V4.md`, `METHODOLOGY_AUDIT_C3.md`, and `VALIDATION_V4.md`.
