"""Dependency-ordered runner, provenance, and stage-to-stage artifact wiring."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone

from .stages import a2,b1,b2,c1,c2,c3
from .validation import pt_recovery

STAGES = ('a2','b1','b2','c1','c2','c3','pt_audit')


def checksum(path:Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def json_file(path:Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def required_file(path:Path, description:str):
    if not path.is_file():
        raise FileNotFoundError(f'{description} missing: {path}\n'
                                'Check config/project.json paths or run prerequisite stage.')
    return path


def project_file(root:Path, rel):
    p=Path(rel)
    return p if p.is_absolute() else root/p


def run_project(root:Path, *, settings_file='config/project.json'):
    root=Path(root).resolve()
    cfg_path=required_file(project_file(root,settings_file),'Project configuration')
    cfg=json_file(cfg_path)
    paths=cfg['paths']; analysis=cfg['analysis']; search=cfg['circular_search']
    stages=analysis.get('stages',list(STAGES))
    if not stages or len(stages)!=len(set(stages)) or any(s not in STAGES for s in stages):
        raise ValueError('analysis.stages must be a nonempty unique subset of '+str(STAGES))
    if list(stages)!=sorted(stages,key=STAGES.index):
        raise ValueError('Stages must follow dependency order '+str(STAGES))
    mode=analysis.get('strength_formulation','bishop_pt')
    if mode not in ('bishop_pt','bishop_only'):
        raise ValueError('analysis.strength_formulation: bishop_pt or bishop_only')
    make_plots=bool(analysis.get('make_plots',True))
    source_dir=project_file(root,paths['input_dir'])
    gp=required_file(source_dir/paths['snapshot_csv'],'Single-step PLAXIS GP CSV')
    mesh=required_file(source_dir/paths['connectivity'],'PLAXIS connectivity')
    curve=required_file(project_file(root,paths['prescribed_curve']),'B1 prescribed curve')
    materials=required_file(project_file(root,paths['materials']),'Material strength config')
    searchcfg=required_file(project_file(root,paths['search_domain']),'C1 search config')
    out=project_file(root,paths['results_dir'])/gp.stem
    out.mkdir(parents=True,exist_ok=True)
    manifest={
        'project':cfg.get('project_name','Unnamed'),
        'started_utc':datetime.now(timezone.utc).isoformat(),
        'config_sha256':checksum(cfg_path),
        'config_file':str(cfg_path),'strength_formulation':mode,
        'snapshot_csv':str(gp),
        'snapshot_sha256':checksum(gp),
        'connectivity_sha256':checksum(mesh),
        'materials_sha256':checksum(materials),
        'search_config_sha256':checksum(searchcfg),
        'stages_requested':stages,
        'stage_outputs':{},
        'warnings':[
            'Stress-integral FS is conditional, not exact BBM yield FS or PLAXIS strength reduction.',
            'C2 sagitta grid provides seeds; optional BFGS uses Kim-Lee Eq (11) variables but penalty feasibility is our implementation.',
            'C3 implements Kim-Lee variable count and 4->7->13 vertex refinement, but uses SLSQP + feasible coordinate search, not the original feasible-direction solver.',
            'Current input day may be post-failure; verify physical interpretation.',
            'Pt recovered by unconstrained P2 projection is evaluated with an explicit nonnegative cap.'
        ]}
    print('\n'+ '='*72)
    print('KimLee FEM–BBM |',gp.name,'| formulation:',mode)
    print('Stages:',' -> '.join(stages))
    print('Results:',out)
    print('='*72)
    for stage in stages:
        target=out/stage.upper()
        target.mkdir(exist_ok=True)
        print('\n' + '-'*20 + ' Stage ' + stage.upper() + ' ' + '-'*20,flush=True)
        if stage=='a2':
            artifact=a2.run(gp,mesh,target,make_plot=make_plots)
        elif stage=='b1':
            artifact=b1.run_stage(
                required_file(out/'A2'/'A2_global_FE_effective_stress.npz','A2 field'),
                curve,target,gp_csv=gp if make_plots else None,make_plot=make_plots)
        elif stage=='b2':
            artifact=b2.run_stage(
                required_file(out/'B1'/'B1_Bishop_quadrature_points.csv','B1 quadrature'),
                target,materials,formulation=mode,make_plot=make_plots)
        elif stage=='c1':
            ns=SimpleNamespace(a2=str(required_file(out/'A2'/'A2_global_FE_effective_stress.npz','A2 field')),
                config=str(searchcfg),connectivity=str(mesh),
                gp_csv=str(gp) if make_plots else None,
                b1_curve=str(curve) if make_plots else None,out=str(target))
            c1.run(ns)
            artifact=target/'C1_config_resolved.json'
        elif stage=='c2':
            ns=SimpleNamespace(a2=str(required_file(out/'A2'/'A2_global_FE_effective_stress.npz','A2 field')),
                c1_dir=str(out/'C1'),c1_config=str(searchcfg),materials=str(materials),
                gp_csv=str(gp) if make_plots else None,out=str(target),
                strength_formulation=mode,
                sagitta_ratio_min=float(search['sagitta_ratio_min']),
                sagitta_ratio_max=float(search['sagitta_ratio_max']),
                sagitta_steps=int(search['sagitta_steps']),
                step=float(search['screen_step_m']),
                refine_step=float(search['refinement_step_m']),
                refine_top=int(search['refine_top_n']),
                refine_diverse_fraction=float(search.get('refine_diverse_fraction',0.25)),
                refine_diverse_pool=int(search.get('refine_diverse_pool',4000)),
                refine_entry_separation_m=float(search.get('refine_entry_separation_m',1.0)),
                refine_exit_separation_m=float(search.get('refine_exit_separation_m',1.5)),
                refine_ratio_separation=float(search.get('refine_ratio_separation',0.015)),
                bfgs_enabled=bool(search.get('bfgs',{}).get('enabled',False)),
                bfgs_starts=int(search.get('bfgs',{}).get('starts',3)),
                bfgs_maxiter=int(search.get('bfgs',{}).get('maxiter',35)),
                bfgs_fd_step_m=list(search.get('bfgs',{}).get('finite_difference_steps_m',[0.025,0.025,0.015])),
                bfgs_entry_sep_m=float(search.get('bfgs',{}).get('seed_entry_separation_m',1.0)),
                bfgs_exit_sep_m=float(search.get('bfgs',{}).get('seed_exit_separation_m',1.5)),
                bfgs_ratio_sep=float(search.get('bfgs',{}).get('seed_sagitta_ratio_separation',0.015)),
                min_sliding_area=float(search['min_sliding_area_m2']),
                min_depth=float(search['min_depth_m']),
                max_negative_force_fraction=float(search['max_negative_force_fraction']),
                geom_tolerance=float(search['geometry_tolerance_m']),
                use_numba_geometry=bool(search.get('use_numba_geometry',False)))
            c2.run(ns)
            artifact=target/'C2_report.json'
        elif stage=='c3':
            c3cfg=cfg.get('noncircular_search',{})
            ns=SimpleNamespace(
                a2=str(required_file(out/'A2'/'A2_global_FE_effective_stress.npz','A2 field')),
                c1_dir=str(out/'C1'),c1_config=str(searchcfg),
                c2_dir=str(out/'C2'),materials=str(materials),out=str(target),
                strength_formulation=mode,
                vertex_levels=list(map(int,c3cfg.get('vertex_levels',[4,7,13]))),
                x_trust_m=float(c3cfg.get('x_trust_m',2.)),
                y_trust_m=float(c3cfg.get('y_trust_m',2.)),
                min_vertex_dx_m=float(c3cfg.get('min_vertex_dx_m',.25)),
                max_turning_angle_deg=float(c3cfg.get('max_turning_angle_deg',50.)),
                min_sliding_area_m2=float(search['min_sliding_area_m2']),
                min_depth_m=float(search['min_depth_m']),
                geometry_tolerance_m=float(search['geometry_tolerance_m']),
                screen_step_m=float(c3cfg.get('screen_step_m',.8)),
                max_negative_force_fraction=float(search['max_negative_force_fraction']),
                slsqp_maxiter=int(c3cfg.get('slsqp_maxiter',14)),
                ftol=float(c3cfg.get('ftol',1e-5)),
                fd_step_m=float(c3cfg.get('fd_step_m',.025)),
                pattern_passes=int(c3cfg.get('pattern_passes',3)),
                pattern_step_m=float(c3cfg.get('pattern_step_m',.5)))
            artifact=c3.run(ns)
        elif stage=='pt_audit':
            artifact=pt_recovery.run(
                required_file(out/'A2'/'A2_global_FE_effective_stress.npz','A2 field'),
                out,materials,target,make_plot=make_plots)
        required_file(Path(artifact),f'{stage.upper()} output')
        manifest['stage_outputs'][stage]={'primary_file':str(artifact),
                                          'sha256':checksum(Path(artifact))}
        (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    manifest['finished_utc']=datetime.now(timezone.utc).isoformat()
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('\nAll requested stages complete. Manifest:',out/'run_manifest.json')
    return manifest
