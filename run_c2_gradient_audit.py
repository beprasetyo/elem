"""Run a finite-difference sensitivity audit of previously saved C2 BFGS starts.
No CLI arguments: paths are read from config/project.json, C1, C2 and A2.
Provides *evidence of gradient sensitivity*; does not claim to fix BFGS.
"""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from shapely.geometry import LineString

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from kimlee.stages.b1 import BishopField
from kimlee.stages.c2 import load_c1,load_params,arc_polyline,arc_tangent,integrate_arc
from kimlee.stages.c2_bfgs import reconstruct_center_entry


def run():
    cfg=json.loads((ROOT/'config/project.json').read_text(encoding='utf8'))
    p=cfg['paths']; s=cfg['circular_search']; mode=cfg['analysis']['strength_formulation']
    stem=Path(p['snapshot_csv']).stem
    res=ROOT/p['results_dir']/stem
    a2=res/'A2/A2_global_FE_effective_stress.npz'
    c2dir=res/'C2'
    if not (c2dir/'C2_report.json').exists():
        raise FileNotFoundError('Missing C2_report.json. Run C2 first.')
    c2=json.loads((c2dir/'C2_report.json').read_text(encoding='utf8'))
    cc,entries,exits,region,surface=load_c1(res/'C1',ROOT/p['search_domain'])
    field=BishopField(a2);strength=load_params(ROOT/p['materials'],cc['allowed_material_indices'])
    rows=[]
    for run in c2.get('BFGS_runs',[]):
        if not run.get('optimization_variables'):continue
        q=np.array(run['optimization_variables'],float)
        def fun(qv):
            try:
                g,A,B,sag=reconstruct_center_entry(qv,surface,cc['entry_x_range_m'],
                        cc['exit_x_range_m'],float(run['exit_x']))
                theta,xy=arc_polyline(g,float(s['screen_step_m']))
                if np.min(np.diff(xy[:,0]))<=0 or not region.buffer(float(s['geometry_tolerance_m'])).covers(LineString(xy)):
                    return np.nan,'GEOMETRY'
                data,status=integrate_arc(g,field,strength,theta,mode)
                if status!='VALID_SCREENED':return np.nan,status
                return data['FS_screened'],'OK'
            except (ValueError,FloatingPointError,IndexError) as exc:
                return np.nan,'INVALID: '+str(exc)[:90]
        base,status=fun(q)
        for h in [.1,.05,.025,.01,.005,.001]:
            grad=[];statuses=[]
            for j in range(3):
                v=np.zeros(3);v[j]=h
                pp,statp=fun(q+v);mm,statm=fun(q-v)
                grad.append(float((pp-mm)/(2*h)) if np.isfinite(pp) and np.isfinite(mm) else np.nan)
                statuses.append(f'{statp}|{statm}')
            rows.append({'seed_id':run.get('seed_id'),'h_m':h,'base_fs':base,
              'dFS_dxc':grad[0],'dFS_dyc':grad[1],'dFS_dx1':grad[2],
              'statuses':';'.join(statuses)})
    out=res/'C2_GRADIENT_AUDIT';out.mkdir(exist_ok=True)
    df=pd.DataFrame(rows);df.to_csv(out/'C2_gradient_step_sensitivity.csv',index=False)
    valid=df[['dFS_dxc','dFS_dyc','dFS_dx1']].notna().all(axis=1).sum() if len(df) else 0
    report={'stage':'C2 BFGS sensitivity diagnostic',
      'n_finite_gradient_triplets':int(valid),'n_total_cases':int(len(df)),
      'caution':'FD gradients may change sign with h and feasibility; no BFGS convergence claim.'}
    (out/'C2_gradient_audit_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print('Gradient diagnostic:',out,report)
    return out

if __name__=='__main__':run()
