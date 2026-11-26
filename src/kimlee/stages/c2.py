"""Stage C2: deterministic circular candidate search and FE-edge refinement.

Circular search uses a sampled sagitta parameterization, an IMPLEMENTATION
CHOICE, not Kim & Lee (1997)'s BFGS circular optimization. Both FS strength
formulations are available through central configuration.
"""
from __future__ import annotations
import json
import math
from collections import Counter
from pathlib import Path
from shapely.geometry import shape, LineString
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from .b1 import BishopField, run_b1
from .b2 import read_strength, compute as b2_compute
from ..core.mechanics import resolve_bishop_traction, equivalent_shear_strength
from ..core.fast_geometry import arc_screen, HAS_NUMBA

GL3_X, GL3_W = np.polynomial.legendre.leggauss(3)


def circle_geometry(entry, exit_, sagitta):
    """Return center, R, alpha, u, n for a circle through two endpoints.

    Arc is theta in [-alpha,alpha]: C + R(sin(theta)*u + cos(theta)*n),
    where u=(exit-entry)/length and n=(u_y,-u_x) points below the chord.
    """
    A=np.asarray(entry,float); B=np.asarray(exit_,float)
    chord=B-A; L=float(np.linalg.norm(chord))
    if L<=1e-10 or sagitta<=0 or not np.isfinite(sagitta):
        raise ValueError('Invalid circular arc span/sagitta')
    u=chord/L; down=np.array([u[1],-u[0]])
    R=L*L/(8*sagitta)+sagitta/2
    C=(A+B)/2+(sagitta-R)*down
    alpha=2*math.atan2(2*sagitta,L)
    return C, R, alpha, u, down


def arc_points(g, theta):
    C,R,_,u,n=g
    theta=np.asarray(theta,float)
    return C[None,:]+R*(np.sin(theta)[:,None]*u[None,:] + np.cos(theta)[:,None]*n[None,:])


def arc_tangent(g, theta):
    _,_,_,u,n=g
    theta=np.asarray(theta,float)
    return np.cos(theta)[:,None]*u[None,:]-np.sin(theta)[:,None]*n[None,:]


def arc_polyline(g, max_step):
    _,R,alpha,_,_=g
    n_seg=max(2, int(math.ceil(2*R*alpha/max_step)))
    theta=np.linspace(-alpha,alpha,n_seg+1)
    return theta,arc_points(g,theta)


def integrate_arc(g, field, strength, theta, formulation='bishop_pt'):
    # Smooth circular-arc tangent, Gaussian integration in arc angle.
    C,R,alpha,u,n=g
    mids=(theta[:-1]+theta[1:])/2
    half=(theta[1:]-theta[:-1])/2
    theta_q=(mids[:,None]+half[:,None]*GL3_X[None,:]).ravel()
    w=(R*half[:,None]*np.broadcast_to(GL3_W[None,:],(len(half),3))).ravel()
    xy=arc_points(g,theta_q)
    tang=arc_tangent(g,theta_q)
    if np.any(tang[:,0]<=0):
        raise ValueError('Non-monotone X derivative of arc')
    ti,stress=field.evaluate(xy)
    mids_index=field.mid[ti]
    if np.any(~np.isin(mids_index, list(strength))):
        raise ValueError('Quadrature point in material without explicit strength')
    sigma_n,tau=resolve_bishop_traction(stress,tang)
    if np.any(~np.isfinite(sigma_n)) or np.any(~np.isfinite(tau)):
        raise ValueError('Non-finite interpolated stress')
    if np.min(sigma_n)<-1e-8:
        return None,'REJECT_TENSILE_EFFECTIVE_NORMAL'
    if formulation=='bishop_pt':
        pt=field.evaluate_pt(xy)
        if not np.isfinite(pt).all() or np.any(pt < -1e-6):
            raise ValueError('Missing/nonfinite/negative Pt on circular surface')
    elif formulation=='bishop_only':
        pt=np.zeros(len(w))
    else:
        raise ValueError('Invalid strength formulation')
    shear_strength=np.zeros(len(w))
    for m,p in strength.items():
        loc=mids_index==m
        shear_strength[loc]=equivalent_shear_strength(sigma_n[loc],pt[loc],p['c_kPa'],p['phi_deg'])
    T=float(np.dot(w,tau)); Rf=float(np.dot(w,shear_strength))
    Tplus=float(np.dot(w,np.maximum(tau,0)))
    Tminus=float(np.dot(w,np.maximum(-tau,0)))
    if T<=1e-6:
        return None,'REJECT_NONPOSITIVE_SIGNED_DRIVING'
    out={
      'FS_screened':Rf/T,
      'R_kN_per_m':Rf,'T_signed_kN_per_m':T,
      'integrated_Pt_kN_per_m':float(np.dot(w,pt)),
      'T_positive_kN_per_m':Tplus,
      'T_negative_kN_per_m':Tminus,
      'negative_force_fraction':Tminus/(Tplus+Tminus) if Tplus+Tminus>0 else 0.,
      'negative_length_fraction':float(w[tau<0].sum()/w.sum()),
      'sigma_normal_min_kPa':float(np.min(sigma_n)),
      'sigma_normal_max_kPa':float(np.max(sigma_n)),
      'material_indices':','.join(map(str,sorted(np.unique(mids_index).tolist()))),
      'quadrature_points':int(len(w)),
      'quad_integrated_length_m':float(np.sum(w)),
    }
    return out,'VALID_SCREENED'


def self_tests():
    A=np.array([33.,69.5]);B=np.array([94.,35.]);s=5.
    g=circle_geometry(A,B,s)
    p=arc_points(g,np.array([-g[2],0.,g[2]]))
    assert np.max(np.abs(p[0]-A))<1e-10
    assert np.max(np.abs(p[2]-B))<1e-10
    assert np.max(np.abs(p[1]-(A+B)/2-s*g[4]))<1e-10
    t=arc_tangent(g,np.linspace(-g[2],g[2],70))
    assert np.max(np.abs(np.linalg.norm(t,axis=1)-1))<1e-12
    assert np.min(t[:,0])>0
    theta,pts=arc_polyline(g,.5)
    assert abs(g[1]*2*g[2]-np.sum(g[1]*np.diff(theta)))<1e-10
    return {'circle_endpoints_max_error_m':float(np.max(np.abs(p[[0,2]]-np.stack([A,B])))),
            'unit_tangent_max_error':float(np.max(np.abs(np.linalg.norm(t,axis=1)-1)))}


def load_c1(c1_dir,cfgpath):
    """Load C1 results and fail if they were generated from an older config.

    C2 deliberately reads entry/exit points from C1's CSV. Merely editing
    entry_samples / exit_samples in JSON does NOT change the saved grid.
    Also supports legacy samples_per_zone in saved and current configurations.
    """
    config=json.loads(Path(cfgpath).read_text(encoding='utf-8'))
    candidates=pd.read_csv(c1_dir/'C1_entry_exit_candidates.csv')
    if not {'zone','X','Y_ground'}.issubset(candidates.columns):
        raise ValueError('C1 candidate CSV incomplete')
    entry=candidates.loc[candidates.zone=='entry',['X','Y_ground']].to_numpy(float)
    exit_=candidates.loc[candidates.zone=='exit',['X','Y_ground']].to_numpy(float)
    if not len(entry) or not len(exit_):
        raise ValueError('Missing C1 entry or exit points')

    expected_entry=int(config.get('entry_samples',config.get('samples_per_zone',21)))
    expected_exit=int(config.get('exit_samples',config.get('samples_per_zone',21)))
    if expected_entry < 2 or expected_exit < 2:
        raise ValueError('entry_samples and exit_samples must both be >= 2')
    xentries=np.array(config['entry_x_range_m'],float)
    xexits=np.array(config['exit_x_range_m'],float)
    if len(entry)!=expected_entry or len(exit_)!=expected_exit:
        raise ValueError(
            f'C1/C2 GRID MISMATCH: JSON requests {expected_entry} entry and '
            f'{expected_exit} exit points, but C1_results contains '
            f'{len(entry)} entry and {len(exit_)} exit points. '
            'Rerun Stage C1 with this JSON before C2.')
    for zone, actual, bounds, n in [
        ('entry',entry,xentries,expected_entry),('exit',exit_,xexits,expected_exit)]:
        expected_x=np.linspace(bounds[0],bounds[1],n)
        if not np.allclose(actual[:,0],expected_x,atol=1e-6,rtol=0):
            raise ValueError(
                f'C1/C2 {zone.upper()} GRID MISMATCH: saved coordinates '
                'do not match the requested range and sample count. Rerun Stage C1.')

    resolved_path=c1_dir/'C1_config_resolved.json'
    if not resolved_path.exists():
        raise ValueError('C1_config_resolved.json missing; rerun Stage C1 first.')
    resolved=json.loads(resolved_path.read_text(encoding='utf-8'))
    resolved_entry=int(resolved.get('entry_samples',resolved.get('samples_per_zone',21)))
    resolved_exit=int(resolved.get('exit_samples',resolved.get('samples_per_zone',21)))
    if (expected_entry,expected_exit)!=(resolved_entry,resolved_exit):
        raise ValueError(
            f'C1 DOMAIN IS STALE: current sample counts '
            f'({expected_entry},{expected_exit}) differ from saved C1 '
            f'({resolved_entry},{resolved_exit}). Rerun Stage C1 first.')
    for key in ['entry_x_range_m','exit_x_range_m','allowed_material_indices',
                'maximum_vertical_depth_below_ground_m',
                'surface_probe_depth_m','plot_x_range_m']:
        if key not in config or key not in resolved:
            raise ValueError(f'Missing {key} in current or saved C1 configuration; rerun C1.')
        v=config[key]; old=resolved[key]
        if isinstance(v,list):
            mismatch=not np.allclose(np.asarray(v,float),np.asarray(old,float),rtol=0,atol=1e-10)
        else:
            mismatch=not np.isclose(float(v),float(old),rtol=0,atol=1e-10)
        if mismatch:
            raise ValueError(
                f'C1 DOMAIN IS STALE: {key} in current JSON = {v}, '
                f'but saved C1_config_resolved.json = {old}. '
                'Rerun stage_C_01_search_domain_independent.py before C2.')

    geo=json.loads((c1_dir/'C1_admissible_domain.geojson').read_text(encoding='utf-8'))
    chosen=[f for f in geo['features'] if f['properties'].get('role')=='admissible_slip_region']
    if len(chosen)!=1:raise ValueError('C1 must have exactly one admissible polygon feature')
    poly=shape(chosen[0]['geometry'])
    if not poly.is_valid or poly.is_empty:raise ValueError('Invalid C1 admissible polygon')
    surface=pd.read_csv(c1_dir/'C1_ground_surface.csv')[['X','Y_ground']].to_numpy(float)
    if np.any(np.diff(surface[:,0])<=0):raise ValueError('C1 ground surface not x-monotonic')
    print(f'C1 verified: {len(entry)} entry x {len(exit_)} exit = '
          f'{len(entry)*len(exit_)} endpoint pairs.')
    return config,entry,exit_,poly,surface


def load_params(path,allowed):
    # Reuse Stage B2's explicit validation (no default/fallback strength values).
    p=read_strength(Path(path),np.asarray(allowed,int))
    if set(p)!=set(allowed):raise ValueError('Strength JSON material keys mismatch')
    return p


def plot_report(path, field, polygon, surface, entry, exit_, results, winners, stats, gp_csv=None, strain_percentile=99.5, strength_formulation='bishop_pt'):
    fig,(ax,ax2)=plt.subplots(2,1,figsize=(12,10),gridspec_kw={'height_ratios':[1.55,1]},constrained_layout=True)
    from matplotlib.patches import Patch
    colors={0:'#dcb47c',4:'#757f8a',5:'#79b7a6',8:'#a9a078'}
    mesh=field.cxy; mat=field.mid
    xleft=min(26.0,float(np.min(entry[:,0]))-4.0)
    xright=max(103.0,float(np.max(exit_[:,0]))+5.0)
    roi=(mesh[:,:,0].max(axis=1)>xleft)&(mesh[:,:,0].min(axis=1)<xright)&(mesh[:,:,1].max(axis=1)>26)&(mesh[:,:,1].min(axis=1)<73)
    for m in sorted(np.unique(mat[roi])):
        pp=mesh[roi&(mat==m)]
        ax.add_collection(PolyCollection(pp, facecolors=colors.get(int(m),'grey'),edgecolors='white',linewidths=.12,alpha=.6,zorder=1))
    if polygon.geom_type=='Polygon':
        xx,yy=np.asarray(polygon.exterior.coords).T
        ax.fill(xx,yy,color='#8b5cf6',alpha=.11,zorder=2)
    if gp_csv is not None:
        strain=pd.read_csv(gp_csv,usecols=['X','Y','Gamma_incremental'])
        gamma=strain['Gamma_incremental'].to_numpy(float)
        threshold=float(np.percentile(gamma,strain_percentile))
        hot=strain[(strain.Gamma_incremental>=threshold)&
                   (strain.X.between(25,105))&(strain.Y.between(25,75))]
        ax.scatter(hot.X,hot.Y,s=4,color='#e11d48',alpha=.55,zorder=6,
                   label=f'Top {100-strain_percentile:g}% incremental shear strain')
    ax.plot(surface[:,0],surface[:,1],color='#1e293b',lw=1.9,zorder=5)
    ax.plot(entry[:,0],entry[:,1],color='#dc2626',lw=4,zorder=7,label='C1 provisional entry zone')
    ax.plot(exit_[:,0],exit_[:,1],color='#2563eb',lw=4,zorder=7,label='C1 provisional exit zone')
    good=results[(results.status=='VALID_SCREENED')]
    # representative set, decimated to limit visual clutter
    for _,r in good.iloc[::max(1,len(good)//75)].iterrows():
        g=circle_geometry([r.entry_x,r.entry_y],[r.exit_x,r.exit_y],r.sagitta_m)
        _,p=arc_polyline(g,1.0)
        ax.plot(p[:,0],p[:,1],color='#64748b',alpha=.17,lw=.65,zorder=3)
    palette=['#e11d48','#16a34a','#2563eb','#a16207','#7c3aed']
    for j,r in enumerate(winners):
        g=circle_geometry([r['entry_x'],r['entry_y']],[r['exit_x'],r['exit_y']],r['sagitta_m'])
        _,pts=arc_polyline(g,.35)
        ax.plot(pts[:,0],pts[:,1],color=palette[j%len(palette)],lw=2.8 if j==0 else 1.55,zorder=8,
                label=(f"Rank {j+1}: FS={r['FS_refined']:.4f}" if j<3 else None))
    ax.set_xlim(xleft,xright);ax.set_ylim(25,75);ax.set_aspect('equal',adjustable='box')
    ax.set_xlabel('PLAXIS X (m)');ax.set_ylabel('PLAXIS Y (m)')
    ax.set_title('C2 | Circular candidates and refined top surfaces (provisional geometry)')
    ax.legend(fontsize=8,loc='upper right',ncol=2)
    ax.grid(alpha=.15)

    if len(good):
        z=ax2.scatter(good['sagitta_ratio'],good['FS_screened'],s=12,c=good['entry_x'],cmap='viridis',alpha=.6)
        fig.colorbar(z,ax=ax2,label='Entry X (m)')
    if len(winners):
        best=winners[0]
        ax2.scatter([best['sagitta_ratio']],[best['FS_refined']],marker='*',s=210, color='#e11d48',edgecolor='black',lw=.7,zorder=5,
                    label=f"Lowest refined FS = {best['FS_refined']:.4f}")
        ax2.legend(loc='upper right',fontsize=9)
    ax2.set_xlabel('Circular sagitta / chord length')
    ax2.set_ylabel('Conditional signed-shear FS (dimensionless)')
    ax2.grid(alpha=.2)
    ax2.set_title(f"{stats['n_total']} trial circles; {stats['n_screened_valid']} screened valid; "
                  f"{stats['n_refined_success']} FE-edge refinements completed")
    fig.suptitle(f'Stage C2 | {strength_formulation} stress-integral circular search | day {field.time:.3f}\n'
                 'NOT an established global critical surface; NOT PLAXIS phi-c reduction',fontsize=12)
    fig.savefig(path,dpi=165)
    plt.close(fig)


def choose_refinement_shortlist(valid, *, n_total, diverse_fraction=.25,
                                diverse_pool=4000,
                                entry_sep=1.0, exit_sep=1.5, ratio_sep=.015):
    """Keep top-ranked candidates AND sample different geometry basins.

    The first subset is *exactly* the best screened candidates. Additional
    candidates are chosen by minimum geometric separation from all selected.
    Diversity selection is deterministic and respects screening-FS ranking.
    """
    if n_total < 1 or not (0 <= diverse_fraction < 1):
        raise ValueError('Invalid refinement budget/diversity fraction')
    n_total=min(int(n_total),len(valid))
    n_top=max(1,int(round(n_total*(1.0-diverse_fraction))))
    chosen=valid.iloc[:n_top].copy()
    chosen['refinement_sampling']='top_screened'
    used=set(chosen['candidate_id'].astype(int))
    extra=[]
    existing=[(float(v.entry_x),float(v.exit_x),float(v.sagitta_ratio)) for v in chosen.itertuples()]
    for r in valid.iloc[n_top:max(n_top,int(diverse_pool))].itertuples():
        if len(chosen)+len(extra)>=n_total:break
        if int(r.candidate_id) in used:continue
        q=(float(r.entry_x),float(r.exit_x),float(r.sagitta_ratio))
        if any(abs(q[0]-v[0])<entry_sep and abs(q[1]-v[1])<exit_sep and
               abs(q[2]-v[2])<ratio_sep for v in existing):
            continue
        existing.append(q);used.add(int(r.candidate_id));extra.append(r.Index)
    if len(chosen)+len(extra)<n_total:
        for r in valid.itertuples():
            if len(chosen)+len(extra)>=n_total:break
            if int(r.candidate_id) in used:continue
            extra.append(r.Index);used.add(int(r.candidate_id))
    if extra:
        other=valid.loc[extra].copy()
        other['refinement_sampling']='diverse_or_fill'
        chosen=pd.concat([chosen,other],ignore_index=True)
    return chosen


def refine_geometry_with_b1(row, *, field, strength, formulation,
                            refine_step, allowed, max_negative_force_fraction=.15, source_label='grid'):
    """Independent exact FE-edge integration at polygonalized arc."""
    row=dict(row)
    g=circle_geometry([row['entry_x'],row['entry_y']],
                      [row['exit_x'],row['exit_y']],row['sagitta_m'])
    _,curve=arc_polyline(g,refine_step)
    df,b1_report=run_b1(field,curve)
    out_df,report=b2_compute(df,strength,formulation=formulation)
    fs=report['global_fos']
    if fs is None:raise ValueError('B2 rejected nonpositive signed driving shear')
    if set(df['Material_Index'].astype(int).unique())-set(allowed):
        raise ValueError('B1 crossed prohibited material')
    if np.any(df['Sigma_n_Bishop_compression'].to_numpy(float)<-1e-8):
        raise ValueError('B1 found tensile effective normal stress')
    if (report['integrated_mobilized_negative_magnitude_kN_per_m'] /
        max(report['integrated_mobilized_positive_kN_per_m']+
            report['integrated_mobilized_negative_magnitude_kN_per_m'],1e-12)) > max_negative_force_fraction:
        raise ValueError('Invalid shear-integral partition')
    row.update({'refinement_status':'OK','FS_refined':float(fs),
                'R_refined_kN_per_m':report['integrated_resistance_kN_per_m'],
                'T_refined_kN_per_m':report['integrated_mobilized_signed_kN_per_m'],
                'FE_subsegments_refined':b1_report['FE_subsegments'],
                'quadrature_points_refined':b1_report['gauss_integration_points'],
                'FS_refinement_difference':float(fs-row.get('FS_screened',fs)),
                'candidate_source':source_label})
    return (row,curve,df,out_df,report,b1_report)


def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    tests=self_tests()
    c1dir=Path(args.c1_dir)
    cfg,entries,exits,region,surface=load_c1(c1dir,Path(args.c1_config))
    allowed=list(map(int,cfg['allowed_material_indices']))
    strength=load_params(args.materials,allowed)
    field=BishopField(args.a2)
    if args.strength_formulation=='bishop_pt':
        if field.pt_nodes is None:
            raise ValueError('A2 has no BBM Pt: rerun project A2 before bishop_pt C2')
        for m in allowed:
            e_mask=field.mid==m
            if not np.any(e_mask) or not np.isfinite(field.pt_nodes[field.dof[e_mask]]).all():
                raise ValueError(f'BBM Pt is missing for allowed material {m}; cannot run bishop_pt')
    ratios=np.linspace(args.sagitta_ratio_min,args.sagitta_ratio_max,args.sagitta_steps)
    if args.sagitta_ratio_min<=0 or args.sagitta_ratio_max<=args.sagitta_ratio_min or args.sagitta_ratio_max>.49:
        raise ValueError('Sagitta ratio range must be 0 < min < max <=0.49')
    if args.step<=0 or args.refine_top<1:raise ValueError('step must be >0 and refine-top >=1')
    region_epsilon=region.buffer(args.geom_tolerance)
    use_numba=bool(getattr(args,'use_numba_geometry',False))
    if use_numba and not HAS_NUMBA:
        raise RuntimeError('Numba requested for C2 geometry but numba is not installed')
    if use_numba:
        # Trigger compilation once, outside the grid loop.
        _test_g=circle_geometry(entries[0],exits[0],np.linalg.norm(exits[0]-entries[0])*ratios[0])
        arc_screen(_test_g,args.step,surface)
        print('C2 numerical arc sampling: Numba backend enabled (geometry only).')
    all_rows=[];curve_cache={}; idx=0
    for ei,A in enumerate(entries):
      for xi,B in enumerate(exits):
        L=float(np.linalg.norm(B-A))
        for si,rat in enumerate(ratios):
          idx+=1
          s=L*rat
          g=circle_geometry(A,B,s)
          center,R,alpha,u,n=g
          if use_numba:
              pts,area,max_depth,not_monotone=arc_screen(g,args.step,surface)
              theta=np.linspace(-alpha,alpha,len(pts))
          else:
              theta,pts=arc_polyline(g,args.step)
              body_depth=np.interp(pts[:,0],surface[:,0],surface[:,1])-pts[:,1]
              area=float(np.sum((body_depth[1:]+body_depth[:-1])*0.5*np.diff(pts[:,0])))
              max_depth=float(body_depth.max())
              not_monotone=(np.min(np.diff(pts[:,0]))<1e-7 or
                            np.min(arc_tangent(g,theta)[:,0])<1e-7)
          row={
            'candidate_id':idx,'entry_i':ei,'exit_i':xi,'sagitta_i':si,
            'entry_x':float(A[0]),'entry_y':float(A[1]),'exit_x':float(B[0]),'exit_y':float(B[1]),
            'sagitta_ratio':float(rat),'sagitta_m':float(s),
            'circle_center_x':float(center[0]),'circle_center_y':float(center[1]),
            'circle_radius_m':float(R),'central_angle_rad':float(2*alpha),
            'arc_length_m':float(2*R*alpha), 'sliding_body_area_m2':area,
            'max_vertical_depth_m':max_depth,
            'status':'UNPROCESSED'
          }
          if not_monotone:
            row['status']='REJECT_NOT_X_MONOTONE'
          elif area<args.min_sliding_area:
            row['status']='REJECT_SMALL_SLIDING_MASS'
          elif max_depth<args.min_depth:
            row['status']='REJECT_TOO_SHALLOW'
          elif not region_epsilon.covers(LineString(pts)):
            row['status']='REJECT_OUTSIDE_C1_DOMAIN'
          else:
            try:
              integrals,status=integrate_arc(g,field,strength,theta,args.strength_formulation)
              row['status']=status
              if integrals is not None:
                  row.update(integrals)
                  if integrals['negative_force_fraction']>args.max_negative_force_fraction:
                      row['status']='REJECT_SHEAR_REVERSAL'
            except Exception as exc:
              row['status']='REJECT_FIELD_EVALUATION_ERROR'
              row['error']=str(exc)[:180]
          all_rows.append(row)
    result=pd.DataFrame(all_rows)
    result.to_csv(out/'C2_all_candidates.csv',index=False,float_format='%.11g')
    valid=result[result.status=='VALID_SCREENED'].sort_values('FS_screened').copy()
    if len(valid)==0:
        raise RuntimeError('No mechanically admissible trial circles; inspect C2_all_candidates.csv and relax provisional C1 zones only with physical justification')
    shortlist=choose_refinement_shortlist(
        valid,n_total=args.refine_top,
        diverse_fraction=float(getattr(args,'refine_diverse_fraction',0.25)),
        diverse_pool=int(getattr(args,'refine_diverse_pool',4000)),
        entry_sep=float(getattr(args,'refine_entry_separation_m',1.0)),
        exit_sep=float(getattr(args,'refine_exit_separation_m',1.5)),
        ratio_sep=float(getattr(args,'refine_ratio_separation',0.015)))
    refined=[]
    for j,(_,r) in enumerate(shortlist.iterrows(),start=1):
        row=r.to_dict()
        try:
            refined.append(refine_geometry_with_b1(
                row,field=field,strength=strength,
                formulation=args.strength_formulation,
                refine_step=args.refine_step,allowed=allowed,
                max_negative_force_fraction=args.max_negative_force_fraction))
        except Exception as exc:
            row.update({'refinement_status':'FAILED',
                        'refinement_error':str(exc)[:500], 'FS_refined':np.nan,
                        'candidate_source':'grid'})
            refined.append((row,None,None,None,None,None))
        if j%25==0 or j==len(shortlist):
            print(f'C2 FE-edge refinement {j}/{len(shortlist)} completed',flush=True)
    complete=sorted((x for x in refined if x[0]['refinement_status']=='OK'),key=lambda x:x[0]['FS_refined'])
    if not complete:
        raise RuntimeError('All shortlist FE-edge refinements failed: inspect C2_refined_shortlist.csv')
    best_grid=complete[0]
    bfgs_rows=[]; bfgs_best=[]
    if bool(getattr(args,'bfgs_enabled',False)):
        from .c2_bfgs import bfgs_refine,select_multistart_seeds
        starts=select_multistart_seeds(
            valid,int(getattr(args,'bfgs_starts',3)),
            entry_delta=float(getattr(args,'bfgs_entry_sep_m',1.0)),
            exit_delta=float(getattr(args,'bfgs_exit_sep_m',1.5)),
            ratio_delta=float(getattr(args,'bfgs_ratio_sep',.015)))
        for k,seed in enumerate(starts,1):
            try:
                b=bfgs_refine(seed,field=field,strength=strength,
                    region=region,ground=surface,config=cfg,
                    formulation=args.strength_formulation,step=args.step,
                    geom_tolerance=args.geom_tolerance,
                    min_sliding_area=args.min_sliding_area,
                    min_depth=args.min_depth,
                    max_negative_force_fraction=args.max_negative_force_fraction,
                    maxiter=int(getattr(args,'bfgs_maxiter',35)),
                    eps=tuple(getattr(args,'bfgs_fd_step_m',[.025,.025,.015])))
                newrow={
                    'candidate_id':f'bfgs_{k}_seed_{int(seed.candidate_id)}',
                    'entry_x':b['entry_x'],'entry_y':b['entry_y'],
                    'exit_x':b['exit_x'],'exit_y':b['exit_y'],
                    'sagitta_m':b['sagitta_m'],'sagitta_ratio':b['sagitta_ratio'],
                    'circle_center_x':b['center_x'],
                    'circle_center_y':b['center_y'],
                    'circle_radius_m':b['radius_m'],
                    'FS_screened':b['best_screened_FS']}
                cand=refine_geometry_with_b1(
                    newrow,field=field,strength=strength,
                    formulation=args.strength_formulation,
                    refine_step=args.refine_step,allowed=allowed,
                    max_negative_force_fraction=args.max_negative_force_fraction,
                    source_label='BFGS')
                row=cand[0]
                bfgs_best.append(cand)
                bfgs_rows.append({k:v for k,v in b.items() if k not in ('integrals','geometry')}
                                 | {'FS_refined':row['FS_refined'],'refinement_status':'OK'})
                print(f'BFGS start {k}/{len(starts)}: screened {b["best_screened_FS"]:.6f}; '
                      f'refined {row["FS_refined"]:.6f}; '
                      f'iterations {b["solver_iterations"]}',flush=True)
            except Exception as exc:
                bfgs_rows.append({'seed_id':int(seed.candidate_id),
                                  'refinement_status':'FAILED',
                                  'error':str(exc)[:500]})
                print(f'BFGS start {k}/{len(starts)} failed: {exc}',flush=True)
        pd.DataFrame(bfgs_rows).to_csv(out/'C2_BFGS_results.csv',index=False)
    complete_all=sorted(complete+bfgs_best,key=lambda x:x[0]['FS_refined'])
    best,curve,df,pointwise,report_b2,report_b1=complete_all[0]
    pd.DataFrame([x[0] for x in refined]).sort_values('FS_refined',na_position='last').to_csv(out/'C2_refined_shortlist.csv',index=False,float_format='%.11g')
    pd.DataFrame(curve,columns=['X','Y']).to_csv(out/'C2_best_circular_surface.csv',index=False,float_format='%.12g')
    pointwise.to_csv(out/'C2_best_surface_pointwise.csv',index=False,float_format='%.11g')
    status_counter=Counter(result.status)
    stats={'n_total':len(result),'n_screened_valid':len(valid),'n_refined_attempted':len(refined),
           'n_refined_success':len(complete),'n_BFGS_attempted':len(bfgs_rows),'n_BFGS_FE_verified':len(bfgs_best),
           'n_BFGS_solver_converged':sum(bool(q.get('solver_success',False)) for q in bfgs_rows),'status_counts':dict(status_counter)}
    # Independent hand algebra check with verified B2 material integrals.
    R_independent=sum(r['c_kPa']*next(q['length_m'] for q in report_b2['material_integrals'] if q['material_index']==i)
           + math.tan(math.radians(r['phi_deg']))*next(q['integrated_normal_kN_per_m'] + q['integrated_pt_kN_per_m'] for q in report_b2['material_integrals'] if q['material_index']==i)
           for i,r in strength.items() if any(q['material_index']==i for q in report_b2['material_integrals']))
    assert abs(R_independent-report_b2['integrated_resistance_kN_per_m'])<1e-6
    top_for_plot=[x[0] for x in complete_all[:4]]
    plot_report(out/'C2_circular_search_diagnostics.png',field,region,surface,entries,exits,result,top_for_plot,stats,args.gp_csv,float(cfg.get('high_strain_percentile',99.5)),args.strength_formulation)
    report={
      'stage':'C2 deterministic grid of circular slip surfaces',
      'method':'C1 grid / sagitta screening + diverse FE-edge shortlist; optional multistart BFGS (xc,yc,x1) with feasibility penalties, independent B1/B2 FE-edge verification',
      'strength_formulation':args.strength_formulation,
      'time':field.time,'A2_source':str(args.a2),'C1_source':str(c1dir),
      'C1_provisional':bool(cfg.get('provisional',True)),
      'parameters':{str(m):p for m,p in strength.items()},
      'geometry_sampling_backend':'numba' if use_numba else 'numpy',
      'search':{'entry_n':len(entries),'exit_n':len(exits),'sagitta_ratios':ratios.tolist(),
                'grid_size':len(result), 'screening_max_step_m':args.step,
                'shortlist_refine_n':args.refine_top, 'diverse_refine_fraction':float(getattr(args,'refine_diverse_fraction',0.25)), 'refinement_polyline_step_m':args.refine_step,
                'minimum_body_area_m2':args.min_sliding_area, 'minimum_max_vertical_depth_m':args.min_depth,
                'max_negative_force_fraction':args.max_negative_force_fraction,'geometric_tolerance_m':args.geom_tolerance},
      'counts':stats,'best_grid_FS_refined':best_grid[0]['FS_refined'], 'BFGS_runs':bfgs_rows, 'best_refined_surface':{k:(int(v) if isinstance(v,np.integer) else float(v) if isinstance(v,np.floating) else v) for k,v in best.items() if (not isinstance(v,float) or np.isfinite(v))},
      'best_B1_report':report_b1,'best_B2_report':report_b2,
      'self_tests':tests,
      'criticality':'Best of grid FE-edge shortlist + feasible FE-verified BFGS candidates; optimizer convergence is reported separately; NOT guaranteed global minimum',
      'limitations':[
         'Entry and exit positions and 22m depth band are provisional C1 choices; results change with search domain.',
         'Circular grid uses sampled sagitta; optional local BFGS with Kim-Lee shape variables (xc,yc,x1) is separately documented.',
         'The C2 screening uses true circular tangents but quadrature does not exactly subdivide at FEM material boundaries; shortlist refinement uses B1 mesh-edge subdivision on finely sampled chords.',
         'Screen-ranked and spatially diverse candidate shortlist is refined, not all candidates; minimum is not globally certified.',
         'BFGS uses explicit penalty rejection and a deterministic exit-ground intersection branch, which are implementation choices and not an exact reproduction of the 1997 paper.',
         'Do not interpret current post-failure day 100.41 as pre-failure stability.',
         'Bishop+Pt Mohr-Coulomb equivalent is not exact BBM yield safety, nor PLAXIS SRM safety factor.',
         'Signed shear numerator/denominator convention and negative-traction filtering require physical review.',
         'The L2-smoothed stress field and sample resolution must be independently checked for FS convergence.',
      ]}
    (out/'C2_report.json').write_text(json.dumps(report,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),encoding='utf-8')
    print(f'=== STAGE C2: CIRCULAR SEARCH | {args.strength_formulation} (CONDITIONAL FS) ===')
    print(f"Day {field.time:.6f}; candidate grid {len(result)}; initial geometry/mechanics pass {len(valid)}")
    print('Rejection counts:',dict(status_counter))
    print(f"B1 FE-edge refinements attempted {len(refined)}, successful {len(complete)}; BFGS {len(bfgs_best)}/{len(bfgs_rows)} FE-verified, {sum(bool(q.get('solver_success',False)) for q in bfgs_rows)}/{len(bfgs_rows)} solver-converged")
    print(f"BEST REFINED: candidate {best['candidate_id']} FS={best['FS_refined']:.6f} (screened={best['FS_screened']:.6f})")
    print(f"Entry=({best['entry_x']:.3f},{best['entry_y']:.3f}); Exit=({best['exit_x']:.3f},{best['exit_y']:.3f})")
    print(f"Sagitta={best['sagitta_m']:.3f} m; radius={best['circle_radius_m']:.3f} m")
    print(f"Signed T={best['T_refined_kN_per_m']:.4f} kN/m; resistance R={best['R_refined_kN_per_m']:.4f} kN/m")
    print('This is the best among tested finite-grid/shortlisted candidates, not a verified global critical surface.')
    print('Saved:',out.resolve())
    return report


