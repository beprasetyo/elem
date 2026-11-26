"""C3 — constrained, sequential noncircular slip search (Kim–Lee-inspired).

The paper defines N vertices and 2N-2 shape variables: all N x-values
and N-2 internal y-values; endpoint y's lie on the ground surface. It uses
kinematic restrictions and feasible-directions optimization with successive
midpoint insertion (4, 7, 13 vertices).

This implementation uses SLSQP with explicit nonlinear constraints PLUS a
feasibility-preserving coordinate search when its gradients are unreliable.
This is a *Kim–Lee-inspired implementation*, NOT the precise 1997 algorithm.
A2 Bishop-effective stress and BBM Pt are evaluated with the existing B2
strength convention, then the candidate is independently verified by B1
FE-edge splitting and B2 integration.
"""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from shapely.geometry import LineString, shape

from .b1 import BishopField, run_b1
from .b2 import compute as compute_b2
from .c2 import load_c1, load_params
from ..core.mechanics import equivalent_shear_strength, resolve_bishop_traction

GX,GW=np.polynomial.legendre.leggauss(3)


def make_vertices(curve, n):
    """Resample ordered x-monotone starting geometry to n vertices."""
    c=np.asarray(curve,dtype=float)
    if c.ndim!=2 or c.shape[1]!=2 or len(c)<2 or n<3:
        raise ValueError('Curve must be (M,2), N >=3')
    if np.any(np.diff(c[:,0])<=0):
        raise ValueError('Seed slip surface must have strictly increasing x')
    x=np.linspace(c[0,0],c[-1,0],n)
    return np.column_stack((x,np.interp(x,c[:,0],c[:,1])))


def to_variables(vertices):
    v=np.asarray(vertices,float)
    return np.r_[v[:,0],v[1:-1,1]]


def from_variables(q, n, ground):
    """N x-coordinates + (N-2) interior y-coordinates => 2N-2 variables."""
    q=np.asarray(q,float)
    if q.shape!=(2*n-2,):
        raise ValueError('Expected 2N-2 shape variables')
    x=q[:n]
    y=np.r_[np.interp(x[0],ground[:,0],ground[:,1]),q[n:],
                np.interp(x[-1],ground[:,0],ground[:,1])]
    return np.column_stack((x,y))


def turning_angles_degrees(verts):
    e=np.diff(np.asarray(verts,float),axis=0)
    a=e[:-1];b=e[1:]
    return np.degrees(np.abs(np.arctan2(a[:,0]*b[:,1]-a[:,1]*b[:,0],
                                    np.einsum('ij,ij->i',a,b))))


def dense_polyline(verts, max_step):
    pieces=[]
    for k,(a,b) in enumerate(zip(verts[:-1],verts[1:])):
        segments=max(1,int(np.ceil(np.linalg.norm(b-a)/max_step)))
        s=np.linspace(0.,1.,segments+1)
        piece=a[None,:]+s[:,None]*(b-a)[None,:]
        pieces.append(piece if k==0 else piece[1:])
    return np.concatenate(pieces,axis=0)


def geometry_checks(vertices, ground, region, config, *, max_step=.8):
    """Check physical admissibility BEFORE field evaluation."""
    p=np.asarray(vertices,float)
    entry_range=config['entry_x_range_m'];exit_range=config['exit_x_range_m']
    if not np.isfinite(p).all():return False,'NONFINITE',{}
    if not (entry_range[0]-1e-7<=p[0,0]<=entry_range[1]+1e-7):return False,'ENTRY_RANGE',{}
    if not (exit_range[0]-1e-7<=p[-1,0]<=exit_range[1]+1e-7):return False,'EXIT_RANGE',{}
    dx=np.diff(p[:,0]);min_gap=float(config.get('min_vertex_dx_m',.25))
    if np.min(dx)<min_gap-1e-8:return False,'X_ORDER',{}
    angles=turning_angles_degrees(p)
    max_turn=float(config.get('max_turning_angle_deg',50))
    if len(angles) and np.max(angles)>max_turn+1e-7:return False,'TURNING_ANGLE',{}
    d=dense_polyline(p,max_step)
    depth=np.interp(d[:,0],ground[:,0],ground[:,1])-d[:,1]
    area=float(np.trapezoid(depth,d[:,0]))
    maxdepth=float(np.max(depth));mindepth=float(np.min(depth))
    diagnostics={'area_m2':area,'max_depth_m':maxdepth,'min_depth_m':mindepth,
                 'max_turning_angle_deg':float(np.max(angles)) if len(angles) else 0.,
                 'min_dx_m':float(np.min(dx)),'vertex_count':len(p)}
    if mindepth< -1.e-5:return False,'ABOVE_GROUND',diagnostics
    if maxdepth>float(config.get('maximum_vertical_depth_below_ground_m',22.))+1e-5:
        return False,'DEPTH_EXCEEDS_C1',diagnostics
    if area<float(config.get('min_sliding_area_m2',20.)):
        return False,'BODY_AREA',diagnostics
    if maxdepth<float(config.get('min_depth_m',.5)):
        return False,'TOO_SHALLOW',diagnostics
    tolerance=float(config.get('geometry_tolerance_m',2e-6))
    if not region.buffer(tolerance).covers(LineString(d)):
        return False,'OUTSIDE_C1_DOMAIN',diagnostics
    return True,'OK',diagnostics


def integrate_polyline(vertices,field,strength,formulation,max_step=.8):
    """GL3 per sampled *straight* segment: accurate tangential orientation."""
    verts=dense_polyline(vertices,max_step)
    a=verts[:-1];b=verts[1:];vec=b-a
    lens=np.linalg.norm(vec,axis=1)
    if np.any(lens<=0):raise ValueError('Zero segment in noncircular slip')
    tangent=vec/lens[:,None]
    u=(GX+1)/2
    xy=(a[:,None,:]+u[None,:,None]*vec[:,None,:]).reshape(-1,2)
    w=(lens[:,None]*GW[None,:]*.5).reshape(-1)
    tangent=np.repeat(tangent,3,axis=0)
    ti,stresses=field.evaluate(xy)
    mats=field.mid[ti]
    if not set(mats.tolist()).issubset(set(strength)):
        raise ValueError('Unconfigured material on noncircular surface')
    sigma,tau=resolve_bishop_traction(stresses,tangent)
    if np.any(sigma<-1e-8):raise ValueError('Negative normal compression')
    if formulation=='bishop_pt':
        pt=field.evaluate_pt(xy)
    elif formulation=='bishop_only':
        pt=np.zeros_like(sigma)
    else:
        raise ValueError('Unsupported strength formulation')
    shear_strength=np.zeros_like(sigma)
    for mat,p in strength.items():
        mask=mats==mat
        shear_strength[mask]=equivalent_shear_strength(sigma[mask],pt[mask],p['c_kPa'],p['phi_deg'])
    R=float(w@shear_strength);T=float(w@tau)
    tpos=float(w@np.maximum(tau,0.));tneg=float(w@np.maximum(-tau,0.))
    neg=tneg/max(tpos+tneg,1e-12)
    if T<=1e-6:raise ValueError('Nonpositive signed driving shear')
    return {'FS_screened':R/T,'R_screened_kN_per_m':R,
            'T_screened_kN_per_m':T,'negative_force_fraction':neg,
            'integral_Pt_kN_per_m':float(w@pt),'n_quadrature':len(w),
            'curve_length_m':float(lens.sum())}


def evaluate(q,n,*,field,strength,region,ground,config,formulation):
    v=from_variables(q,n,ground)
    valid,why,diagnostics=geometry_checks(v,ground,region,config,
                                         max_step=float(config.get('screen_step_m',.8)))
    if not valid:return None,why,v
    try:
        integral=integrate_polyline(v,field,strength,formulation,
                                    float(config.get('screen_step_m',.8)))
        if integral['negative_force_fraction']>float(config.get('max_negative_force_fraction',.15)):
            return None,'SHEAR_REVERSAL',v
    except (ValueError,IndexError,FloatingPointError) as exc:
        return None,f'FIELD: {exc}',v
    return dict(integral,**diagnostics),'OK',v


def constraints_for_slsqp(q,n,ground,config):
    """Continuous necessary conditions; full-domain validity checked separately."""
    verts=from_variables(q,n,ground)
    x=verts[:,0]
    dx=np.diff(x)-float(config.get('min_vertex_dx_m',.25))
    angles=turning_angles_degrees(verts)
    turn=float(config.get('max_turning_angle_deg',50.))-angles
    d=np.interp(x[1:-1],ground[:,0],ground[:,1])-verts[1:-1,1]
    # Interior points must lie strictly within soil with small upper tolerance.
    return np.r_[dx,turn,d+.0001]


def create_bounds(vertices, ground, config, x_trust, y_trust):
    n=len(vertices)
    bounds=[]
    for i in range(n):
        x0=vertices[i,0]
        low,high=(config['entry_x_range_m'] if i==0 else
                  config['exit_x_range_m'] if i==n-1 else
                  (x0-x_trust,x0+x_trust))
        bounds.append((max(low,ground[0,0]),min(high,ground[-1,0])))
    for y0 in vertices[1:-1,1]:
        bounds.append((float(y0-y_trust),float(y0+y_trust)))
    return bounds


def feasible_coordinate_search(q, score, bounds, *, passes=3, step=.6):
    """Monotonic derivative-free fallback using feasible steps only.

    This is *not* the Bazaraa feasible-direction algorithm used by Kim–Lee.
    """
    q=np.array(q,float);fs=score(q);trace=[]
    if not np.isfinite(fs):return q,fs,trace
    delta=float(step)
    for k in range(int(passes)):
        changed=False
        for j in range(len(q)):
            for direction in (-1.,1.):
                trial=q.copy();trial[j]=min(bounds[j][1],max(bounds[j][0],q[j]+direction*delta))
                if abs(trial[j]-q[j])<1e-10:continue
                val=score(trial)
                if val<fs-1e-11:
                    q,fs=trial,val;changed=True
        trace.append(dict(pass_index=k+1,step_m=delta,FS=fs,improved=changed))
        delta*=.5
    return q,fs,trace


def verify_with_fe_edge(verts,field,strength,formulation,config):
    try:
        b1pts,b1report=run_b1(field,verts)
        _,b2report=compute_b2(b1pts,strength,formulation=formulation)
        fs=b2report.get('global_fos')
        if fs is None or not np.isfinite(fs):raise ValueError('Invalid global FS')
        neg=b2report['integrated_mobilized_negative_magnitude_kN_per_m']
        pos=b2report['integrated_mobilized_positive_kN_per_m']
        if set(b1pts['Material_Index'].to_numpy(dtype=int))-set(strength):
            raise ValueError('FE quadrature crossed material without strength parameters')
        if (b1pts['Sigma_n_Bishop_compression'].to_numpy(dtype=float)<-1e-8).any():
            raise ValueError('FE quadrature has tensile effective normal stress')
        if neg/max(pos+neg,1e-12)>float(config.get('max_negative_force_fraction',.15)):
            raise ValueError('FE-integrated shear reversal')
        if b1report['outside_domain_length_m']>1e-7:raise ValueError('Outside FEM domain')
        return float(fs),b1report,b2report
    except (ValueError,IndexError,FloatingPointError) as exc:
        return None,{'verification_error':str(exc)},None


def _interp_midpoints(vertices):
    v=np.asarray(vertices,float)
    m=.5*(v[:-1]+v[1:])
    return np.array([p for pair in zip(v[:-1],m) for p in pair]+[v[-1]])


def optimize_one_level(initial, *,field,strength,region,ground,config,formulation):
    n=len(initial);bounds=create_bounds(initial,ground,config,
          float(config.get('x_trust_m',2.0)),float(config.get('y_trust_m',2.0)))
    q0=to_variables(initial)
    cache={};count={'calls':0,'valid':0,'invalid':0}
    best={'fs':np.inf,'q':None,'info':None}
    def score(q):
        count['calls']+=1
        key=tuple(np.round(np.asarray(q,float),10))
        if key in cache:return cache[key]
        info,why,_=evaluate(q,n,field=field,strength=strength,region=region,
                      ground=ground,config=config,formulation=formulation)
        if info is None:
            count['invalid']+=1
            val=np.inf
        else:
            count['valid']+=1
            val=float(info['FS_screened'])
            if val<best['fs']:
                best.update(fs=val,q=np.array(q,float),info=info)
        cache[key]=val
        return val
    initialfs=score(q0)
    if not np.isfinite(initialfs):
        return {'status':'INITIAL_INVALID','error':'Initial N-vertex geometry rejected by constraints',
                'n_vertices':n},initial
    def objective(q):
        val=score(q)
        if np.isfinite(val):return val
        # Penalize invalid objective only for SLSQP iteration. Final candidates
        # MUST pass the independent real validity tests.
        q=np.asarray(q,float)
        return 2.5+.001*np.dot(q-q0,q-q0)
    def nonlinear(q):return constraints_for_slsqp(q,n,ground,config)
    solver=minimize(objective,q0,method='SLSQP',bounds=bounds,
           constraints=[{'type':'ineq','fun':nonlinear}],
           options={'maxiter':int(config.get('slsqp_maxiter',12)),
                    'ftol':float(config.get('ftol',1e-5)),'eps':float(config.get('fd_step_m',.025)),
                    'disp':False})
    if best['q'] is None:
        return {'status':'NO_VALID_OPTIMUM','n_vertices':n},initial
    seed=best['q']
    qpat,_,trace=feasible_coordinate_search(seed,score,bounds,
        passes=int(config.get('pattern_passes',3)),step=float(config.get('pattern_step_m',.5)))
    chosen=best['q']
    finalinfo,why,verts=evaluate(chosen,n,field=field,strength=strength,region=region,
                                 ground=ground,config=config,formulation=formulation)
    assert finalinfo is not None,why
    exact,b1report,b2report=verify_with_fe_edge(verts,field,strength,formulation,config)
    result={'status':'FE_VERIFIED' if exact is not None else 'FE_VERIFICATION_FAILED',
      'n_vertices':n,'n_variables':2*n-2,'initial_screened_FS':float(initialfs),
      'best_screened_FS':float(finalinfo['FS_screened']),
      'FE_verified_FS':exact,
      'solver_success':bool(solver.success),'solver_message':str(solver.message),
      'solver_nit':int(solver.nit),'objective_calls':count['calls'],
      'unique_feasible_evaluations':count['valid'],
      'unique_invalid_evaluations':count['invalid'],
      'coord_search_trace':trace,'diagnostics':finalinfo,
      'B1_report':b1report,'B2_report':b2report,
      'algorithm':'SLSQP finite-difference + feasible coordinate polish (not original Kim-Lee feasible directions)'}
    return result,verts


def plot_results(path, field, region, ground, circular, levels, curves, formulation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,6.5),constrained_layout=True)
    if region.geom_type=='Polygon':
        px,py=np.asarray(region.exterior.coords).T
        ax.fill(px,py,color='#c084fc',alpha=.17,label='C1 allowed domain')
    ax.plot(ground[:,0],ground[:,1],color='#334155',lw=2,label='Ground')
    ax.plot(circular[:,0],circular[:,1],ls='--',color='#64748b',lw=1.7,
            label='C2 circular reference')
    palette=['#0f766e','#ea580c','#b91c1c']
    for j,(rec,v) in enumerate(zip(levels,curves)):
        if v is None:continue
        fs=rec.get('FE_verified_FS')
        label=f'C3 {len(v)} vertices: FS={fs:.4f}' if fs is not None else f'C3 {len(v)} vertices (unverified)'
        ax.plot(v[:,0],v[:,1],lw=2.2,color=palette[j%len(palette)],marker='o',ms=3,
                label=label)
    xa=min(circular[:,0].min(),float(ground[:,0].min()));xb=max(circular[:,0].max(),float(ground[:,0].max()))
    ax.set_xlim(max(15,xa),min(xb,104));ax.set_ylim(25,75)
    ax.set_aspect('equal',adjustable='box');ax.set_xlabel('PLAXIS X (m)');ax.set_ylabel('PLAXIS Y (m)')
    ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower left')
    ax.set_title(f'Stage C3 — Noncircular candidate search (conditional {formulation} FS)\n'
        'Kim–Lee-inspired constraints; optimizer not a replication of original feasible-direction solver')
    fig.savefig(path,dpi=160);plt.close(fig)


def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    c2dir=Path(args.c2_dir)
    with (c2dir/'C2_report.json').open(encoding='utf-8') as f:
        c2report=json.load(f)
    if c2report.get('strength_formulation')!=args.strength_formulation:
        raise ValueError('C2 and C3 strength modes mismatch: rerun C2')
    C2_fs=float(c2report['best_refined_surface']['FS_refined'])
    base=pd.read_csv(c2dir/'C2_best_circular_surface.csv')[['X','Y']].to_numpy(float)
    c1config,entries,exits,region,ground=load_c1(Path(args.c1_dir),Path(args.c1_config))
    strength=load_params(args.materials,c1config['allowed_material_indices'])
    for mat, params in strength.items():
        prior=c2report.get('parameters',{}).get(str(mat),{})
        if not prior or any(not np.isclose(float(params[k]),float(prior.get(k,np.nan)),atol=1e-10,rtol=0)
                            for k in ('c_kPa','phi_deg')):
            raise ValueError(f'C2/C3 material strength mismatch for index {mat}. Rerun C2 with current materials.')
    field=BishopField(args.a2)
    if abs(field.time-float(c2report['time']))>1e-6:
        raise ValueError('C2 report day differs from current A2 snapshot. Rerun C2.')
    if args.strength_formulation=='bishop_pt' and field.pt_nodes is None:
        raise ValueError('C3 bishop_pt requires A2 with recovered Pt')
    setting=dict(vars(args))
    for key in ('out','a2','c1_dir','c1_config','c2_dir','materials','strength_formulation'):setting.pop(key,None)
    config=dict(c1config,**setting)
    if not args.vertex_levels or args.vertex_levels[0]!=4 or any(n<4 for n in args.vertex_levels):
        raise ValueError('C3 must start with 4 vertices (Kim-Lee step 1)')
    if any(n2!=2*n1-1 for n1,n2 in zip(args.vertex_levels[:-1],args.vertex_levels[1:])):
        raise ValueError('C3 vertex levels must grow by midpoint insertion N_next=2*N-1')
    levels=[];curves=[]; current=make_vertices(base,4)
    for n in args.vertex_levels:
        if len(current)!=n:
            if n==2*len(current)-1:current=_interp_midpoints(current)
            else:current=make_vertices(current,n)
        result,optimized=optimize_one_level(current,field=field,strength=strength,
                    region=region,ground=ground,config=config,
                    formulation=args.strength_formulation)
        levels.append(result)
        if result['status']!='FE_VERIFIED':
            curves.append(None)
            print(f'C3 N={n}: {result["status"]}. Stop progression; inspect diagnostics.',flush=True)
            break
        curves.append(optimized)
        current=optimized
        pd.DataFrame(optimized,columns=['X','Y']).to_csv(out/f'C3_N{n}_surface.csv',index=False)
        print(f'C3 N={n}: FS_screened={result["best_screened_FS"]:.6f} '
              f'FS_FE={result["FE_verified_FS"]:.6f} '
              f'SLSQP_success={result["solver_success"]}',flush=True)
    feasible=[(i,r) for i,r in enumerate(levels) if r.get('FE_verified_FS') is not None]
    bestidx=min(feasible,key=lambda z:z[1]['FE_verified_FS'])[0] if feasible else None
    # Never silently present a worse-than-circular candidate as an improved critical surface.
    output={'stage':'C3 noncircular search','time_day':float(field.time),
      'strength_formulation':args.strength_formulation,
      'source_C2_refined_FS':C2_fs,'source_C2_dir':str(c2dir),
      'reference_is_provisional':True,'optimization_method':
         'SLSQP+feasibility-preserving pattern search, NOT original Kim-Lee feasible-direction implementation',
      'levels':levels,
      'best_C3_FE_verified_FS':float(levels[bestidx]['FE_verified_FS']) if bestidx is not None else None,
      'best_C3_level':int(levels[bestidx]['n_vertices']) if bestidx is not None else None,
      'C3_improves_C2':bool(levels[bestidx]['FE_verified_FS']<C2_fs) if bestidx is not None else False,
      'limitations':['No proof of global optimum','C1 provisional zones',
         'Stage C2 BFGS did not converge formally','Geometry and sampled stress fields can be nonsmooth',
         'FS is an equivalent Bishop+Pt stress-integral proxy, not exact BBM yield FS',
         'SLSQP/coordinate search substitute for Kim-Lee feasible-direction method']}
    (out/'C3_report.json').write_text(json.dumps(output,indent=2,default=float),encoding='utf-8')
    plot_results(out/'C3_diagnostics.png',field,region,ground,base,levels,curves,args.strength_formulation)
    print('C3 completed (candidate verification only). Best FE FS:',output['best_C3_FE_verified_FS'])
    return out/'C3_report.json'
