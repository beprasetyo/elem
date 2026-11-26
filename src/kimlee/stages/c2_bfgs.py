"""Optional Kim--Lee Eq. (11)-style circular BFGS refinement.

Physical shape variables are (xc, yc, x1). A continuous circle passes through
A=(x1, ground(x1)); exit is found from the first appropriate intersection of
the lower circle branch with ground inside C1's exit interval. We use multistart
scipy BFGS with finite differences (cf. Kim--Lee Eq. (12)). Geometry/material
feasibility is enforced with rejection penalties. THIS is not a faithful
replica of the authors' particular initialization/line search details.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import brentq, minimize
from shapely.geometry import LineString


def reconstruct_center_entry(q, surface, entry_range, exit_range, reference_exit=None):
    """Convert Kim-Lee (xc,yc,x1) variables to an admissible circular arc.

    Returns (geometry, entry, exit, sagitta) or raises ValueError.
    The circular surface is the *lower* branch and daylights at ground.
    """
    from .c2 import circle_geometry
    xc,yc,x1=map(float,q)
    if not (np.isfinite(q).all() and entry_range[0] <= x1 <= entry_range[1]):
        raise ValueError('Entry outside allowed interval')
    ground_x,ground_y=surface[:,0],surface[:,1]
    y1=float(np.interp(x1,ground_x,ground_y))
    A=np.array([x1,y1]);C=np.array([xc,yc]);R=float(np.hypot(x1-xc,y1-yc))
    if not (R>0.5 and yc>y1+1e-6):
        raise ValueError('Circle center must lie above starting point')
    lo=max(float(exit_range[0]),x1+1e-4,xc-R+1e-8)
    hi=min(float(exit_range[1]),xc+R-1e-8)
    if lo>=hi:raise ValueError('Circle cannot intersect exit search interval')
    xs=np.unique(np.r_[lo,ground_x[(ground_x>lo)&(ground_x<hi)],hi])
    f=lambda x: float(np.interp(x,ground_x,ground_y)-yc+np.sqrt(max(0.,R*R-(x-xc)**2)))
    vals=np.array([f(v) for v in xs])
    roots=[]
    for i in range(len(xs)-1):
        a,b=xs[i],xs[i+1]; fa,fb=vals[i],vals[i+1]
        if abs(fa)<1e-7:roots.append(float(a))
        if fa*fb<0:roots.append(float(brentq(f,a,b,xtol=1e-11)))
    if abs(vals[-1])<1e-7:roots.append(float(xs[-1]))
    roots=sorted(set(round(x,10) for x in roots))
    if not roots:raise ValueError('No daylight in exit interval')
    x2=min(roots,key=lambda x:abs(x-reference_exit)) if reference_exit is not None else roots[0]
    B=np.array([x2,float(np.interp(x2,ground_x,ground_y))]);chord=B-A
    L=float(np.linalg.norm(chord))
    if L<1:raise ValueError('Circle too small')
    u=chord/L; down=np.array([u[1],-u[0]])
    s=float(R+np.dot(C-(A+B)/2,down))
    if s<=0 or s/L>=0.49:
        raise ValueError('Undesired circular arc branch / sagitta')
    g=circle_geometry(A,B,s)
    if np.linalg.norm(g[0]-C)>1e-5 or abs(g[1]-R)>1e-5:
        raise ValueError('Circle center mismatch')
    return g,A,B,s


def select_multistart_seeds(valid, n, entry_delta=1.0, exit_delta=1.5, ratio_delta=.015):
    """Select low-FS but geometrically distinct seeds."""
    seeds=[]
    for _,row in valid.iterrows():
        if all(abs(row.entry_x-x.entry_x)>entry_delta or
               abs(row.exit_x-x.exit_x)>exit_delta or
               abs(row.sagitta_ratio-x.sagitta_ratio)>ratio_delta for x in seeds):
            seeds.append(row)
            if len(seeds)>=n:break
    return seeds


def bfgs_refine(seed, *, field, strength, region, ground, config,
                formulation, step=0.8, geom_tolerance=2e-6,
                min_sliding_area=20., min_depth=.5, max_negative_force_fraction=.15,
                maxiter=35, eps=(.025,.025,.015), gtol=2e-4):
    """Minimize conditional screened FS with Kim--Lee circular shape variables.

    The objectives outside the admissible domain are penalized; therefore
    smooth unconstrained BFGS is only an approximation to Kim & Lee's search.
    """
    from .c2 import arc_polyline, arc_tangent, integrate_arc
    g0,A0,B0,s0=None,None,None,None
    from .c2 import circle_geometry
    g0=circle_geometry([seed.entry_x,seed.entry_y],[seed.exit_x,seed.exit_y],seed.sagitta_m)
    q0=np.array([g0[0][0],g0[0][1],seed.entry_x],float)
    entry_range=config['entry_x_range_m'];exit_range=config['exit_x_range_m']
    buf=region.buffer(geom_tolerance)
    best={'fs':np.inf,'q':None,'g':None,'A':None,'B':None,'s':None,'integrals':None}
    tally={'total':0,'feasible':0,'invalid':0}
    cache={}

    def feasible(q):
        g,A,B,s=reconstruct_center_entry(q,ground,entry_range,exit_range,float(seed.exit_x))
        theta,points=arc_polyline(g,step)
        if np.min(np.diff(points[:,0]))<1e-7 or np.min(arc_tangent(g,theta)[:,0])<1e-7:
            raise ValueError('Nonmonotone circle')
        depth=np.interp(points[:,0],ground[:,0],ground[:,1])-points[:,1]
        area=float(np.trapezoid(depth,points[:,0]))
        if area<min_sliding_area or depth.max()<min_depth:
            raise ValueError('Body too small/shallow')
        if not buf.covers(LineString(points)):
            raise ValueError('Outside C1 domain')
        integrals,status=integrate_arc(g,field,strength,theta,formulation)
        if status!='VALID_SCREENED' or integrals is None:
            raise ValueError('Stress-integration rejected candidate: '+status)
        if integrals['negative_force_fraction']>max_negative_force_fraction:
            raise ValueError('Shear reversal')
        return float(integrals['FS_screened']),g,A,B,s,integrals

    def objective(q):
        tally['total']+=1
        key=tuple(np.round(q,9))
        if key in cache:return cache[key]
        try:
            fs,g,A,B,s,integrals=feasible(q)
            tally['feasible']+=1
            if fs<best['fs']:
                best.update(fs=fs,q=np.array(q,float),g=g,A=A,B=B,s=s,integrals=integrals)
            value=fs
        except (ValueError,KeyError,IndexError,FloatingPointError):
            tally['invalid']+=1
            # A finite barrier prevents crashes in BFGS's line search. The
            # penalty is not a perfect smooth constrained optimization method.
            value=1.8+0.001*np.dot(np.asarray(q)-q0,np.asarray(q)-q0)
        cache[key]=float(value)
        return float(value)

    init_fs=objective(q0)
    solution=minimize(objective,q0,method='BFGS',options={
        'maxiter':int(maxiter),'gtol':float(gtol),
        'eps':np.asarray(eps,float),'disp':False})
    if best['q'] is None:
        raise RuntimeError('BFGS start has no feasible candidate')
    g=best['g'];A=best['A'];B=best['B']
    return {
      'seed_id':int(seed.candidate_id),
      'initial_screened_FS':float(init_fs),
      'best_screened_FS':float(best['fs']),
      'solver_success':bool(solution.success),'solver_message':str(solution.message),
      'solver_iterations':int(solution.nit),
      'objective_calls':tally['total'],'feasible_calls':tally['feasible'],
      'invalid_calls':tally['invalid'],
      'center_x':float(g[0][0]),'center_y':float(g[0][1]),
      'entry_x':float(A[0]),'entry_y':float(A[1]),
      'exit_x':float(B[0]),'exit_y':float(B[1]),
      'sagitta_m':float(best['s']),'sagitta_ratio':float(best['s']/np.linalg.norm(B-A)),
      'radius_m':float(g[1]),
      'optimization_variables':best['q'].tolist(),
      'integrals':best['integrals'],
      'geometry':g
    }
