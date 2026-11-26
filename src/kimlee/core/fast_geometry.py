"""Optional Numba acceleration for C2's numerical arc sampling ONLY.

No Shapely, TriFinder, sparse linear algebra or FEM constitutive code is JITed.
The NumPy reference path remains available and is the default.
"""
from __future__ import annotations
import math
import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    @njit(cache=True)
    def _arc_screen_numba(cx,cy,r,alpha,ux,uy,nx,ny,step,surfx,surfy):
        nseg = max(2, int(math.ceil(2.0*r*alpha/step)))
        pts = np.empty((nseg+1,2),dtype=np.float64)
        previous_x = -1e100
        not_monotone = False
        for i in range(nseg+1):
            angle=-alpha + (2.0*alpha*i)/nseg
            st=math.sin(angle); ct=math.cos(angle)
            x=cx+r*(st*ux+ct*nx)
            y=cy+r*(st*uy+ct*ny)
            pts[i,0]=x;pts[i,1]=y
            if i and x-previous_x<1e-7:
                not_monotone=True
            if ct*ux-st*nx < 1e-7:
                not_monotone=True
            previous_x=x
        depths=np.interp(pts[:,0],surfx,surfy)-pts[:,1]
        max_depth=np.max(depths)
        area=0.0
        for i in range(nseg):
            area+=0.5*(depths[i]+depths[i+1])*(pts[i+1,0]-pts[i,0])
        return pts,area,max_depth,not_monotone
else:
    _arc_screen_numba=None


def arc_screen(g, step, surface):
    """Return points, sliding area, max depth, nonmonotone flag.

    Arc parameterization matches Stage C2's circle_geometry; reference test
    against NumPy path must pass before acceleration is enabled.
    """
    if not HAS_NUMBA:
        raise RuntimeError('Numba geometry backend requested; install numba or disable the option')
    center,r,alpha,u,n=g
    return _arc_screen_numba(float(center[0]),float(center[1]),float(r),float(alpha),
                             float(u[0]),float(u[1]),float(n[0]),float(n[1]),
                             float(step),np.ascontiguousarray(surface[:,0],dtype=float),
                             np.ascontiguousarray(surface[:,1],dtype=float))
