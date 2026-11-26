"""Stress traction and adopted BBM-informed equivalent-MC strength convention.

Stress components are PLAXIS Bishop-effective and compression-negative.
The oriented surface tangent points downslope toward increasing X.
The upward local normal is (-ty,tx) (toward sliding soil above the curve).
NO pore-pressure or Seff term is added a second time.
"""
from __future__ import annotations
import numpy as np


def resolve_bishop_traction(stress, tangent):
    """Return compression-positive normal traction and signed shear traction.

    Parameters
    ----------
    stress : (N,3) [sigma_xx^B, sigma_yy^B, sigma_xy^B]
    tangent : (N,2) unit tangent t
    """
    stress=np.asarray(stress,float)
    t=np.asarray(tangent,float)
    if stress.ndim!=2 or stress.shape[1]!=3 or t.shape!=(len(stress),2):
        raise ValueError('Expected stress (N,3) and tangent (N,2)')
    n=np.column_stack((-t[:,1],t[:,0]))
    sxx,syy,sxy=stress.T
    tx,ty=t.T;nx,ny=n.T
    sigma_n=-(nx*nx*sxx+2*nx*ny*sxy+ny*ny*syy)
    tau=tx*nx*sxx+(tx*ny+ty*nx)*sxy+ty*ny*syy
    return sigma_n,tau


def equivalent_shear_strength(sigma_n_bishop, pt, c_kpa, phi_deg):
    """Equivalent MC post-processing, tau_f=c'+(sigma_n^B+Pt)*tan(phi').

    Pt=0 represents the bishop_only diagnostic mode. This is not the exact
    yield function of the Barcelona Basic Model.
    """
    return np.asarray(c_kpa,float)+(np.asarray(sigma_n_bishop,float)+np.asarray(pt,float))*np.tan(np.deg2rad(phi_deg))
