"""Quadratic P2 triangle interpolation shared by FE recovery and field queries."""
from __future__ import annotations
import numpy as np


def p2_shape(bary):
    """6-node triangle: 3 vertices followed by 3 edge midpoint nodes."""
    l1,l2,l3=np.moveaxis(np.asarray(bary,dtype=float),-1,0)
    return np.stack([l1*(2*l1-1),l2*(2*l2-1),l3*(2*l3-1),
                     4*l1*l2,4*l2*l3,4*l3*l1],axis=-1)
