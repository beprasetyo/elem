import numpy as np
import pytest
from kimlee.core.fast_geometry import HAS_NUMBA, arc_screen
from kimlee.stages.c2 import circle_geometry, arc_polyline, arc_tangent

@pytest.mark.skipif(not HAS_NUMBA,reason='Numba is optional')
def test_numba_arc_sampling_matches_numpy():
    x=np.linspace(20,105,300)
    ys=70-0.40*(x-20)+0.0015*(x-20)**2
    surface=np.column_stack((x,ys))
    cases=[((35,70),(85,42),r) for r in (0.025,0.05,0.1,0.2,0.35)]
    cases += [((27,69),(90,36),r) for r in (0.035,0.09,0.3)]
    for A,B,r in cases:
        g=circle_geometry(A,B,np.linalg.norm(np.array(B)-np.array(A))*r)
        theta,p=arc_polyline(g,0.8)
        depth=np.interp(p[:,0],x,ys)-p[:,1]
        ref_area=float(np.sum((depth[1:]+depth[:-1])*0.5*np.diff(p[:,0])))
        ref_monotone=bool(np.min(np.diff(p[:,0]))<1e-7 or np.min(arc_tangent(g,theta)[:,0])<1e-7)
        q,area,maxdepth,monotone=arc_screen(g,0.8,surface)
        np.testing.assert_allclose(q,p,rtol=0,atol=1e-10)
        np.testing.assert_allclose(area,ref_area,rtol=0,atol=1e-9)
        np.testing.assert_allclose(maxdepth,depth.max(),rtol=0,atol=1e-10)
        assert bool(monotone)==ref_monotone
