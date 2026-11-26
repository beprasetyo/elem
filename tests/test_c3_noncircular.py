import numpy as np
from shapely.geometry import box
from kimlee.stages.c3 import (make_vertices,to_variables,from_variables,
  _interp_midpoints,turning_angles_degrees,geometry_checks,
  constraints_for_slsqp,feasible_coordinate_search,dense_polyline)


def test_exact_variable_count_and_endpoints():
    ground=np.array([[0.,50.],[100.,50.]])
    v=np.array([[10.,50.],[20.,45.],[35.,43.],[50.,50.]])
    q=to_variables(v)
    assert len(q)==2*len(v)-2
    np.testing.assert_allclose(from_variables(q,4,ground),v)


def test_midpoint_growth_four_to_seven_to_thirteen():
    v=np.column_stack([np.arange(4.),np.array([5.,3.,3.,5.])])
    w=_interp_midpoints(v)
    z=_interp_midpoints(w)
    assert len(w)==7 and len(z)==13
    np.testing.assert_allclose(w[::2],v)
    np.testing.assert_allclose(z[::2],w)


def test_geometry_constraints_reject_sharp_kink():
    ground=np.column_stack([np.linspace(0,100,5),np.full(5,50.)])
    region=box(0,20,100,50)
    cfg={'entry_x_range_m':[5,15],'exit_x_range_m':[75,85],
         'min_vertex_dx_m':.25,'max_turning_angle_deg':40,
         'min_sliding_area_m2':20.,'min_depth_m':.5,
         'maximum_vertical_depth_below_ground_m':25.,'geometry_tolerance_m':2e-6}
    v=np.array([[10.,50.],[30.,45.],[60.,45.],[80.,50.]])
    good,reason,info=geometry_checks(v,ground,region,cfg)
    assert good and reason=='OK'
    bad=v.copy();bad[2,1]=25.
    good,reason,info=geometry_checks(bad,ground,region,cfg)
    assert not good and reason in {'TURNING_ANGLE','DEPTH_EXCEEDS_C1'}


def test_feasible_coordinate_search_declines_invalid():
    bounds=[(-2.,2.),(-2.,2.)]
    def f(q):
        if q[0]>0:return np.inf
        return (q[0]+1)**2+(q[1]-.5)**2
    q,val,trace=feasible_coordinate_search([0.,0.],f,bounds,passes=4,step=.5)
    assert np.isfinite(val) and val<1.25 and q[0]<=0
    assert all(np.isfinite(t['FS']) for t in trace)


def test_resample_and_turning():
    base=np.column_stack((np.linspace(0,10,101),48-2*np.sin(np.linspace(0,np.pi,101))))
    v=make_vertices(base,7)
    assert len(v)==7
    dense=dense_polyline(v,.3)
    assert dense[0,0]==0 and dense[-1,0]==10
    assert np.max(turning_angles_degrees(np.column_stack((np.arange(4),np.zeros(4)))))<1e-10


def test_continuous_constraint_vector_length():
    ground=np.column_stack([np.linspace(0,100,5),np.full(5,50.)])
    v=np.array([[10.,50.],[30.,45.],[60.,45.],[80.,50.]])
    config={'min_vertex_dx_m':.25,'max_turning_angle_deg':50}
    cons=constraints_for_slsqp(to_variables(v),4,ground,config)
    assert len(cons)==3+2+2 and np.min(cons)>=0
