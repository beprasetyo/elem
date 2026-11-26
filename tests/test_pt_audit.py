import numpy as np
import pandas as pd
from kimlee.stages.b1 import BishopField
from kimlee.validation.pt_recovery import _compare_at_points, audit_surface, gp_audit


def artificial_field():
    field=object.__new__(BishopField)
    field.cxy=np.array([[[0.,0.],[1.,0.],[0.,1.]]])
    field.dof=np.arange(6).reshape(1,6)
    field.pt_nodes=np.array([0.,0.,0.,-1.,-1.,-1.])
    field.mid=np.array([5])
    field.finder=lambda xs,ys: np.zeros(len(np.atleast_1d(xs)),dtype=int)
    return field


def test_pt_alternatives_are_not_equivalent():
    field=artificial_field()
    xy=np.array([[.2,.2],[.3,.2],[.2,.3]])
    values,_=_compare_at_points(field,xy)
    assert np.all(values['raw_P2']<0)
    assert np.all(values['GP_clipped_P2']==0)
    assert np.all(values['positive_P1_corner']==0)
    assert np.all(values['nodal_clipped_P2']==0)


def test_audit_surface_fixed_fs_arithmetic(tmp_path):
    field=artificial_field()
    path=tmp_path/'surface.csv'
    xy=np.array([[.2,.2],[.3,.2],[.2,.3]])
    pd.DataFrame({
        'X':xy[:,0], 'Y':xy[:,1], 'ds_weight_m':[1.,2.,3.],
        'Sigma_n_Bishop_compression':[10.,10.,10.],
        'Tau_tn_signed':[5.,5.,5.],
        'Material_Index':[5,5,5],
        'Pt_kPa':[0.,0.,0.],
    }).to_csv(path,index=False)
    report=audit_surface(field,path,{5:{'c_kPa':0.,'phi_deg':45.}})
    assert abs(report['basis_FSBishop_only']-2.)<1e-12
    assert abs(report['Pt_evaluation_variants']['GP_clipped_P2']['FS_conditional']-2.)<1e-12
    assert report['Pt_evaluation_variants']['raw_P2']['FS_conditional']<2.


def test_gauss_point_negativity_count():
    field=artificial_field()
    data={
        'gauss_pt_raw':np.array([[0.,2.]]),
        'gauss_pt_recovered':np.array([[-.5,2.2]]),
        'material_index':np.array([5]),
        'corner_xy':field.cxy,
        'gauss_weights':np.array([.5,.5]),
    }
    summary=gp_audit(data,field)
    assert len(summary)==1
    assert summary[0]['raw_GP_negative_count']==0
    assert summary[0]['recovered_GP_negative_count']==1
    assert summary[0]['recovered_GP_negative_fraction']==.5
