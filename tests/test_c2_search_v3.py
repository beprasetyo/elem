import numpy as np
import pandas as pd
import pytest
from kimlee.stages.c2 import circle_geometry, choose_refinement_shortlist
from kimlee.stages.c2_bfgs import reconstruct_center_entry,select_multistart_seeds


def test_center_entry_reconstruct_linear_slope():
    xs=np.linspace(10,110,501)
    ground=np.column_stack((xs,75-.40*(xs-10)))
    A=np.array([32.,float(np.interp(32,xs,ground[:,1]))])
    B=np.array([84.,float(np.interp(84,xs,ground[:,1]))])
    sag=np.linalg.norm(B-A)*.12
    old=circle_geometry(A,B,sag)
    g,a,b,s=reconstruct_center_entry([*old[0],A[0]],ground,[20,50],[70,100],B[0])
    np.testing.assert_allclose(a,A,atol=1e-8,rtol=0)
    np.testing.assert_allclose(b,B,atol=1e-8,rtol=0)
    np.testing.assert_allclose(g[0],old[0],atol=1e-8,rtol=0)
    np.testing.assert_allclose(s,sag,atol=1e-8,rtol=0)


def test_shortlist_preserves_top_and_adds_diversity():
    rows=[]
    for i in range(200):
        rows.append(dict(candidate_id=i+1,entry_x=40+i*.02,exit_x=80+i*.025,
                         sagitta_ratio=.1+i*.00004,FS_screened=.9+i*.00001))
    df=pd.DataFrame(rows).sort_values('FS_screened')
    shortlist=choose_refinement_shortlist(df,n_total=40,diverse_fraction=.25,diverse_pool=150,
                                            entry_sep=.3,exit_sep=.3,ratio_sep=.015)
    assert len(shortlist)==40
    assert shortlist.candidate_id.is_unique
    assert set(range(1,31)).issubset(set(shortlist.candidate_id))
    assert set(shortlist.refinement_sampling)=={'top_screened','diverse_or_fill'}


def test_seed_selection_is_deterministic():
    df=pd.DataFrame([dict(candidate_id=i+1,entry_x=40+i*.5,exit_x=80,
                          sagitta_ratio=.1,FS_screened=.92+i*.0001) for i in range(10)])
    starts=select_multistart_seeds(df,3,entry_delta=1,exit_delta=1.5,ratio_delta=.02)
    assert [int(x.candidate_id) for x in starts]==[1,4,7]
