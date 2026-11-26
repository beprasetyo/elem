"""Fast unit checks; no PLAXIS files required."""
import sys
import math
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import numpy as np
import pandas as pd
from kimlee.core.mechanics import equivalent_shear_strength, resolve_bishop_traction
from kimlee.core.shape import p2_shape
from kimlee.stages.b2 import compute, run_self_tests
from kimlee.stages.c2 import circle_geometry, arc_points, self_tests as circular_self_tests


class CoreMathTests(unittest.TestCase):
    def test_p2_partition(self):
        n=p2_shape(np.array([[1/3,1/3,1/3],[1,0,0],[.5,.5,0]]))
        np.testing.assert_allclose(n.sum(axis=1),[1,1,1],atol=1e-14)

    def test_compression_and_shear_sign(self):
        # Horizontal t=(1,0), upward n=(0,1); PLAXIS compression negative.
        sig=np.array([[-50.,-100.,20.]])
        n,tau=resolve_bishop_traction(sig,np.array([[1.,0.]]))
        self.assertAlmostEqual(n[0],100.)
        self.assertAlmostEqual(tau[0],20.)

    def test_bishop_pt_equals_fredlund_algebra(self):
        net=50.;seff=.8;s=40.;ks=.27;phi=32.
        bishop=net+seff*s;pt=ks*s
        lhs=equivalent_shear_strength(bishop,pt,0.,phi)
        rhs=(net+(seff+ks)*s)*math.tan(math.radians(phi))
        self.assertAlmostEqual(float(lhs),rhs,places=13)

    def test_pt_increases_equivalent_resistance_when_nonnegative(self):
        no_pt=equivalent_shear_strength(80.,0.,0.,32.)
        yes_pt=equivalent_shear_strength(80.,10.,0.,32.)
        self.assertGreater(float(yes_pt),float(no_pt))

    def test_b2_analytic_and_signed(self):
        run_self_tests()

    def test_b2_rejects_missing_pt(self):
        df=pd.DataFrame({'ds_weight_m':[1.,1.], 'Sigma_n_Bishop_compression':[80.,80.],
                         'Tau_tn_signed':[10.,10.],'Material_Index':[0,0]})
        with self.assertRaisesRegex(ValueError,'Pt_kPa'):
            compute(df,{0:{'name':'soil','c_kPa':0,'phi_deg':32}},formulation='bishop_pt')

    def test_circle_geometry(self):
        result=circular_self_tests()
        self.assertLess(result['circle_endpoints_max_error_m'],1e-8)

if __name__=='__main__':unittest.main()
