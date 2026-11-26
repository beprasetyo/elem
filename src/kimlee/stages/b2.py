"""Stage B2: conditional stress-integral safety factor for a prescribed curve.

Modes: bishop_only (regression) or bishop_pt (main BBM-informed equivalent MC).
No double counting of Seff/suction, no claim of exact BBM yielding.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ..core.mechanics import equivalent_shear_strength

REQUIRED = [
    'Path_Segment', 'Subsegment', 'Element_ID', 'Material_Index',
    'X', 'Y', 'Arclength_m', 'ds_weight_m',
    'tangent_x','tangent_y','normal_x','normal_y',
    'Sigxx_E','Sigyy_E','Sigxy',
    'Sigma_n_Bishop_compression','Tau_tn_signed',
]
STRESS_MEASURE = 'Bishop_effective_PLAXIS'


def read_b1(path: Path, report_path: Path | None):
    if not path.is_file():
        raise FileNotFoundError(f'B1 quadrature CSV does not exist: {path}')
    df = pd.read_csv(path)
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f'B1 CSV missing columns {missing}; use Stage B1 Bishop output')
    if len(df) < 3:
        raise ValueError('At least three quadrature records required')
    numbers = df[REQUIRED].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(numbers.to_numpy(dtype=float)).all():
        raise ValueError('B1 quadrature CSV contains nonfinite values')
    df = df.copy().sort_values('Arclength_m', kind='stable').reset_index(drop=True)
    w = df.ds_weight_m.to_numpy(dtype=float)
    if np.any(w <= 0):
        raise ValueError('Quadrature weights must be positive')
    s = df.Arclength_m.to_numpy(dtype=float)
    if np.any(np.diff(s) <= 0):
        raise ValueError('Quadrature arclength must increase strictly')
    ids = df.Material_Index.to_numpy(dtype=float)
    if not np.all(ids == np.round(ids)):
        raise ValueError('Material_Index must consist of integer IDs')
    if not np.all(df.Element_ID.to_numpy(dtype=float) == np.round(df.Element_ID.to_numpy(dtype=float))):
        raise ValueError('Element_ID must be integer-valued')
    # Validate oriented tangent/normal from B1; no automatic sign flipping.
    t = df[['tangent_x','tangent_y']].to_numpy(dtype=float)
    n = df[['normal_x','normal_y']].to_numpy(dtype=float)
    err_ortho = float(np.max(np.abs(np.einsum('ij,ij->i', t, n))))
    err_tunit = float(np.max(np.abs(np.linalg.norm(t,axis=1)-1)))
    err_nunit = float(np.max(np.abs(np.linalg.norm(n,axis=1)-1)))
    err_normal = float(np.max(np.abs(n - np.column_stack((-t[:,1],t[:,0])))))
    if max(err_ortho,err_tunit,err_nunit,err_normal) > 1e-7:
        raise ValueError('Invalid B1 tangent/normal system: values must be orthonormal, n=(-ty,tx)')
    if np.any(t[:,0] <= 0) or np.any(n[:,1] <= 0):
        raise ValueError('Expected left-to-right curve (tangent_x>0) and upward normal (normal_y>0). '
                         'Do not reverse or flip surface sign automatically.')
    # Independent Cartesian tensor transform check at every integration point.
    sxx = df.Sigxx_E.to_numpy(dtype=float)
    syy = df.Sigyy_E.to_numpy(dtype=float)
    sxy = df.Sigxy.to_numpy(dtype=float)
    tx,ty = t.T
    nx,ny = n.T
    normal = -(nx*nx*sxx + 2*nx*ny*sxy + ny*ny*syy)
    tau = tx*nx*sxx + (tx*ny+ty*nx)*sxy + ty*ny*syy
    transform_error = float(max(np.max(np.abs(normal-df.Sigma_n_Bishop_compression.to_numpy())),
                                np.max(np.abs(tau-df.Tau_tn_signed.to_numpy()))))
    if transform_error > 1e-6:
        raise ValueError(f'B1 stress transformation inconsistent (max error={transform_error:.4g})')
    if np.any(normal < -1e-8):
        raise ValueError('Tensile Bishop normal stress detected. Explicit tensile/cap rule required; '
                         'B2 does not silently clamp compression to zero.')

    report = None
    if report_path is None:
        raise ValueError('B1 validation report is required to verify Bishop effective-stress metadata. '
                         'Keep B1_Bishop_validation_report.json next to the quadrature CSV, '
                         'or supply --b1-report explicitly.')
    if report_path is not None:
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if report.get('stress_measure') != STRESS_MEASURE:
            raise ValueError('B1 report does not certify Bishop effective stress')
        if report.get('stress_components') != ['Sigxx_E','Sigyy_E','Sigxy']:
            raise ValueError('B1 report stress columns mismatch')
        if int(report.get('gauss_integration_points', -1)) != len(df):
            raise ValueError('B1 report quadrature count inconsistent')
        if abs(float(report['total_length_m']) - np.sum(w)) > 1e-5:
            raise ValueError('B1 report length inconsistent with quadrature weights')
        if sorted(map(int, report['unique_material_indices'])) != sorted(map(int,np.unique(ids))):
            raise ValueError('B1 report material index mismatch')
    return df, report, transform_error


def read_strength(path: Path | None, materials):
    if path is None:
        return None
    user = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(user, dict):
        raise ValueError('Parameter JSON must be an object keyed by material index')
    params = {}
    for m in sorted(materials):
        key = str(int(m))
        if key not in user:
            raise ValueError(f'Missing material {m} in strength JSON. Never assign a fallback strength.')
        x = user[key]
        if not isinstance(x,dict) or x.get('c_kPa') is None or x.get('phi_deg') is None:
            raise ValueError(f'Material {m} requires explicit c_kPa and phi_deg (non-null)')
        try:
            c=float(x['c_kPa'])
            phi=float(x['phi_deg'])
        except (ValueError,TypeError) as e:
            raise ValueError(f'Invalid strength parameters for material {m}: {e}')
        if not (np.isfinite(c) and np.isfinite(phi) and c >= 0 and 0 <= phi < 89):
            raise ValueError(f'Material {m}: require c>=0 kPa, 0<=phi<89 degrees, both finite')
        params[int(m)]={'c_kPa':c,'phi_deg':phi,'name':str(x.get('name',f'Material {m}'))}
    return params


def compute(df:pd.DataFrame, params: dict | None, formulation='bishop_pt'):
    """Conditional FS: bishop_only or bishop_pt (BBM-informed equivalent MC)."""
    if formulation not in ('bishop_pt','bishop_only'):
        raise ValueError('Unknown strength formulation')
    if formulation=='bishop_pt':
        if 'Pt_kPa' not in df:
            raise ValueError('B1 does not include Pt_kPa. Regenerate A2 and B1.')
        pt=df['Pt_kPa'].to_numpy(float)
        if not np.isfinite(pt).all() or np.any(pt < -1e-6):
            raise ValueError('BBM Pt absent/nonfinite/negative on prescribed surface; do not invent Pt.')
    else:
        pt=np.zeros(len(df),dtype=float)
    w = df.ds_weight_m.to_numpy(dtype=float)
    sig = df.Sigma_n_Bishop_compression.to_numpy(dtype=float)
    tau = df.Tau_tn_signed.to_numpy(dtype=float)
    mids = df.Material_Index.to_numpy(dtype=int)
    length = float(np.sum(w))
    net = float(np.dot(w,tau))
    plus = float(np.dot(w,np.maximum(tau,0)))
    minus = float(np.dot(w,np.maximum(-tau,0)))
    negative_length = float(np.sum(w[tau<0]))
    out = df.copy()
    out['F_mobilized_signed_kN_per_m'] = tau*w
    out['F_mobilized_positive_kN_per_m'] = np.maximum(tau,0)*w
    out['F_mobilized_negative_magnitude_kN_per_m'] = np.maximum(-tau,0)*w
    report = {
        'stage':'B2 Bishop + BBM Pt stress-integral prescribed surface',
        'strength_formulation':formulation,
        'input_quadrature_points':int(len(df)),
        'materials':sorted(map(int,np.unique(mids))),
        'surface_length_m':length,
        'integrated_mobilized_signed_kN_per_m':net,
        'integrated_mobilized_positive_kN_per_m':plus,
        'integrated_mobilized_negative_magnitude_kN_per_m':minus,
        'integrated_mobilized_absolute_kN_per_m':plus+minus,
        'negative_shear_length_m':negative_length,
        'negative_shear_length_fraction':negative_length/length,
        'negative_shear_force_fraction_of_absolute':minus/(plus+minus) if plus+minus>0 else None,
        'net_shear_positive':net>0,
        'global_fos':None,
        'strength_parameters_supplied':params is not None,
        'stress_measure':STRESS_MEASURE,
        'sign_convention':('X-increasing curve, n=(-ty,tx) into sliding material above curve; '
                           'positive tau_tn opposes sliding along +t; signed tau_m integrated'),
        'limitations':[
            'The prescribed B1 curve is DIAGNOSTIC, not validated/optimized critical surface.',
            'Bishop-effective Mohr-Coulomb post-processing is not the full BBM yield criterion.',
            'No suction, PWater or Seff is added again to effective stresses.',
            'Bishop+Pt is BBM-informed equivalent Mohr-Coulomb, not exact BBM yielding.',
            'Uniaxial local shear sign reversals retained; no absolute-value denominator.',
            'Check stress recovery sensitivity, shear traction direction, material strengths and geometry.',
            'A stress-integrated factor of safety differs from a PLAXIS phi-c strength reduction factor.',
            'Snapshot is day 100.41 and may postdate the physical failure time.',
        ]
    }
    material_rows=[]
    for m in sorted(np.unique(mids)):
        mask = mids==m
        L=float(w[mask].sum()); N=float(np.dot(w[mask],sig[mask])); T=float(np.dot(w[mask],tau[mask]))
        PT=float(np.dot(w[mask],pt[mask]))
        row = {
            'material_index':int(m),'length_m':L,
            'integrated_normal_kN_per_m':N,
            'integrated_pt_kN_per_m':PT,
            'integrated_mobilized_signed_kN_per_m':T,
            'integrated_mobilized_negative_magnitude_kN_per_m':float(np.dot(w[mask],np.maximum(-tau[mask],0))),
            'c_coefficient_m':L,'tan_phi_coefficient_kN_per_m':N+PT,
        }
        if params is not None:
            p=params[int(m)]
            cohesion_force=p['c_kPa']*L
            friction_force=math.tan(math.radians(p['phi_deg']))*(N+PT)
            row.update({'name':p['name'],'c_kPa':p['c_kPa'],'phi_deg':p['phi_deg'],
                        'cohesion_resistance_kN_per_m':cohesion_force,
                        'friction_resistance_kN_per_m':friction_force,
                        'integrated_resistance_kN_per_m':cohesion_force+friction_force})
        material_rows.append(row)

    if params is not None:
        strength = np.empty(len(df),float)
        cohesion = np.empty(len(df),float)
        friction = np.empty(len(df),float)
        for m,p in params.items():
            mask=mids==m
            cohesion[mask]=p['c_kPa']
            friction[mask]=(sig[mask]+pt[mask])*math.tan(math.radians(p['phi_deg']))
            strength[mask]=equivalent_shear_strength(sig[mask],pt[mask],p['c_kPa'],p['phi_deg'])
        out['Pt_used_kPa']=pt
        out['Cohesion_kPa']=cohesion
        out['Friction_Resistance_kPa']=friction
        out['Shear_Strength_kPa']=strength
        out['F_resisting_kN_per_m']=strength*w
        R=float(np.dot(strength,w))
        report['integrated_resistance_kN_per_m']=R
        if net>0:
            report['global_fos']=R/net
            report['status']='CONDITIONAL_FS_DIAGNOSTIC_NOT_CRITICAL'
        else:
            report['status']='NO_VALID_FS_NONPOSITIVE_SIGNED_DRIVING_TRACTION'
        report['strength_parameters']={str(m):p for m,p in params.items()}
    else:
        report['status']='STRESS_AUDIT_ONLY_NO_MATERIAL_STRENGTH_NO_FS'
    report['material_integrals']=material_rows
    return out,report


def plot(d, report, path):
    s=d.Arclength_m.to_numpy(float)
    w=d.ds_weight_m.to_numpy(float)
    tau=d.Tau_tn_signed.to_numpy(float)
    fig,axs=plt.subplots(2,1,figsize=(12,8.5),constrained_layout=True)
    axs[0].plot(s,tau,'.',markersize=3,color='#b45309',label='Signed mobilized shear (Bishop)')
    if 'Shear_Strength_kPa' in d.columns:
        axs[0].plot(s,d.Shear_Strength_kPa,'.',markersize=3,color='#2563eb',label=f'Available strength ({report["strength_formulation"]})')
    else:
        axs[0].plot(s,d.Sigma_n_Bishop_compression,'.',markersize=2,color='#2563eb',alpha=.65,
                    label='Bishop normal compression (no strength parameters)')
    neg=tau<0
    if np.any(neg):
        axs[0].scatter(s[neg],tau[neg],s=14,c='#dc2626',label='Opposite-direction shear',zorder=3)
    axs[0].axhline(0, color='black',lw=.8)
    axs[0].set_xlabel('Arclength along prescribed surface (m)')
    axs[0].set_ylabel('Stress (kPa, if PLAXIS stress unit is kPa)')
    axs[0].legend(fontsize=9); axs[0].grid(alpha=.2)
    axs[0].set_title('Local stress distribution (diagnostic curve)')

    cumulative_tau=np.cumsum(w*tau)
    axs[1].plot(s,cumulative_tau,color='#b45309',linewidth=1.8,label='Integrated signed mobilized traction')
    if 'F_resisting_kN_per_m' in d.columns:
        axs[1].plot(s,np.cumsum(d.F_resisting_kN_per_m),color='#2563eb',lw=1.8,label='Integrated available resistance')
    axs[1].axhline(0,color='black',lw=.8)
    axs[1].set_xlabel('Arclength along prescribed surface (m)')
    axs[1].set_ylabel('Integrated force per out-of-plane thickness (kN/m)')
    axs[1].legend(fontsize=9);axs[1].grid(alpha=.2)
    fs=report['global_fos']
    subtitle=(f'CONDITIONAL diagnostic FS = {fs:.4f}' if fs is not None
              else 'NO FS: enter verified material c and phi for each material')
    fig.suptitle('Stage B2 | '+report['strength_formulation']+' stress integration | '+subtitle,fontsize=12)
    fig.savefig(path,dpi=150)
    plt.close(fig)


def run_self_tests():
    # Algebraic test: single material, constant 100kPa compression, 20kPa
    # positive traction, c=10kPa and phi=30deg, arbitrary 2m quadrature.
    t=np.array([[1.0,0.0],[1.0,0.0]])
    n=np.array([[0.0,1.0],[0.0,1.0]])
    pts = pd.DataFrame({
        'Path_Segment':[1,1],'Subsegment':[1,1],'Element_ID':[1,1],
        'Material_Index':[0,0], 'X':[0.5,1.5], 'Y':[0.,0.],
        'Arclength_m':[0.5,1.5], 'ds_weight_m':[1.,1.],
        'tangent_x':t[:,0], 'tangent_y':t[:,1],
        'normal_x':n[:,0], 'normal_y':n[:,1],
        'Sigxx_E':[-110,-110], 'Sigyy_E':[-100,-100], 'Sigxy':[20.,20.],
        'Sigma_n_Bishop_compression':[100.,100.], 'Tau_tn_signed':[20.,20.]})
    tmp, report = compute(pts,{0:{'name':'synthetic','c_kPa':10,'phi_deg':30}},formulation='bishop_only')
    expected=(10+100*math.tan(math.radians(30)))/20
    assert math.isclose(report['global_fos'],expected,rel_tol=1e-12)
    # A local reverse shear must be algebraically retained, not abs or clipped.
    pts2=pts.copy()
    pts2.loc[0,'Tau_tn_signed']=-5
    pts2.loc[0,'Sigxy']=-5
    _,result2=compute(pts2,{0:{'name':'synthetic','c_kPa':10,'phi_deg':30}},formulation='bishop_only')
    assert math.isclose(result2['integrated_mobilized_signed_kN_per_m'],15,rel_tol=1e-12)
    assert math.isclose(result2['integrated_mobilized_negative_magnitude_kN_per_m'],5,rel_tol=1e-12)
    # Reversed driving shear is not 'fixed' by flipping signs/absolute values.
    pts3=pts.copy()
    pts3.loc[:,'Tau_tn_signed']=-20
    pts3.loc[:,'Sigxy']=-20
    _,result3=compute(pts3,{0:{'name':'synthetic','c_kPa':10,'phi_deg':30}},formulation='bishop_only')
    assert result3['global_fos'] is None
    pts['Pt_kPa']=[10.,10.]
    _,pt_result=compute(pts,{0:{'name':'synthetic','c_kPa':10,'phi_deg':30}},formulation='bishop_pt')
    expected_pt=(10+110*math.tan(math.radians(30)))/20
    assert math.isclose(pt_result['global_fos'],expected_pt,rel_tol=1e-12)
    print(f'Self-tests passed. Bishop-only={expected:.8f}, Bishop+Pt={expected_pt:.8f}.')


def run_stage(b1_csv, out, material_json, formulation='bishop_pt', make_plot=True):
    """Consume B1 output from disk and save FS report and quadrature strengths."""
    b1_csv=Path(b1_csv); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    df,b1_report,err=read_b1(b1_csv,b1_csv.parent/'B1_Bishop_validation_report.json')
    strength=read_strength(Path(material_json),sorted(df.Material_Index.astype(int).unique()))
    pointwise,report=compute(df,strength,formulation=formulation)
    report['b1_transform_recheck_error_kPa']=err
    report['input_b1_csv']=str(b1_csv)
    report['input_b1_sha256']=hashlib.sha256(b1_csv.read_bytes()).hexdigest()
    report['time_day']=b1_report['time_day']
    report['input_materials_file']=str(material_json)
    pointwise.to_csv(out/'B2_pointwise_strength.csv',index=False,float_format='%.12g')
    (out/'B2_report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    if make_plot: plot(pointwise,report,out/'B2_diagnostics.png')
    print(f'B2: formulation={formulation}; diagnostic FS={report["global_fos"]}')
    return out/'B2_report.json'
