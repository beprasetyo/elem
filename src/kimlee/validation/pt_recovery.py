"""Stage Pt-Audit: provenance and nonnegative-recovery sensitivity checks.

Audits Gauss-point Pt recovery and compares FOUR fixed-surface strength
assumptions WITHOUT reoptimizing the slip surface. This is a diagnostic,
not proof that any alternative is a correct BBM constitutive representation.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from ..stages.b1 import BishopField, cross
from ..core.shape import p2_shape

SURFACE_FILES = {
    'B2_prescribed_diagnostic': ('B1','B1_Bishop_quadrature_points.csv'),
    'C2_best_refined_fixed_geometry': ('C2','C2_best_surface_pointwise.csv'),
}


def _compare_at_points(field:BishopField, xy):
    xy=np.atleast_2d(np.asarray(xy,dtype=float))
    tri_idx=np.asarray(field.finder(xy[:,0],xy[:,1]),dtype=int)
    if np.any(tri_idx<0):
        raise ValueError('Pt audit points outside active elements')
    tri=field.cxy[tri_idx]
    v1=tri[:,1]-tri[:,0];v2=tri[:,2]-tri[:,0];dp=xy-tri[:,0]
    det=cross(v1,v2)
    l2=cross(dp,v2)/det;l3=cross(v1,dp)/det
    bary=np.column_stack((1-l2-l3,l2,l3))
    if np.any(bary<-1e-7):
        raise ValueError('Pt audit point outside located element')
    N=p2_shape(bary)
    node=field.pt_nodes[field.dof[tri_idx]]
    raw=np.einsum('qi,qi->q',N,node)
    p2_gpclip=np.maximum(raw,0.0)
    # Diagnostic alternate: first clip nodal Pt, then interpolate and clip
    # remaining P2 undershoot (P2 basis contains negative shape function lobes).
    p2_nodalclip=np.maximum(np.einsum('qi,qi->q',N,np.maximum(node,0)),0)
    # Positivity-preserving P1 using ONLY the three recovered corner values;
    # lower-order field, deliberately not recommended as "more accurate".
    p1_cornerclip=np.einsum('qi,qi->q',bary,np.maximum(node[:,:3],0.0))
    return {'raw_P2':raw,'GP_clipped_P2':p2_gpclip,
            'nodal_clipped_P2':p2_nodalclip,'positive_P1_corner':p1_cornerclip},tri_idx


def _weighted_stats(mask, w, orig, pred):
    ww=w[mask]; yy=orig[mask]; hh=pred[mask]
    w_sum=np.sum(ww)
    return {'weighted_MAE_kPa':float(np.sum(ww*np.abs(yy-hh))/w_sum),
            'weighted_RMSE_kPa':float(np.sqrt(np.sum(ww*(yy-hh)**2)/w_sum)),
            'weighted_bias_kPa':float(np.sum(ww*(hh-yy))/w_sum)}


def gp_audit(z, field):
    raw=z['gauss_pt_raw'].astype(float)
    recovered=z['gauss_pt_recovered'].astype(float)
    mat=z['material_index'].astype(int)
    tri=z['corner_xy'].astype(float)
    area=np.abs(cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]))/2
    # quadrature weights sum to one, as used in A2 mass projection
    w=area[:,None]*np.broadcast_to(z['gauss_weights'],raw.shape)
    valid=np.isfinite(raw)&np.isfinite(recovered)
    assert np.isfinite(raw[valid]).all()
    out=[]
    for m in np.unique(mat):
        msk=(mat[:,None]==m)&valid
        if not msk.any():
            continue
        orig=raw[msk];pred=recovered[msk];ww=w[msk]
        neg=pred<0
        original_negative=int(np.sum(orig<-1e-9))
        out.append({'material_index':int(m), 'count_GP':int(msk.sum()),
                    'raw_GP_negative_count':original_negative,
                    'raw_GP_mean_kPa':float(np.mean(orig)),
                    'raw_GP_max_kPa':float(np.max(orig)),
                    'recovered_GP_min_kPa':float(np.min(pred)),
                    'recovered_GP_negative_count':int(neg.sum()),
                    'recovered_GP_negative_fraction':float(np.mean(neg)),
                    'recovered_GP_negative_area_weight_fraction':float(np.sum(ww[neg])/np.sum(ww)),
                    'negative_deficit_area_weighted_kPa':float(-np.sum(ww[neg]*pred[neg])/np.sum(ww)),
                    'P2_raw_MAE_kPa':float(np.sum(ww*np.abs(pred-orig))/np.sum(ww)),
                    'P2_GPclipped_MAE_kPa':float(np.sum(ww*np.abs(np.maximum(pred,0)-orig))/np.sum(ww)),
                    'P2_raw_RMSE_kPa':float(np.sqrt(np.sum(ww*(pred-orig)**2)/np.sum(ww))),
                    'P2_GPclipped_RMSE_kPa':float(np.sqrt(np.sum(ww*(np.maximum(pred,0)-orig)**2)/np.sum(ww))),
                    'raw_GP_near_zero_fraction_1e-6':float(np.mean(orig<=1e-6))})
    return out


def audit_surface(field, surface_path, strength):
    if not surface_path.exists():return None
    df=pd.read_csv(surface_path)
    required={'X','Y','ds_weight_m','Sigma_n_Bishop_compression','Tau_tn_signed','Material_Index'}
    if not required.issubset(df.columns):
        raise ValueError(f'Cannot audit {surface_path}: missing {sorted(required-set(df.columns))}')
    if not np.isfinite(df[list(required)].to_numpy(float)).all():
        raise ValueError(f'Nonfinite surface data in {surface_path}')
    xy=df[['X','Y']].to_numpy(float)
    variants,tri=_compare_at_points(field,xy)
    ids=df.Material_Index.to_numpy(int)
    if not np.array_equal(ids,field.mid[tri]):
        raise ValueError('Inconsistent material IDs between saved surface and A2')
    w=df.ds_weight_m.to_numpy(float)
    norm=df.Sigma_n_Bishop_compression.to_numpy(float)
    T=float(np.dot(w,df.Tau_tn_signed.to_numpy(float)))
    if T<=0:raise ValueError('Nonpositive signed driving force; no meaningful FS sensitivity')
    c=np.zeros(len(df));tanphi=np.zeros(len(df))
    for m in np.unique(ids):
        p=strength[int(m)]
        c[ids==m]=float(p['c_kPa'])
        tanphi[ids==m]=math.tan(math.radians(float(p['phi_deg'])))
    base=float(np.dot(w,c+norm*tanphi))
    results={}
    for name,pt in variants.items():
        R=base+float(np.dot(w,pt*tanphi))
        results[name]={'R_kN_per_m':R,'T_signed_kN_per_m':T,'FS_conditional':R/T,
                       'integrated_Pt_kN_per_m':float(np.dot(w,pt)),
                       'negative_Pt_points_before_cap':int(np.sum(pt<-1e-12)),
                       'min_Pt_used_kPa':float(np.min(pt))}
    current=results['GP_clipped_P2']['FS_conditional']
    for result in results.values():
        result['FS_difference_vs_current']=result['FS_conditional']-current
        result['FS_relative_difference_percent']=100*(result['FS_conditional']/current-1)
    # Confirm saved B1/C2 B2 used the identical maximum(0,raw) rule
    if 'Pt_kPa' in df.columns:
        maxdiff=float(np.max(np.abs(df.Pt_kPa.to_numpy(float)-variants['GP_clipped_P2'])))
        if maxdiff>1e-5:raise ValueError(f'Saved Pt field differs from re-evaluation by {maxdiff}')
    else: maxdiff=None
    return {'point_count':int(len(df)), 'length_m':float(np.sum(w)),
            'saved_Pt_vs_recomputed_max_abs_kPa':maxdiff,
            'T_signed_kN_per_m':T,'basis_FSBishop_only':base/T,
            'Pt_evaluation_variants':results}


def run(a2_npz, results_root, materials_path, out, make_plot=True):
    a2_npz=Path(a2_npz); results_root=Path(results_root)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    field=BishopField(a2_npz)
    if field.pt_nodes is None:
        raise ValueError('Pt audit requires A2 file with Pt nodal recovery')
    with np.load(a2_npz,allow_pickle=False) as z:
        table=gp_audit(z,field)
        raw_gp=z['gauss_pt_raw'].astype(float)
        smooth_gp=z['gauss_pt_recovered'].astype(float)
        mat_gp=z['material_index'].astype(int)
    params={int(k):v for k,v in json.loads(Path(materials_path).read_text(encoding='utf-8')).items()}
    surfaces={}
    for name,parts in SURFACE_FILES.items():
        path=results_root/parts[0]/parts[1]
        if path.exists():surfaces[name]=audit_surface(field,path,params)
    report={'stage':'Pt recovery audit','A2_file':str(a2_npz),
            'description':'Unconstrained P2 Pt recovery negativity and fixed-geometry FS sensitivity',
            'GP_statistics_per_material':table,
            'fixed_geometry_sensitivity':surfaces,
            'limitations':[
                'Original GP Pt is a constitutive state output, not nodal ground truth.',
                'GP clipping is a nonnegativity heuristic, not a constrained projection.',
                'Alternatives in this audit are sensitivities, NOT validated substitutes for Pt recovery.',
                'Positive P1 corner interpolation sacrifices P2 resolution; nodal-clipped P2 can overshoot.',
                'Fixed surface FS sensitivity does not show how the optimized C2 surface or minimum changes.',
                'Bishop+Pt strength is an approximate BBM-informed MC postprocessing criterion.'
            ]}
    (out/'Pt_audit_report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    pd.DataFrame(table).to_csv(out/'Pt_audit_GP_material_statistics.csv',index=False)
    rows=[]
    for surf,details in surfaces.items():
        for mode,rec in details['Pt_evaluation_variants'].items():
            rows.append(dict(surface=surf,variant=mode,**rec))
    pd.DataFrame(rows).to_csv(out/'Pt_audit_fixed_surface_sensitivity.csv',index=False)
    if make_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,2,figsize=(13.1,5.2),constrained_layout=True)
        for m in sorted(set(mat_gp.tolist())):
            mask=(mat_gp==m)[:,None]&np.isfinite(raw_gp)&np.isfinite(smooth_gp)
            if not mask.any():continue
            orig=raw_gp[mask];smooth=smooth_gp[mask]
            focus=(orig<=5.)&(smooth<=6.)
            orig=orig[focus];smooth=smooth[focus]
            # Deterministic subsample for legible scatter.
            if len(orig)>9000:
                ii=np.linspace(0,len(orig)-1,9000,dtype=int)
                orig=orig[ii];smooth=smooth[ii]
            axs[0].scatter(orig,smooth,s=2,alpha=.16,label=f'Mat {m}',rasterized=True)
        axs[0].plot([0,5],[0,5],color='black',lw=1,ls='--',label='Identity')
        axs[0].axhline(0,color='#e11d48',lw=1.1)
        axs[0].set_xlim(-.12,5.1);axs[0].set_ylim(-1.4,5.5)
        axs[0].set_xlabel('Original Gauss Pt (kPa)')
        axs[0].set_ylabel('Global P2 recovered Pt (kPa)')
        axs[0].set_title('Pt recovery near zero (undershoot region)')
        axs[0].legend(fontsize=8)
        labels=[];deltas=[]
        for surf,detail in surfaces.items():
            for variant in ('raw_P2','nodal_clipped_P2','positive_P1_corner'):
                rr=detail['Pt_evaluation_variants'][variant]
                labels.append(surf.split('_')[0]+' / '+variant.replace('_',' '))
                deltas.append(1000*rr['FS_difference_vs_current'])
        if deltas:
            xx=np.arange(len(deltas))
            axs[1].barh(xx,deltas,color=['#64748b' if d<0 else '#0d9488' for d in deltas])
            axs[1].set_yticks(xx,labels,fontsize=8)
            axs[1].invert_yaxis()
            axs[1].axvline(0,color='black',lw=.8)
            axs[1].set_xlabel('FS difference vs current GP-clipped P2 (×10⁻³)')
            axs[1].set_title('Fixed-surface Pt treatment sensitivity')
        else:
            axs[1].text(.1,.5,'No B1/C2 pointwise output found',transform=axs[1].transAxes)
        fig.savefig(out/'Pt_audit_diagnostics.png',dpi=160)
        plt.close(fig)
    print('Pt Audit: per material GP statistics')
    for row in table:
        print(f"  Mat {row['material_index']}: negatives {row['recovered_GP_negative_count']}/"
              f"{row['count_GP']} ({100*row['recovered_GP_negative_fraction']:.2f}%), "
              f"min={row['recovered_GP_min_kPa']:.4f} kPa")
    for name,details in surfaces.items():
        variants=details['Pt_evaluation_variants']
        print(f"  {name}: FS current={variants['GP_clipped_P2']['FS_conditional']:.6f}; "
              f"P1 sensitivity={variants['positive_P1_corner']['FS_conditional']:.6f}")
    print('Pt Audit saved:',out)
    return out/'Pt_audit_report.json'
