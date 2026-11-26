"""Stage A2: material-wise P2 L2 recovery of Bishop effective stress and BBM Pt.

The Pt field is separate; Seff*PWater is already reflected in Bishop stresses.
Only the original stress-smoothing method is an adaptation, not a verified
reproduction of the Kim & Lee (1997) stress smoothing discretization.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from ..core.shape import p2_shape

STRESS = ['Sigxx_E', 'Sigyy_E', 'Sigxy']
STRESS_MEASURE = 'Bishop_effective_PLAXIS'
OUTPUT_STEM = 'A2_global_FE_effective_stress'
REQUIRED = ['Element_ID', 'Local_GP', 'Material_Index', 'X', 'Y'] + STRESS
# 12-point, degree-6 triangle integration rule; weights sum to unity.
WEIGHTS = np.array([0.0508449063702068]*3 + [0.116786275726379]*3
                   + [0.0828510756183736]*6, dtype=float)
# Local first six nodes in PLAXIS T15: vertices (1,2,3) and edge midpoints (4,5,6).
BARY_NODES_6 = np.array([[1,0,0],[0,1,0],[0,0,1],
                         [0.5,0.5,0],[0,0.5,0.5],[0.5,0,0.5]], dtype=float)


def read_connectivity(path):
    """The PLAXIS exported text contains 15 tab-separated rows/element;
    decimal comma and E notation in coordinates. Row 1 element/cluster;
    row 2 material label; row 3 soil region label.
    """
    df = pd.read_csv(path, sep='\t', dtype=str)
    df.columns = df.columns.str.strip()
    needed = ['Soil element', 'Node', 'Local number', 'X [m]', 'Y [m]']
    if any(c not in df for c in needed):
        raise ValueError('Connectivity columns missing; expected ' + ', '.join(needed))
    for c in needed:
        df[c] = df[c].fillna('').str.strip()
    if len(df) % 15:
        raise ValueError('Connectivity should have exactly 15 rows per triangle')
    loc = df['Local number'].to_numpy(dtype=int).reshape(-1,15)
    if not np.all(loc == np.arange(1,16)):
        raise ValueError('Unexpected local node numbering; must be 1,...,15')
    labels = df['Soil element'].to_numpy().reshape(-1,15)
    element_id = np.array([int(x.group(1)) if (x:=re.search(r'El\.\s*(\d+)', s)) else -1
                           for s in labels[:,0]], dtype=int)
    if len(np.unique(element_id)) != len(element_id) or (element_id<0).any():
        raise ValueError('Duplicate/missing connectivity element IDs')
    nodes = df['Node'].to_numpy(dtype=int).reshape(-1,15)
    xy = np.column_stack([pd.to_numeric(df[c].str.replace(',','.',regex=False)).to_numpy(dtype=float)
                          for c in ['X [m]','Y [m]']]).reshape(-1,15,2)
    if not np.isfinite(xy).all():
        raise ValueError('Non-finite connectivity coordinate')
    material_name = np.char.replace(labels[:,1].astype(str),r'\_','_')
    return element_id, nodes, xy, material_name


def gp_barycentric(tri, points):
    # tri E x 3 x 2, points E x 12 x 2. Affine geometry must be checked.
    v1 = tri[:,1]-tri[:,0]
    v2 = tri[:,2]-tri[:,0]
    dp = points - tri[:,None,0]
    det = v1[:,0]*v2[:,1] - v1[:,1]*v2[:,0]
    if np.any(np.abs(det) < 1e-12):
        raise ValueError('Zero-area element')
    l2 = (dp[:,:,0]*v2[:,None,1]-dp[:,:,1]*v2[:,None,0])/det[:,None]
    l3 = (v1[:,None,0]*dp[:,:,1]-v1[:,None,1]*dp[:,:,0])/det[:,None]
    return np.stack([1-l2-l3,l2,l3],axis=-1), np.abs(det)/2


def quad_polynomial(xy):
    x,y = xy[...,0],xy[...,1]
    return 3+0.01*x-0.02*y+0.0002*x*x-0.0001*x*y+0.0003*y*y


def run(csv_path, conn_path, out, make_plot=True):
    """Recover global P2 field and BBM Pt; all paths come from the project config."""
    csv_path, conn_path, out = Path(csv_path), Path(conn_path), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    no_plot = not make_plot
    print('Reading PLAXIS connectivity:',conn_path)
    mesh_eid, mesh_nodes, mesh_xy, mesh_names = read_connectivity(conn_path)
    print('Full mesh: {:,} triangles, {:,} unique nodes'.format(len(mesh_eid),len(np.unique(mesh_nodes))))

    print('Reading stress CSV:',csv_path)
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED if c not in df]
    if missing: raise ValueError('Missing CSV columns: '+str(missing))
    if not np.isfinite(df[REQUIRED].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite values in CSV mandatory fields')
    if 'Time' in df and df.Time.nunique()!=1:
        raise ValueError('CSV must contain a single time')
    for c in ('Element_ID','Local_GP','Material_Index'):
        arr = df[c].to_numpy(dtype=float)
        if np.any(arr!=np.round(arr)):raise ValueError('Not integer: '+c)
    # Cross-check exported effective vs total and Bishop-active water pressure,
    # if the optional PLAXIS fields exist. No recorrection is applied.
    bishop_checks = {}
    if {'Sigxx_T','Sigyy_T','PActive'}.issubset(df.columns):
        for total,effective in [('Sigxx_T','Sigxx_E'), ('Sigyy_T','Sigyy_E')]:
            e = df[total].to_numpy(dtype=float) - df[effective].to_numpy(dtype=float) - df['PActive'].to_numpy(dtype=float)
            if not np.isfinite(e).all():
                raise ValueError('Nonfinite Bishop consistency-check fields')
            bishop_checks[f'{total}_minus_{effective}_minus_PActive_max_abs'] = float(np.max(np.abs(e)))
            if np.max(np.abs(e))>1e-4:
                raise ValueError('PLAXIS total/effective stress cross-check failed')
    if {'PActive','Seff','PWater'}.issubset(df.columns):
        e = df['PActive'].to_numpy(dtype=float) - df['Seff'].to_numpy(dtype=float)*df['PWater'].to_numpy(dtype=float)
        if not np.isfinite(e).all():
            raise ValueError('Nonfinite Bishop saturation-check fields')
        bishop_checks['PActive_minus_Seff_times_PWater_max_abs'] = float(np.max(np.abs(e)))
        if np.max(np.abs(e))>1e-4:
            raise ValueError('PLAXIS Bishop saturation cross-check failed')
    print('Bishop consistency checks:', bishop_checks if bishop_checks else 'optional columns absent')
    df = df.sort_values(['Element_ID','Local_GP'],kind='stable').reset_index(drop=True)
    eid_gp = df.Element_ID.to_numpy(dtype=int)
    loc_gp = df.Local_GP.to_numpy(dtype=int)
    active_eid, first_idx, count = np.unique(eid_gp,return_index=True,return_counts=True)
    E = len(active_eid)
    if not np.all(count==12) or not np.all(loc_gp.reshape(-1,12)==np.arange(1,13)):
        raise ValueError('Every active element must have 12 uniquely numbered Gauss points')
    mat = df.Material_Index.to_numpy(dtype=int).reshape(E,12)
    if not np.all(mat==mat[:,[0]]):raise ValueError('Material change within same element')
    mat = mat[:,0]
    lookups = pd.Index(mesh_eid).get_indexer(active_eid)
    if np.any(lookups<0):raise ValueError('CSV element ID missing from mesh')
    act_nodes, act_xy, act_names = mesh_nodes[lookups],mesh_xy[lookups],mesh_names[lookups]
    xy_gp = df[['X','Y']].to_numpy(dtype=float).reshape(E,12,2)
    bary,area=gp_barycentric(act_xy[:,:3],xy_gp)
    if (bary.min() < -1e-7) or (bary.max()>1+1e-7):
        raise ValueError('Gauss point outside corresponding connectivity triangle')
    # Assumes straight P2 geometry, checked at all 15 nodes for active elements.
    b15=np.array([[4,0,0],[0,4,0],[0,0,4],[2,2,0],[0,2,2],[2,0,2],
                  [3,1,0],[1,3,0],[0,3,1],[0,1,3],[1,0,3],[3,0,1],
                  [2,1,1],[1,2,1],[1,1,2]],dtype=float)/4
    pos_expected=np.einsum('nl,elc->enc',b15,act_xy[:,:3])
    max_geom_error=float(np.max(np.linalg.norm(pos_expected-act_xy,axis=2)))
    if max_geom_error>1e-6:
        raise ValueError('Curved/nonaffine active elements detected; requires isoparametric treatment! '
                         f'Max node deviation {max_geom_error} m')
    bary_mean=bary.mean(axis=0)
    gauss_deviation=float(np.max(np.abs(bary-bary_mean[None,:,:])))
    if gauss_deviation>1e-6:raise ValueError('Inconsistent local GP numbering across elements')
    if abs(WEIGHTS.sum()-1)>1e-12:raise ValueError('Quadrature weights do not sum to 1')
    N=p2_shape(bary_mean)
    if np.max(np.abs(N.sum(axis=1)-1))>1e-11:raise ValueError('P2 shape functions fail partition of unity')

    # Duplicate shared node DOFs at boundaries between different material indices.
    pair=np.stack([np.repeat(mat,6),act_nodes[:,:6].reshape(-1)],axis=1)
    pair_unique,inv=np.unique(pair,axis=0,return_inverse=True)
    dofs=inv.reshape(E,6)
    ndof=len(pair_unique)
    # Full mesh uniqueness check: same node must have same coordinates.
    nd_id=np.unique(act_nodes[:,:6].reshape(-1),return_index=True)[1]
    # Compute nodal coordinates for P2 query evaluation and patch-test reference.
    dof_xy=np.zeros((ndof,2)); np.add.at(dof_xy,dofs.ravel(),act_xy[:,:6,:].reshape(-1,2))
    dof_counts=np.bincount(dofs.ravel(),minlength=ndof)
    dof_xy/=dof_counts[:,None]
    coord_error=float(np.max(np.linalg.norm(act_xy[:,:6,:]-dof_xy[dofs],axis=2)))
    if coord_error>1e-6:raise ValueError('Inconsistent coordinates for same node ID')
    Klocal=N.T@(WEIGHTS[:,None]*N)
    vals=(area[:,None,None]*Klocal[None,:,:]).reshape(-1)
    rows=np.broadcast_to(dofs[:,:,None],(E,6,6)).reshape(-1)
    cols=np.broadcast_to(dofs[:,None,:],(E,6,6)).reshape(-1)
    M=coo_matrix((vals,(rows,cols)),shape=(ndof,ndof)).tocsc()
    print('Active elements {:,}/{:,}; global P2 nodal DOFs {:,}; matrix nnz {:,}'.format(E,len(mesh_eid),ndof,M.nnz))
    stress_gp=df[STRESS].to_numpy(dtype=float).reshape(E,12,3)
    rhs_local=area[:,None,None]*np.einsum('gi,g,egc->eic',N,WEIGHTS,stress_gp)
    rhs=np.zeros((ndof,3),dtype=float)
    np.add.at(rhs,dofs.reshape(-1),rhs_local.reshape(-1,3))
    print('Solving global L2 projection for PLAXIS Bishop effective stresses...')
    lu=splu(M)
    stress_nodes=lu.solve(rhs)
    stress_recovered=np.einsum('gi,eic->egc',N,stress_nodes[dofs])
    if not np.isfinite(stress_nodes).all():raise ValueError('Recovered nodal stress nonfinite')

    # BBM Pt is a *strength-state variable*, not a correction to Bishop stresses.
    # Material-specific recovery: do not interpolate across distinct Material_Index.
    # Missing Pt in non-BBM materials (e.g. hard rock) is explicitly kept NaN.
    raw_pt=(df['Pt'].to_numpy(dtype=float).reshape(E,12)
            if 'Pt' in df else np.full((E,12),np.nan))
    pt_good=np.zeros(E,dtype=bool)
    pt_valid_materials=[]
    pt_missing_materials=[]
    for material in np.unique(mat):
        emask=mat==material
        pg=raw_pt[emask]
        if np.isfinite(pg).all():
            if np.any(pg < -1e-8):
                raise ValueError(f'Material {material} has negative Pt. Check BBM output.')
            pt_good[emask]=True
            pt_valid_materials.append(int(material))
        elif np.isnan(pg).all():
            pt_missing_materials.append(int(material))
        else:
            raise ValueError(f'Material {material} has partially missing or nonfinite Pt GP data.')
    rhs_pt=np.zeros(ndof,dtype=float)
    if pt_good.any():
        rhs_pt_local=area[pt_good,None]*np.einsum(
            'gi,g,eg->ei',N,WEIGHTS,raw_pt[pt_good])
        np.add.at(rhs_pt, dofs[pt_good].ravel(), rhs_pt_local.ravel())
    nodal_pt=lu.solve(rhs_pt)
    good_dof=np.zeros(ndof,dtype=bool)
    good_dof[dofs[pt_good].ravel()]=True
    nodal_pt[~good_dof]=np.nan
    recovered_pt=np.full((E,12),np.nan)
    if pt_good.any():
        recovered_pt[pt_good]=np.einsum('gi,ei->eg',N,nodal_pt[dofs[pt_good]])
    pt_min_raw=float(np.nanmin(recovered_pt)) if pt_good.any() else None
    pt_negative_gp=int(np.sum(recovered_pt < 0))
    print('BBM Pt projection raw minimum:', pt_min_raw,
          '; negative GP projections:',pt_negative_gp)
    pt_ratio_checks={}
    if 'Suction' in df:
        suction_gp=df['Suction'].to_numpy(float).reshape(E,12)
        for material in pt_valid_materials:
            mask=mat==material
            positive_s=suction_gp[mask]>1e-6
            if positive_s.any():
                ratios=raw_pt[mask][positive_s]/suction_gp[mask][positive_s]
                if not np.isfinite(ratios).all():
                    raise ValueError(f'Pt/Suction nonfinite in material {material}')
                pt_ratio_checks[str(material)]={
                    'ratio_median':float(np.median(ratios)),
                    'ratio_min':float(np.min(ratios)),
                    'ratio_max':float(np.max(ratios)),
                    'n_positive_suction_points':int(positive_s.sum())}
    print('BBM Pt recovery: present material indices', pt_valid_materials,
          '; missing material indices',pt_missing_materials)

    # Exactness patch test with synthetic quadratic tensor-free stress field.
    quad_gp=quad_polynomial(xy_gp)
    rhs_patch_local=area[:,None]*np.einsum('gi,g,eg->ei',N,WEIGHTS,quad_gp)
    rhs_patch=np.zeros(ndof,dtype=float)
    np.add.at(rhs_patch,dofs.reshape(-1),rhs_patch_local.reshape(-1))
    patch_nodal=lu.solve(rhs_patch)
    patch_error=float(np.max(np.abs(patch_nodal-quad_polynomial(dof_xy))))
    if patch_error>1e-7:raise AssertionError(f'Quadratic patch test failed: {patch_error}')
    print('Quadratic patch test: PASS. Max |recovered-exact| = {:.3e}'.format(patch_error))

    delta=stress_recovered-stress_gp
    residuals={}
    for j,name in enumerate(STRESS):
        err=delta[:,:,j].reshape(-1)
        spread=np.percentile(stress_gp[:,:,j],[5,95]); denom=max(spread[1]-spread[0],1e-12)
        residuals[name]={
            'MAE_in_sample':float(np.mean(np.abs(err))),
            'RMSE_in_sample':float(np.sqrt(np.mean(err**2))),
            'P95_abs_error_in_sample':float(np.percentile(np.abs(err),95)),
            'RMSE_over_P5_P95_spread':float(np.sqrt(np.mean(err**2))/denom),
            'raw_minmax':list(map(float,[stress_gp[:,:,j].min(),stress_gp[:,:,j].max()])),
            'recovered_minmax':list(map(float,[stress_recovered[:,:,j].min(),stress_recovered[:,:,j].max()]))
        }
        print(f'{name}: MAE {residuals[name]["MAE_in_sample"]:.4g} / RMSE {residuals[name]["RMSE_in_sample"]:.4g} (in-sample)')

    # Name mapping provided by connectivity, not inferred from model state data.
    mappings={}
    for index in np.unique(mat):
        names,cts=np.unique(act_names[mat==index],return_counts=True)
        mappings[str(index)]={str(n):int(c) for n,c in zip(names,cts)}

    np.savez_compressed(out/(OUTPUT_STEM+'.npz'),
       stress_measure=np.array(STRESS_MEASURE), stress_components=np.array(STRESS),
       stress_sign_convention=np.array('PLAXIS compression negative'),
       time=np.array(float(df.Time.iloc[0]) if 'Time' in df else np.nan),
       element_id=active_eid, material_index=mat,
       corner_nodes=act_nodes[:,:3], node_id_6=act_nodes[:,:6],
       nodal_dof_index_6=dofs, corner_xy=act_xy[:,:3],
       gauss_xy=xy_gp, gauss_local_gp=np.arange(1,13),
       gauss_stress_raw=stress_gp,gauss_stress_recovered=stress_recovered,
       material_nodal_pair=pair_unique,nodal_xy=dof_xy,nodal_stress=stress_nodes,
       gauss_weights=WEIGHTS,gauss_bary=bary_mean,
       nodal_pt=nodal_pt, gauss_pt_raw=raw_pt, gauss_pt_recovered=recovered_pt,
       pt_valid_materials=np.asarray(pt_valid_materials,dtype=int),
       pt_missing_materials=np.asarray(pt_missing_materials,dtype=int),
       gauss_PWater=df.PWater.to_numpy(dtype=float).reshape(E,12) if 'PWater' in df else np.full((E,12),np.nan),
       gauss_Seff=df.Seff.to_numpy(dtype=float).reshape(E,12) if 'Seff' in df else np.full((E,12),np.nan))
    report={
      'stage':'A2 global P2 L2 projection of PLAXIS Bishop effective stresses',
      'stress_measure':STRESS_MEASURE, 'stress_components':STRESS,
      'stress_sign_convention':'PLAXIS compression negative',
      'Bishop_export_consistency_checks':bishop_checks,
      'not_asserted_identical_to_Kim_Lee':True,
      'csv':str(csv_path),'connectivity':str(conn_path),
      'time':float(df.Time.iloc[0]) if 'Time' in df else None,
      'mesh_elements_all':int(len(mesh_eid)),'mesh_nodes_all':int(len(np.unique(mesh_nodes))),
      'active_elements':int(E),'inactive_elements':int(len(mesh_eid)-E),
      'stress_points':int(len(df)),'DOFs_material_duplicated':int(ndof),
      'material_name_map':mappings,
      'quadrature':'12-point triangle degree-six, normalized weights',
      'stress_recovery_basis':'Quadratic P2, local node order 1..6 in T15 element',
      'full_mesh_geometry_max_error_active_m':max_geom_error,
      'max_gauss_bary_variation':gauss_deviation,
      'shared_node_max_coord_difference_m':coord_error,
      'patch_test_quadratic_max_abs_error':patch_error,
      'GP_residuals':residuals,
      'Pt_recovery_valid_materials':pt_valid_materials,
      'Pt_recovery_missing_materials':pt_missing_materials,
      'Pt_raw_recovered_min_kPa':pt_min_raw,
      'Pt_raw_GP_ratio_to_suction':pt_ratio_checks,
      'Pt_negative_recovered_GP_count':pt_negative_gp,
      'Pt_nonnegative_evaluation_rule':'max(0, recovered Pt); raw values retained for audit',
      'Pt_is_separate_strength_variable_not_Bishop_stress_correction': True,
      'limitations':[
        'An L2 projection, not verified as identical to Kim & Lee paper global smoothing algorithm',
        'Smoothing is performed independently across Material_Index, including 5 vs 8',
        'Quadratic recovery resolves lower order stresses than 15-node quartic displacement elements',
        'In-sample GP residual is not out-of-sample validation or physical accuracy',
        'Stress gradients near failure/shear bands may be attenuated or overshoot',
        'Requires active triangles for evaluating new coordinates; inactive elements are excluded',
        'PWater/Seff are retained only for audit; no suction correction is added to Bishop effective stresses',
        'BBM Pt is independently projected as a strength-related field; not an exact BBM yield surface',
        'Do not use this file with older B1 versions that assume TOTAL stresses or label Sigxx_T/Sigyy_T',
        'Verify original vs deformed coordinates in PLAXIS before future-stage curve evaluation'
      ]}
    (out/(OUTPUT_STEM+'_report.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')

    if not no_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        gg=xy_gp.reshape(-1,2); ori=stress_gp.reshape(-1,3); rcv=stress_recovered.reshape(-1,3)
        focus=(gg[:,0]>=30)&(gg[:,0]<=105)&(gg[:,1]>=18)&(gg[:,1]<=85)
        if focus.sum()<100:focus[:]=True
        idx=np.flatnonzero(focus)
        if len(idx)>35000:idx=np.random.default_rng(20261121).choice(idx,35000,replace=False)
        left,right=np.percentile(np.concatenate([ori[idx,2],rcv[idx,2]]),[1,99])
        diff=rcv[idx,2]-ori[idx,2]
        lim=max(np.percentile(np.abs(diff),98),1e-10)
        fig,axs=plt.subplots(2,2,figsize=(13,9.5),constrained_layout=True)
        for ax,s,title,vmin,vmax in [
           (axs[0,0],ori[idx,2],'Original GP shear stress (effective tensor)',left,right),
           (axs[0,1],rcv[idx,2],'Global FE P2 recovered shear stress (effective tensor)',left,right),
           (axs[1,0],diff,'P2 recovered - raw GP stress',-lim,lim)]:
            im=ax.scatter(gg[idx,0],gg[idx,1],s=1.2,c=s,cmap='coolwarm',vmin=vmin,vmax=vmax,rasterized=True)
            ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)');fig.colorbar(im,ax=ax,label='PLAXIS stress unit')
        axs[1,1].hist(diff,bins=100)
        axs[1,1].set_title('Shear stress residual distribution');axs[1,1].set_xlabel('Recovered minus raw');axs[1,1].set_ylabel('GP count')
        fig.suptitle('A2 Bishop effective stress projection • day {:.4f} • {} active T15 elements'.format(report['time'],E))
        fig.savefig(out/(OUTPUT_STEM+'_diagnostics.png'),dpi=150)
        plt.close(fig)
    print('Saved to:',out)
    print('STATUS: Bishop-effective stress and separate Pt recovery complete. No FS.')
    return out/(OUTPUT_STEM+'.npz')


