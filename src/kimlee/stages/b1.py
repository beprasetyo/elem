"""Stage B1: integrate Bishop stress and recovered Pt on a prescribed curve.

The prescribed curve is diagnostic. Exact FE-edge splits, GL3 quadrature,
material-wise interpolation, and explicit sign orientation are preserved.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from ..core.shape import p2_shape
from ..core.mechanics import resolve_bishop_traction

GL3_X, GL3_W = np.polynomial.legendre.leggauss(3)
EXPECTED_MEASURE = 'Bishop_effective_PLAXIS'
EXPECTED_COMPONENTS = ['Sigxx_E', 'Sigyy_E', 'Sigxy']


def cross(a, b):
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


class BishopField:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            if 'stress_measure' not in z or 'stress_components' not in z:
                raise ValueError('A2 field lacks stress_measure/stress_components metadata. '
                                 'Regenerate A2 using Bishop effective stress script.')
            measure = str(z['stress_measure'].item())
            components = z['stress_components'].astype(str).tolist()
            if measure != EXPECTED_MEASURE or components != EXPECTED_COMPONENTS:
                raise ValueError(
                    'WRONG A2 FIELD: requires Bishop effective [Sigxx_E, Sigyy_E, Sigxy]. '
                    f'Found measure={measure!r}, components={components!r}. '
                    'Do not use the old total-stress A2 NPZ.')
            sign = str(z['stress_sign_convention'].item()) if 'stress_sign_convention' in z else ''
            if sign != 'PLAXIS compression negative':
                raise ValueError(f'Unexpected stress sign convention: {sign!r}')
            self.time = float(z['time'].item())
            self.eid = z['element_id'].astype(int)
            self.mid = z['material_index'].astype(int)
            self.corner_nodes = z['corner_nodes'].astype(int)
            self.cxy = z['corner_xy'].astype(float)
            self.dof = z['nodal_dof_index_6'].astype(int)
            self.stress_nodes = z['nodal_stress'].astype(float)
            self.pt_nodes = z['nodal_pt'].astype(float) if 'nodal_pt' in z else None
        self.E = len(self.eid)
        if self.cxy.shape != (self.E,3,2) or self.dof.shape != (self.E,6):
            raise ValueError('Unexpected A2 element geometry / P2 DOF dimensions')
        if self.stress_nodes.ndim != 2 or self.stress_nodes.shape[1] != 3:
            raise ValueError('A2 nodal_stress must have 3 Bishop effective components')
        if not np.isfinite(self.cxy).all() or not np.isfinite(self.stress_nodes).all():
            raise ValueError('Nonfinite geometry/effective stress in A2 field')
        if self.dof.min() < 0 or self.dof.max() >= len(self.stress_nodes):
            raise ValueError('Nodal DOF index outside nodal stress array')
        if len(np.unique(self.eid)) != self.E:
            raise ValueError('Duplicate active FE element ID in A2 field')
        # The A2 field contains affine straight-edged triangle geometry.
        self.areas = np.abs(cross(self.cxy[:,1]-self.cxy[:,0],
                                  self.cxy[:,2]-self.cxy[:,0]))/2
        if np.any(self.areas <= 0):
            raise ValueError('Degenerate triangles in active mesh')
        # TriFinder on *active* FEM elements only (do not interpolate outside).
        nodes, inv = np.unique(self.corner_nodes.ravel(), return_inverse=True)
        coords = np.zeros((len(nodes),2),float)
        np.add.at(coords,inv,self.cxy.reshape(-1,2))
        counts=np.bincount(inv, minlength=len(nodes))
        coords /= counts[:,None]
        max_node_diff = float(np.max(np.linalg.norm(coords[inv].reshape(self.E,3,2)-self.cxy,axis=2)))
        if max_node_diff > 1e-6:
            raise ValueError(f'Shared mesh corners have inconsistent coordinates ({max_node_diff:g} m)')
        self.triang = mtri.Triangulation(coords[:,0],coords[:,1], triangles=inv.reshape(-1,3))
        self.finder = self.triang.get_trifinder()
        # Unique mesh edges: provides exact intersections for the prescribed curve.
        edge_nodes=np.concatenate((self.corner_nodes[:,[0,1]], self.corner_nodes[:,[1,2]],
                                   self.corner_nodes[:,[2,0]]), axis=0)
        edge_xy=np.concatenate((self.cxy[:,[0,1]],self.cxy[:,[1,2]],
                                self.cxy[:,[2,0]]),axis=0)
        order=np.argsort(edge_nodes,axis=1)
        edge_nodes=np.take_along_axis(edge_nodes,order,axis=1)
        edge_xy=np.take_along_axis(edge_xy,order[:,:,None],axis=1)
        _,unq=np.unique(edge_nodes,axis=0,return_index=True)
        self.edges=edge_xy[unq]

    def evaluate(self, xy):
        """Return element index and stress tensor components for in-domain locations."""
        xy=np.atleast_2d(np.asarray(xy,dtype=float))
        ti=np.asarray(self.finder(xy[:,0],xy[:,1]),dtype=int)
        if np.any(ti<0):
            raise ValueError('Evaluation point outside active FEM triangles')
        tri=self.cxy[ti]
        v1=tri[:,1]-tri[:,0]
        v2=tri[:,2]-tri[:,0]
        dp=xy-tri[:,0]
        det=cross(v1,v2)
        l2=cross(dp,v2)/det
        l3=cross(v1,dp)/det
        bary=np.column_stack((1-l2-l3,l2,l3))
        if np.any(bary < -1e-7) or np.any(bary > 1+1e-7):
            raise ValueError('Interpolation barycentric point outside triangle')
        N=p2_shape(bary)
        return ti,np.einsum('qi,qic->qc',N,self.stress_nodes[self.dof[ti]])

    def evaluate_pt(self, xy, raw=False):
        """Interpolate *separate* BBM Pt field with same material-wise P2 basis."""
        if self.pt_nodes is None:
            raise ValueError('A2 file has no recovered BBM Pt; rerun new A2.')
        xy=np.atleast_2d(np.asarray(xy,float))
        ti=np.asarray(self.finder(xy[:,0],xy[:,1]),int)
        if np.any(ti<0): raise ValueError('Pt point outside active FEM mesh')
        tri=self.cxy[ti]
        v1=tri[:,1]-tri[:,0]; v2=tri[:,2]-tri[:,0]
        dp=xy-tri[:,0]; det=cross(v1,v2)
        l2=cross(dp,v2)/det; l3=cross(v1,dp)/det
        N=p2_shape(np.column_stack((1-l2-l3,l2,l3)))
        raw_values=np.einsum('qi,qi->q',N,self.pt_nodes[self.dof[ti]])
        # A global unconstrained P2 projection may undershoot below physical Pt=0.
        # Preserve raw Pt separately in B1; cap only strength-evaluation value.
        return raw_values if raw else np.maximum(raw_values,0.)

    def segment_breaks(self,p0,p1):
        """Intersect a line segment with all active triangle edges; reject edge overlap."""
        v=p1-p0
        length=float(np.linalg.norm(v))
        if length < 1e-10:
            raise ValueError('Zero-length segment of prescribed surface')
        a=self.edges[:,0]
        b=self.edges[:,1]
        v_edge=b-a
        bbox=(np.minimum(a[:,0],b[:,0])<=max(p0[0],p1[0])+1e-9) & (np.maximum(a[:,0],b[:,0])>=min(p0[0],p1[0])-1e-9) & (np.minimum(a[:,1],b[:,1])<=max(p0[1],p1[1])+1e-9) & (np.maximum(a[:,1],b[:,1])>=min(p0[1],p1[1])-1e-9)
        a,b,v_edge=a[bbox],b[bbox],v_edge[bbox]
        denom=cross(v,v_edge)
        rel=a-p0
        parallel=np.abs(denom)<1e-12*length*np.linalg.norm(v_edge,axis=1)
        collinear=parallel & (np.abs(cross(rel,np.broadcast_to(v,rel.shape)))<1e-9*length)
        if collinear.any():
            along_a=(a[collinear]-p0)@v/(length**2)
            along_b=(b[collinear]-p0)@v/(length**2)
            overlap=np.minimum(1,np.maximum(along_a,along_b))-np.maximum(0,np.minimum(along_a,along_b))
            if np.any(overlap>1e-8):
                raise ValueError('Surface overlaps FE edge; move surface to chosen material side')
        notparallel=~parallel
        t=cross(rel[notparallel],v_edge[notparallel])/denom[notparallel]
        u=cross(rel[notparallel],np.broadcast_to(v,rel[notparallel].shape))/denom[notparallel]
        valid=(t>1e-10)&(t<1-1e-10)&(u>=-1e-9)&(u<=1+1e-9)
        return np.unique(np.round(np.r_[0,t[valid],1],decimals=11))


def read_curve(path):
    data=pd.read_csv(path)
    if not {'X','Y'}.issubset(data.columns):
        raise ValueError('Surface CSV needs columns X,Y')
    points=data[['X','Y']].to_numpy(dtype=float)
    if len(points)<2 or not np.isfinite(points).all():
        raise ValueError('Surface must contain at least 2 finite coordinates')
    if np.any(np.linalg.norm(np.diff(points,axis=0),axis=1)<1e-9):
        raise ValueError('Consecutive duplicate surface points')
    return points


def run_b1(mesh,curve):
    rec=[]
    full_length=0.0
    outside_len=0.0
    bad=[]
    fe_subsegments=0
    prev_material=None
    material_crossings=0
    # Use a consistent tangent direction defined by input CSV row order.
    for k in range(len(curve)-1):
        p0,p1=curve[k:k+2]
        v=p1-p0
        L=float(np.linalg.norm(v))
        tangent=v/L
        normal=np.array((-tangent[1],tangent[0]))
        t_breaks=mesh.segment_breaks(p0,p1)
        for a,b in zip(t_breaks[:-1],t_breaks[1:]):
            if b-a < 1e-10:
                continue
            midpoint=p0+v*(a+b)/2
            local_id=int(mesh.finder(midpoint[0],midpoint[1]))
            length=L*(b-a)
            if local_id < 0:
                outside_len+=length
                bad.append([k+1,float(a),float(b)])
                continue
            current_material=int(mesh.mid[local_id])
            if prev_material is not None and current_material!=prev_material:
                material_crossings+=1
            prev_material=current_material
            fe_subsegments+=1
            u=(a+b)/2+(b-a)/2*GL3_X
            xy=p0[None,:]+u[:,None]*v[None,:]
            weights=L*(b-a)/2*GL3_W
            triangles,stress=mesh.evaluate(xy)
            pt_raw=(mesh.evaluate_pt(xy,raw=True) if mesh.pt_nodes is not None
                else np.full(len(xy),np.nan))
            pt=np.maximum(pt_raw,0.)
            # PLAXIS effective stress uses compression negative.
            sxx,syy,sxy=stress[:,0],stress[:,1],stress[:,2]
            tx,ty=tangent
            nx,ny=normal
            normal_compression,tau=resolve_bishop_traction(
                stress,np.broadcast_to(tangent,(len(stress),2)))
            for j in range(len(GL3_X)):
                rec.append(dict(
                    Path_Segment=k+1,Subsegment=fe_subsegments,
                    Element_ID=int(mesh.eid[triangles[j]]),
                    Material_Index=int(mesh.mid[triangles[j]]),
                    X=float(xy[j,0]),Y=float(xy[j,1]),
                    Arclength_m=float(full_length+u[j]*L),
                    ds_weight_m=float(weights[j]),
                    tangent_x=float(tx),tangent_y=float(ty),
                    normal_x=float(nx),normal_y=float(ny),
                    Sigxx_E=float(sxx[j]),Sigyy_E=float(syy[j]),Sigxy=float(sxy[j]),
                    Sigma_n_Bishop_compression=float(normal_compression[j]),
                    Tau_tn_signed=float(tau[j]),
                    Pt_kPa=float(pt[j]),Pt_recovered_raw_kPa=float(pt_raw[j]),
                ))
        full_length+=L
    if outside_len>1e-7:
        raise ValueError(f'Surface crosses {outside_len:.6g} m of outside/inactive mesh; '
                         f'bad intervals {bad[:8]}. No extrapolation permitted.')
    if not rec:
        raise ValueError('No FEM quadrature points produced; check surface and mesh')
    df=pd.DataFrame.from_records(rec).sort_values('Arclength_m',kind='stable').reset_index(drop=True)
    error=float(df['ds_weight_m'].sum()-full_length)
    if abs(error)>1e-7*max(1.,full_length):
        raise ValueError('Quadrature weights do not sum to curve length')
    # Independent symmetric tensor check for every point.
    t=df[['tangent_x','tangent_y']].to_numpy()
    n=df[['normal_x','normal_y']].to_numpy()
    sigma=np.zeros((len(df),2,2))
    sigma[:,0,0]=df.Sigxx_E.to_numpy()
    sigma[:,1,1]=df.Sigyy_E.to_numpy()
    sigma[:,0,1]=sigma[:,1,0]=df.Sigxy.to_numpy()
    sigcheck=-np.einsum('ni,nij,nj->n',n,sigma,n)
    taucheck=np.einsum('ni,nij,nj->n',t,sigma,n)
    transform_error=float(max(np.max(np.abs(sigcheck-df.Sigma_n_Bishop_compression.to_numpy())),
                               np.max(np.abs(taucheck-df.Tau_tn_signed.to_numpy()))))
    if transform_error>1e-10:
        raise AssertionError('Tensor stress transformation self-test failed')
    report={
        'stage':'B1 prescribed slip surface — Bishop effective stress (NO FS)',
        'status':'PASS numerical geometry/field query; diagnostic surface only',
        'stress_measure':EXPECTED_MEASURE,
        'stress_components':EXPECTED_COMPONENTS,
        'time_day':float(mesh.time),
        'total_length_m':float(full_length),
        'FE_subsegments':fe_subsegments,
        'gauss_integration_points':int(len(df)),
        'unique_intersected_elements':int(df.Element_ID.nunique()),
        'unique_material_indices':sorted(int(m) for m in df.Material_Index.unique()),
        'material_transitions':material_crossings,
        'outside_domain_length_m':float(outside_len),
        'quadrature_length_error_m':error,
        'effective_stress_transform_selftest_max_abs':transform_error,
        'Sigma_n_Bishop_compression_minmax':list(map(float,[df.Sigma_n_Bishop_compression.min(),df.Sigma_n_Bishop_compression.max()])),
        'Tau_tn_signed_minmax':list(map(float,[df.Tau_tn_signed.min(),df.Tau_tn_signed.max()])),
        'negative_effective_normal_compression_points':int((df.Sigma_n_Bishop_compression<0).sum()),
        'pt_finite_quadrature_points':int(np.isfinite(df.Pt_kPa.to_numpy()).sum()),
        'pt_clipped_quadrature_points':int(np.sum(df.Pt_recovered_raw_kPa.to_numpy()<0)),
        'pt_raw_min_kPa':float(np.nanmin(df.Pt_recovered_raw_kPa.to_numpy())) if df.Pt_kPa.notna().any() else None,
        'pt_missing_materials': sorted(int(m) for m in df.loc[df.Pt_kPa.isna(),'Material_Index'].unique()),
        'limitations':[
            'This is a prescribed DIAGNOSTIC curve, not a verified/critical failure surface',
            'Effective normal stress is Bishop stress from PLAXIS, with compression positive in output',
            'PWater/Suction/Seff are NOT used or added again to stresses',
            'Signed tangential traction is not yet defined as mobilized driving shear force',
            'Pt is interpolated separately from Bishop stress (no double-counting); no FS here',
            'No c, phi, phi_b, shear resistance, factor of safety, or slip surface optimization',
            'A2 P2 recovery can smooth shear-band stress peaks and is not validated identical to Kim & Lee',
            'No extrapolation outside active mesh; FE material boundaries are respected',
            'Check deformed vs original coordinate reference consistency before physical interpretation'
        ]
    }
    return df,report


def plot_b1(mesh,curve,df,out,gp_csv=None):
    fig,axes=plt.subplots(2,2,figsize=(14,9.5),constrained_layout=True)
    ax=axes[0,0]
    if gp_csv:
        d=pd.read_csv(gp_csv,usecols=['X','Y','Gamma_incremental'])
        minx,maxx=curve[:,0].min()-8,curve[:,0].max()+8
        miny,maxy=curve[:,1].min()-8,curve[:,1].max()+8
        d=d[d.X.between(minx,maxx)&d.Y.between(miny,maxy)]
        if len(d)>30000:
            d=d.iloc[::int(np.ceil(len(d)/30000))]
        vals=np.log10(np.clip(d.Gamma_incremental.to_numpy(dtype=float),0,None)+1e-8)
        s=ax.scatter(d.X,d.Y,c=vals,cmap='magma',vmin=-6,vmax=-0.3,s=3,rasterized=True)
        fig.colorbar(s,ax=ax,label='log10 incremental shear strain')
    else:
        ax.tripcolor(mesh.triang,facecolors=mesh.mid.astype(float),edgecolors='none',cmap='tab20',alpha=.25)
    ax.plot(curve[:,0],curve[:,1],'-o',color='#00cbe8',lw=2.5,ms=4,label='Prescribed diagnostic curve')
    ax.scatter(df.X,df.Y,c='#eab308',s=2,alpha=.65,label='Quadrature locations',rasterized=True)
    ax.set_aspect('equal',adjustable='box');ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)')
    ax.set_title('(a) Surface and quadrature locations')
    ax.legend(loc='lower left',fontsize=8)
    ax.set_xlim(curve[:,0].min()-8,curve[:,0].max()+8)
    ax.set_ylim(curve[:,1].min()-10,curve[:,1].max()+10)

    ax=axes[0,1]
    ax.plot(df.Arclength_m,df.Sigma_n_Bishop_compression,'.',ms=2.5,color='#1d4ed8')
    ax.axhline(0,lw=.7,color='black',alpha=.35)
    ax.set_title('(b) Bishop effective normal stress')
    ax.set_xlabel('Arclength (m)');ax.set_ylabel('sigma_n^B (compression positive; PLAXIS unit)')
    ax.grid(alpha=.25)

    ax=axes[1,0]
    ax.plot(df.Arclength_m,df.Tau_tn_signed,'.',ms=2.5,color='#c2410c')
    ax.axhline(0,lw=.7,color='black',alpha=.35)
    ax.set_title('(c) Signed tangential shear stress')
    ax.set_xlabel('Arclength (m)');ax.set_ylabel('tau_tn signed (PLAXIS unit)')
    ax.grid(alpha=.25)

    ax=axes[1,1]
    for key,color in zip(EXPECTED_COMPONENTS,['#2563eb','#059669','#a855f7']):
        ax.plot(df.Arclength_m,df[key],'.',ms=2,label=key,color=color)
    ax.set_title('(d) Recovered PLAXIS effective components')
    ax.set_xlabel('Arclength (m)');ax.set_ylabel('Effective stress (PLAXIS unit)')
    ax.legend(fontsize=8);ax.grid(alpha=.25)

    fig.suptitle(f'Stage B1 | Bishop effective stress | day {mesh.time:.4f} | NO FS',fontsize=13)
    fig.savefig(out/'B1_Bishop_effective_diagnostics.png',dpi=160)
    plt.close(fig)


def run_stage(field_npz, surface_csv, out, gp_csv=None, make_plot=True):
    """Read A2 NPZ and prescribed curve; write B1 quadrature CSV + report."""
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    mesh=BishopField(field_npz)
    curve=read_curve(surface_csv)
    df,report=run_b1(mesh,curve)
    report.update(input_field=str(field_npz),input_surface=str(surface_csv),
                  prescribed_surface_source='diagnostic/not optimized')
    csv=out/'B1_Bishop_quadrature_points.csv'
    df.to_csv(csv,index=False,float_format='%.12g')
    (out/'B1_Bishop_validation_report.json').write_text(
        json.dumps(report,indent=2),encoding='utf-8')
    if make_plot: plot_b1(mesh,curve,df,out,gp_csv)
    print(f'B1: {len(df)} GP points; Pt finite={report["pt_finite_quadrature_points"]}; '
          f'length={report["total_length_m"]:.3f} m')
    return csv
