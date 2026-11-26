"""Stage C1: geometry-based entry/exit sampling and admissible search domain.

Material masks and 22 m depth cap are provisional research assumptions.
Supports independent entry_samples and exit_samples from JSON.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Polygon, mapping

COLORS = {0: '#DDBA82', 4: '#78818A', 5: '#78B4A1', 8: '#B8AA79'}
MATERIAL_NAMES_DEFAULT = {'0':'Sandy_Clay_BBM', '4':'HardRock_MCC',
                          '5':'Silty_Clay_BBM', '8':'Silty_Clay_BBM'}


def open_bishop_a2(path):
    with np.load(path, allow_pickle=False) as z:
        if str(z['stress_measure'].item()) != 'Bishop_effective_PLAXIS':
            raise ValueError('Stage C1 requires A2 Bishop-effective NPZ')
        if list(z['stress_components'].astype(str)) != ['Sigxx_E','Sigyy_E','Sigxy']:
            raise ValueError('Unexpected A2 stress component metadata')
        out = {k: z[k].copy() for k in ['element_id','material_index','corner_nodes','corner_xy','gauss_xy']}
        out['time'] = float(z['time'].item())
    ids, mats, nd, xy = (out[k] for k in ['element_id','material_index','corner_nodes','corner_xy'])
    if (xy.shape!=(len(ids),3,2) or nd.shape!=(len(ids),3)
        or np.any(~np.isfinite(xy)) or len(ids)!=len(np.unique(ids))):
        raise ValueError('Malformed element geometry')
    return out


def unique_boundary_edges(corner_nodes, corner_xy, element_selector=None):
    if element_selector is None:
        element_selector = np.ones(len(corner_nodes),dtype=bool)
    n=corner_nodes[element_selector]
    x=corner_xy[element_selector]
    en=np.concatenate([n[:,[0,1]],n[:,[1,2]],n[:,[2,0]]])
    ex=np.concatenate([x[:,[0,1]],x[:,[1,2]],x[:,[2,0]]])
    key=np.sort(en,axis=1)
    _, idx, inv, count=np.unique(key,axis=0,return_index=True,return_inverse=True,return_counts=True)
    outer=count[inv]==1
    if np.any(count>2):
        raise ValueError('Non-manifold mesh: an edge shared by >2 triangles')
    return en[outer], ex[outer]


def ground_y(x, boundary_edges):
    """Topmost intersection of external ACTIVE boundary with vertical x line.

    Valid for the locally x-monotonic ground geometry in the submitted mesh.
    We do not extrapolate beyond available boundary coordinates.
    """
    x = np.atleast_1d(np.asarray(x,float))
    p0, p1 = boundary_edges[:,0,:], boundary_edges[:,1,:]
    a,b=p0[:,0],p1[:,0]
    result=np.full(len(x),np.nan)
    for i,xi in enumerate(x):
        mask=(np.minimum(a,b)-1e-8<=xi)&(xi<=np.maximum(a,b)+1e-8)
        if not mask.any():continue
        k0,k1=p0[mask],p1[mask]
        dx=k1[:,0]-k0[:,0]
        y=np.where(np.abs(dx)>1e-12,
                   k0[:,1]+(xi-k0[:,0])*(k1[:,1]-k0[:,1])/np.where(np.abs(dx)>1e-12,dx,1),
                   np.maximum(k0[:,1],k1[:,1]))
        result[i]=float(np.max(y))
    if np.any(~np.isfinite(result)):
        raise ValueError('Some surface x coordinates not covered by active mesh')
    return result


def surface_polyline(edges,xmin,xmax):
    allx=np.unique(np.r_[xmin,xmax,edges[:,:,0].ravel()]);x=allx[(allx>=xmin)&(allx<=xmax)]
    return np.column_stack((x,ground_y(x,edges)))


def parse_connectivity_mesh(path):
    """Optional full-mesh audit; uses 15-row blocks and comma-decimal coords."""
    df=pd.read_csv(path,sep='\t',dtype=str).fillna('')
    df.columns=df.columns.str.strip()
    for c in ('Soil element','Node','Local number','X [m]','Y [m]'):
        if c not in df:raise ValueError('Connectivity missing '+c)
    n=len(df)
    if n%15:raise ValueError('Expected 15 rows per element in connectivity.txt')
    labels=df['Soil element'].str.strip().to_numpy().reshape(-1,15)
    local=df['Local number'].str.strip().to_numpy(dtype=int).reshape(-1,15)
    if not np.all(local==np.arange(1,16)):raise ValueError('Unexpected T15 local numbering')
    ids=[]
    for s in labels[:,0]:
        m=re.search(r'El\.\s*(\d+)',s)
        if not m:raise ValueError('Cannot parse element label '+repr(s))
        ids.append(int(m.group(1)))
    names=np.char.replace(labels[:,1].astype(str),r'\_','_')
    return np.asarray(ids,int),names


def shape_polygons(mesh, allowed):
    xy=mesh['corner_xy'][np.isin(mesh['material_index'],allowed)]
    if not len(xy):raise ValueError('Allowed material indices have no active mesh elements')
    rings=np.concatenate([xy,xy[:,:1]],axis=1)
    polygons=shapely.polygons(rings)
    if not np.all(shapely.is_valid(polygons)):
        raise ValueError('Invalid triangular mesh geometry')
    soil=shapely.union_all(polygons,grid_size=1e-8)
    if soil.is_empty or not soil.is_valid:raise ValueError('Invalid allowed material union')
    return soil


def depth_strip(surface,depth):
    return Polygon(np.vstack((surface,
         np.column_stack((surface[::-1,0],surface[::-1,1]-depth)))))


def polygons_in(geom):
    if geom.is_empty:return []
    if geom.geom_type=='Polygon':return [geom]
    if hasattr(geom,'geoms'):
        return [p for g in geom.geoms for p in polygons_in(g)]
    return []


def draw_polygons(ax, geom, **kwargs):
    for poly in polygons_in(geom):
        ext=np.asarray(poly.exterior.coords)
        ax.fill(ext[:,0],ext[:,1],**kwargs)
        for hole in poly.interiors:
            hp=np.asarray(hole.coords)
            ax.fill(hp[:,0],hp[:,1],color='white',zorder=3)


def run(args):
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    entry=list(map(float,cfg['entry_x_range_m']))
    exit=list(map(float,cfg['exit_x_range_m']))
    xview=list(map(float,cfg.get('plot_x_range_m',[max(0,entry[0]-30),exit[1]+30])))
    depth=float(cfg.get('maximum_vertical_depth_below_ground_m',22))
    allowed=list(map(int,cfg['allowed_material_indices']))
    if not entry[0]<entry[1]<exit[0]<exit[1]:
        raise ValueError('Must have ordered, non-overlapping entry/exit x intervals')
    if depth<=0 or xview[0]>=entry[0] or xview[1]<=exit[1]:
        raise ValueError('Invalid depth or plot x-range')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    mesh=open_bishop_a2(args.a2)
    nodes, edges=unique_boundary_edges(mesh['corner_nodes'],mesh['corner_xy'])
    surface=surface_polyline(edges,xview[0],xview[1])
    roi_surface=surface_polyline(edges,entry[0],exit[1])
    strip=depth_strip(roi_surface,depth)
    allowed_soil=shape_polygons(mesh,allowed)
    eligible=shapely.intersection(allowed_soil,strip,grid_size=1e-8)
    if eligible.is_empty or not eligible.is_valid:
        raise ValueError('Search domain empty or invalid')
    # Independent sampling densities. Legacy samples_per_zone remains supported.
    n_entry = int(cfg.get('entry_samples', cfg.get('samples_per_zone', 21)))
    n_exit = int(cfg.get('exit_samples', cfg.get('samples_per_zone', 21)))
    if n_entry < 2 or n_exit < 2:
        raise ValueError('entry_samples and exit_samples must both be >= 2')
    entry_x=np.linspace(*entry,n_entry)
    exit_x=np.linspace(*exit,n_exit)
    entry_y=ground_y(entry_x,edges);exit_y=ground_y(exit_x,edges)
    # Material inclusion 5 cm below surface, positive clearance to avoid edge ambiguity.
    edge_clearance=float(cfg.get('surface_probe_depth_m',0.05))
    entry_point=shapely.points(entry_x,entry_y-edge_clearance)
    exit_point=shapely.points(exit_x,exit_y-edge_clearance)
    entry_inside=np.asarray(shapely.covers(allowed_soil,entry_point))
    exit_inside=np.asarray(shapely.covers(allowed_soil,exit_point))
    if not entry_inside.all() or not exit_inside.all():
        raise ValueError('Some proposed entry/exit points are not on surface of allowed materials; '
                         f'entry valid={entry_inside.sum()}/{len(entry_inside)}, '
                         f'exit valid={exit_inside.sum()}/{len(exit_inside)}')

    # Full connectivity audit is optional but confirms elements inactive in this phase.
    mesh_audit={'active_elements':int(len(mesh['element_id']))}
    if args.connectivity:
        full_id,full_names=parse_connectivity_mesh(args.connectivity)
        missing=~np.isin(full_id,mesh['element_id'])
        excluded_names=Counter(full_names[missing].tolist())
        mesh_audit.update(full_elements=int(len(full_id)),
                          not_active_in_snapshot=int(missing.sum()),
                          not_active_materials=dict(excluded_names))

    # Diagnostic old B1 prescribed curve: it is not required to meet surface.
    old_curve=None
    surface_offsets=None
    if args.b1_curve:
        old_curve=pd.read_csv(args.b1_curve)[['X','Y']].to_numpy(dtype=float)
        if len(old_curve)>1:
            ytop=ground_y(old_curve[[0,-1],0],edges)
            surface_offsets={
              'first_point_depth_below_surface_m':float(ytop[0]-old_curve[0,1]),
              'last_point_depth_below_surface_m':float(ytop[1]-old_curve[-1,1]),
              'old_curve_not_a_verified_entry_exit_surface':True}

    # Export search geometry to GeoJSON for exact material-aware candidate checks in C2.
    geo={'type':'FeatureCollection','features':[
        {'type':'Feature','properties':{'role':'admissible_slip_region',
          'materials':allowed,'definition':'active allowed-material union intersected with vertical ground-depth band',
          'provisional':True},'geometry':mapping(eligible)},
        {'type':'Feature','properties':{'role':'allowed_material_union',
          'materials':allowed},'geometry':mapping(allowed_soil)}]}
    (out/'C1_admissible_domain.geojson').write_text(json.dumps(geo),encoding='utf-8')
    pd.DataFrame(surface,columns=['X','Y_ground']).to_csv(out/'C1_ground_surface.csv',index=False)
    candidate=pd.concat([
       pd.DataFrame({'zone':'entry','X':entry_x,'Y_ground':entry_y,'probe_inside_allowed_material':entry_inside}),
       pd.DataFrame({'zone':'exit','X':exit_x,'Y_ground':exit_y,'probe_inside_allowed_material':exit_inside})],ignore_index=True)
    candidate.to_csv(out/'C1_entry_exit_candidates.csv',index=False)
    config_resolved=dict(cfg)
    config_resolved.update({'A2_input':str(args.a2),'stress_measure':'Bishop_effective_PLAXIS',
          'time_day':mesh['time'],'search_geometry_file':'C1_admissible_domain.geojson',
          'ground_surface_file':'C1_ground_surface.csv','entry_exit_candidates_file':'C1_entry_exit_candidates.csv',
          'entry_samples':n_entry,'exit_samples':n_exit,
          'stage':'C1; provisional zones require user review; no FS or critical surface computed'})
    (out/'C1_config_resolved.json').write_text(json.dumps(config_resolved,indent=2),encoding='utf-8')
    report={
      'stage':'C1 Entry–Exit search domain only (NO FS)','day':mesh['time'],
      'entry_x_range_m':entry,'exit_x_range_m':exit,
      'entry_samples':n_entry,'exit_samples':n_exit,
      'candidate_grid_entry_exit_pairs':n_entry*n_exit,
      'entry_y_at_interval_ends_m':ground_y(entry,edges).tolist(),
      'exit_y_at_interval_ends_m':ground_y(exit,edges).tolist(),
      'surface_y_at_entry_center_m':float(ground_y([(entry[0]+entry[1])/2],edges)[0]),
      'surface_y_at_exit_center_m':float(ground_y([(exit[0]+exit[1])/2],edges)[0]),
      'allowed_materials':allowed,
      'candidate_maximum_depth_m':depth,
      'eligible_region_area_m2':float(eligible.area),
      'eligible_region_polygon_count':len(polygons_in(eligible)),
      'entry_probe_all_inside':bool(entry_inside.all()),
      'exit_probe_all_inside':bool(exit_inside.all()),
      'active_mesh_external_boundary_edges':int(len(edges)),
      'mesh_audit':mesh_audit,
      'diagnostic_old_B1_curve_offsets':surface_offsets,
      'surface_is_active_mesh_upper_envelope':True,
      'cautions':[
        'Entry and exit intervals are provisional, not verified landslide boundaries.',
        'The admissible domain excludes materials not selected and cannot override mesh geometry.',
        'Circular arcs must also be checked against surface and self-intersection and stay in admissible polygon.',
        'The 22 m band limits candidate depths geometrically; it does not assert geological failure depth.',
        'B2 signed-shear FS objective remains conditional until stress/force directional admissibility is validated.',
        'No factor of safety or critical surface has been computed.']}
    (out/'C1_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

    # Produce a single explanatory figure in the active mesh coordinate system.
    fig,ax=plt.subplots(figsize=(14,7.7),constrained_layout=True)
    tri=mesh['corner_xy']; mats=mesh['material_index']
    visibility=np.any((tri[:,:,0]>=xview[0]-6)&(tri[:,:,0]<=xview[1]+6),axis=1)
    pc=PolyCollection(tri[visibility],facecolors=[COLORS.get(int(m),'#bdc3c7') for m in mats[visibility]],
          edgecolors='white',linewidths=.045,alpha=.9,zorder=1)
    ax.add_collection(pc)
    # Visually distinguish boundary of the admissible search polygon.
    draw_polygons(ax,eligible,color='#8B5CF6',alpha=.24,zorder=3)
    for poly in polygons_in(eligible):
        boundary=np.asarray(poly.exterior.coords)
        ax.plot(boundary[:,0],boundary[:,1],color='#6D28D9',lw=1.05,ls='--',alpha=.85,zorder=4)
    ax.plot(surface[:,0],surface[:,1],color='#1F2937',lw=2.3,zorder=5)
    ax.plot(entry_x,entry_y,color='#C62828',lw=5,solid_capstyle='round',zorder=8)
    ax.plot(exit_x,exit_y,color='#1565C0',lw=5,solid_capstyle='round',zorder=8)
    if old_curve is not None:
        ax.plot(old_curve[:,0],old_curve[:,1],color='#D97706',lw=1.8,ls=':',zorder=6)
        ax.scatter(old_curve[[0,-1],0],old_curve[[0,-1],1],marker='x',s=55,color='#D97706',zorder=7)
    # Mark high incremental shear strain locations (context only, not used to infer endpoints).
    hi_n=0
    if args.gp_csv:
        df=pd.read_csv(args.gp_csv,usecols=['X','Y','Gamma_incremental'])
        dv=df['Gamma_incremental'].to_numpy(dtype=float)
        q=float(cfg.get('high_strain_percentile',99.5))
        if not np.isfinite(dv).all():raise ValueError('Nonfinite gamma in optional GP CSV')
        cutoff=np.percentile(dv,q)
        hot=df[df.Gamma_incremental>=cutoff]
        hot=hot[(hot.X.between(xview[0],xview[1]))]
        hi_n=len(hot)
        ax.scatter(hot.X,hot.Y,s=2.1,c='#E11D48',alpha=.45,zorder=5,
                   rasterized=True)
    ey0=entry_y.mean();xy0=entry_x.mean(); yy=exit_y.mean();xx=exit_x.mean()
    ax.annotate('ENTRY RANGE\n(provisional)',xy=(xy0,ey0),xytext=(xy0-16,ey0+11),
       ha='center',fontsize=10,fontweight='bold',color='#B91C1C',
       arrowprops={'arrowstyle':'->','color':'#B91C1C','lw':1.3})
    ax.annotate('EXIT RANGE\n(provisional)',xy=(xx,yy),xytext=(xx+9,yy+14),
       ha='center',fontsize=10,fontweight='bold',color='#1D4ED8',
       arrowprops={'arrowstyle':'->','color':'#1D4ED8','lw':1.3})
    ax.set_xlim(*xview)
    # Use relevant vertical extent; keep rock visible for stratigraphic context.
    ymin=max(float(np.min(surface[:,1])-34),float(np.min(tri[:,:,1]))-3)
    ymax=float(np.max(surface[:,1])+15)
    ax.set_ylim(ymin,ymax)
    ax.set_aspect('equal',adjustable='box')
    ax.set_xlabel('PLAXIS X (m)',fontsize=12)
    ax.set_ylabel('PLAXIS Y (m)',fontsize=12)
    ax.set_title('Stage C1 | Entry–Exit and Admissible Search Domain\n'
                 f'PLAXIS Bishop A2 mesh · day {mesh["time"]:.3f} · NO FS SEARCH',
                 loc='left',fontsize=13,fontweight='bold')
    ax.grid(alpha=.14,zorder=0)
    handles=[Patch(facecolor=COLORS[i],label=f'Index {i}: {MATERIAL_NAMES_DEFAULT[str(i)]}')
             for i in sorted(np.unique(mats)) if str(i) in MATERIAL_NAMES_DEFAULT]
    handles += [Patch(facecolor='#8B5CF6',alpha=.36,edgecolor='#6D28D9',
                label=f'Admissible area (materials {allowed}; depth ≤ {depth:g} m)'),
                plt.Line2D([],[],color='#C62828',lw=4,label='Entry zone'),
                plt.Line2D([],[],color='#1565C0',lw=4,label='Exit zone')]
    if old_curve is not None:
        handles.append(plt.Line2D([],[],color='#D97706',ls=':',lw=2,
                                  label='Old B1 interior diagnostic curve'))
    if args.gp_csv:
        handles.append(plt.Line2D([],[],marker='.',color='#E11D48',lw=0,
                                  label=f'Top {100-float(cfg.get("high_strain_percentile",99.5)):g}% incremental shear strain'))
    ax.legend(handles=handles,loc='lower left',fontsize=8.3,ncol=2,framealpha=.94)
    fig.savefig(out/'C1_search_domain.png',dpi=200)
    plt.close(fig)

    print('=== STAGE C1: ENTRY–EXIT SEARCH DOMAIN (NO FS) ===')
    print(f'Snapshot day {mesh["time"]:.6f} | active elements {len(mesh["element_id"]):,}')
    print(f'Full-mesh audit: {mesh_audit}')
    print(f'Entry x = {entry} m, ground Y endpoints = {report["entry_y_at_interval_ends_m"]}')
    print(f'Exit  x = {exit} m, ground Y endpoints = {report["exit_y_at_interval_ends_m"]}')
    print(f'Sampling: {n_entry} entry x {n_exit} exit = {n_entry*n_exit} endpoint pairs')
    print(f'Allowed material indices: {allowed}; max depth below local ground: {depth:.1f} m')
    print(f'Admissible area {eligible.area:.3f} m^2 ({len(polygons_in(eligible))} polygon(s)); valid={eligible.is_valid}')
    print(f'Entry/exit surface probes inside allowed materials: {entry_inside.all()} / {exit_inside.all()}')
    if surface_offsets:
        print('Old B1 curve endpoint depths below ground:',surface_offsets)
    print(f'High strain plotted: {hi_n} points')
    print('Saved:',out.resolve())
    print('Provisional search zones ONLY. No circular search or factor of safety computed.')


