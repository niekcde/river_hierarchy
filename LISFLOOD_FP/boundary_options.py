"""Outlet sensitivity options; retain discharge evaluation at the original port."""
import numpy as np
from diffusive_network import Network


def build_boundary(frame, n, dx, slope_factor=1., extension_m=0.):
    if not np.isfinite([n,dx,slope_factor,extension_m]).all() or slope_factor<=0 or extension_m<0:
        raise ValueError('Invalid boundary settings')
    f=frame.copy()
    sinks=set(f.v)-set(f.u)
    if len(sinks)!=1: raise ValueError('Expected one original outlet')
    sink=int(next(iter(sinks)))
    terminal=f[f.v.eq(sink)]
    if len(terminal)!=1: raise ValueError('Expected one terminal reach')
    downstream_area=None
    if extension_m:
        row=terminal.iloc[0].copy()
        slope=(row.head_us-row.head_ds)/row.link_length_m
        row['u']=sink;row['v']=int(max(f.u.max(),f.v.max()))+1
        row['head_us']=row.head_ds;row['head_ds']=row.head_us-slope*extension_m
        row['link_length_m']=extension_m
        if 'id_link' in row:row['id_link']=int(f.id_link.max())+1
        f.loc[len(f)]=row
    model=Network.from_frame(f,dx=dx,roughness=n)
    model.outlet_slope*=slope_factor
    gauge=model.node_ids.index(sink)
    if extension_m:
        face_ids=np.flatnonzero(model.faces.link.to_numpy()==len(frame))
        downstream_area=np.zeros(len(model.bed))
        np.add.at(downstream_area,model.a[face_ids],model.width[face_ids]*model.dx[face_ids]/2)
        np.add.at(downstream_area,model.b[face_ids],model.width[face_ids]*model.dx[face_ids]/2)
        first_face=int(face_ids[0])
    else:first_face=None
    return model,dict(gauge=gauge,first_extension_face=first_face,downstream_area=downstream_area,
                     original_outlet_node=sink,extension_m=extension_m,slope_factor=slope_factor)


def gauge_discharge(model, info, depth):
    """Flux reconstructed at the original port, including half-cell storage.

For an extension, Q_g = Q_first_extension_face + (W*dx/2)*dh_g/dt.
This measures at the original cross-section, not half a cell downstream.
The internal node has no external source term.
"""
    q,out=model.fluxes(depth)
    first=info['first_extension_face']
    if first is None:return out,q
    g=info['gauge']
    dh=(model.incidence.getrow(g)@q)[0]/model.storage_area[g]
    return float(q[first]+model.width[first]*model.dx[first]/2*dh),q
