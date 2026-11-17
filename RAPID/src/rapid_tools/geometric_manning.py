"""Whole-network Manning geometry factor; no depth, roughness or simulated flow.

G_M=sum L B**(2/5) f**(3/5) S**(-3/10). Allocation f is recomputed
from connectivity and local widths, including accumulation at confluences.
"""
import numpy as np
from .current_domains import allocation_tables


def manning_geometry_edges(frame):
    f=frame.copy()
    if 'id_us_node' not in f:f['id_us_node']=f['u']
    if 'id_ds_node' not in f:f['id_ds_node']=f['v']
    f=f[['id_link','id_us_node','id_ds_node','link_length_m','width','slope']].copy()
    if f.empty or f.id_link.duplicated().any():raise ValueError('Require nonempty unique physical links')
    if not np.isfinite(f[['link_length_m','width','slope']]).all().all() or not (f[['link_length_m','width','slope']]>0).all().all():raise ValueError('Geometry must be finite and positive')
    nodes,edges=allocation_tables(f)
    sinks=nodes[nodes.out_degree==0]
    if len(sinks)!=1 or not np.isclose(sinks.visit_probability.iloc[0],1.,rtol=1e-10):raise ValueError('Require a connected single-outlet network')
    fractions=edges.set_index('id_link').r_edge.reindex(f.id_link).to_numpy()
    if not ((fractions>0)&(fractions<=1+1e-10)).all():raise ValueError('Invalid width-derived allocation')
    f['geometry_flow_fraction']=fractions
    f['G_manning_contribution']=f.link_length_m*f.width**.4*fractions**.6*f.slope**(-.3)
    return f


def manning_geometry_factor(frame):
    return float(manning_geometry_edges(frame).G_manning_contribution.sum())
