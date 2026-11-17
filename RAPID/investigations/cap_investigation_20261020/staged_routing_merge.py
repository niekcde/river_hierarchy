"""Attribute lineage for staged slopes/geometry; never edits source files."""
import numpy as np
import pandas as pd

def assemble(source,parent,slopes,geometry_actions=()):
    assert source.id_link.is_unique and parent.id_link.is_unique and slopes.id_link.is_unique
    si=source.set_index('id_link');bi=parent.set_index('id_link');s=slopes.set_index('id_link')
    f=si.reindex(s.index).copy();lineage=[]
    declared={int(i) for action in geometry_actions if action.get('action')=='accepted_reference_reach_restoration' for i in action['restored_parent_links']}
    for link in f.index[~f.index.isin(si.index)]:
        row=s.loc[link]
        if pd.notna(row.parent_link):
            pid=int(row.parent_link)
            if row.role!='exterior':raise ValueError('Unexpected parent correspondence for selected new link')
            reason='explicit exterior parent correspondence'
        elif int(link)<0 and -int(link) in declared:
            pid=-int(link);reason='accepted restoration record; selected-side link'
        else:raise ValueError(f'Missing hydraulic attribute lineage for new link {link}')
        p=bi.loc[pid]
        # Restored link geometry and orientation must match the actual parent.
        np.testing.assert_allclose(row[['u','v','link_length_m']].astype(float),p[['u','v','link_length_m']].astype(float),rtol=1e-10,atol=1e-7)
        f.loc[link]=p.reindex(f.columns)
        lineage.append(dict(child_link=int(link),parent_link=pid,width=float(p.width),attribute_source=reason,role=row.role))
    common=s.index.intersection(si.index)
    lengthchanges=int((~np.isclose(si.loc[common,'link_length_m'],s.loc[common,'link_length_m'],rtol=1e-10,atol=1e-7)).sum())
    nodechanges=int(((si.loc[common,'u']!=s.loc[common,'u'])|(si.loc[common,'v']!=s.loc[common,'v'])).sum())
    for k in ['u','v','link_length_m','head_us','head_ds','slope']:f[k]=s[k]
    f['rapid_effective_length_m']=f.link_length_m/f['count']
    assert np.isfinite(f[['width','depth_m','count','rapid_effective_length_m']]).all().all()
    assert (f[['width','depth_m','count','rapid_effective_length_m']]>0).all().all()
    return f.reset_index(),lineage,lengthchanges,nodechanges
