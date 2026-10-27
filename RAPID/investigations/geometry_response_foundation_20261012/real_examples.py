"""Replace the toy teaching notebook with two audited current physical domains."""
from pathlib import Path
import sys,json,shutil,hashlib
import numpy as np
import pandas as pd
import networkx as nx
import nbformat as nb

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'RAPID/src'),str(ROOT/'RAPID/investigations/parent_child_predictor_exploration')]
from rapid_tools.current_review import build_current_analysis_frame
from rapid_tools.current_domains import load_domain_pair,allocation_tables
from features import describe,HYDRAULIC
OUT=ROOT/'RAPID/outputs/real_geometry_examples_20261012'
OUT.mkdir(exist_ok=True)
FOUNDATION=ROOT/'RAPID/outputs/geometry_response_foundation_20261012T120140'


def routes(frame):
    nodes,edges=allocation_tables(frame);out={}
    for r in edges.itertuples():out.setdefault(r.u,[]).append(r)
    source=int(nodes[nodes.in_degree.eq(0)].node.iloc[0]);sink=int(nodes[nodes.out_degree.eq(0)].node.iloc[0]);rows=[]
    def walk(u,ids,p,length,inverse_width):
        if len(rows)>100: raise ValueError('Explicit routes are restricted to small teaching examples')
        if u==sink:
            rows.append(dict(route=' → '.join(map(str,ids)),P_k=p,L_k_m=length,W_harmonic_k_m=length/inverse_width))
            return
        for e in out[u]:walk(e.v,ids+[e.id_link],p*e.p_outgoing,length+e.L_e_m,inverse_width+e.L_e_m/e.W_e_m)
    walk(source,[],1.,0.,0.)
    return pd.DataFrame(rows)


def descriptors(frame):
    raw,meta=describe(frame,False);assert meta['status']=='ok'
    m={k:float(v) for k,v in raw.items() if k not in HYDRAULIC}
    nodes,edges=allocation_tables(frame);rt=routes(frame);eff=[];weights=[]
    for u,g in edges.groupby('u'):
        if len(g)>1:
            p=g.p_outgoing.to_numpy();eff.append(np.exp(-p@np.log(p)));weights.append(g.r_edge.sum())
    m['entropic_branches_mean']=float(np.average(eff,weights=weights)) if eff else np.nan
    m['entropic_branches_sd']=float(np.sqrt(np.average((np.array(eff)-m['entropic_branches_mean'])**2,weights=weights))) if eff else np.nan
    m['path_harmonic_width_min']=rt.W_harmonic_k_m.min();m['path_harmonic_width_max']=rt.W_harmonic_k_m.max()
    rt=rt.sort_values('L_k_m');cum=rt.P_k.cumsum().to_numpy()
    quantiles=[float(rt.L_k_m.iloc[min(np.searchsorted(cum,p),len(rt)-1)]) for p in [.1,.5,.9]]
    m['route_length_quantiles']=' / '.join(f'{x:.3f}' for x in quantiles)+' m (P10/P50/P90)'
    if m['bifurcations']==0:
        for k in m:
            if k.startswith(('dominance_','effective_branches_','split_entropy_','split_exposure_','split_impact_','branch_length_contrast_')) and k.endswith(('_mean','_sd')):m[k]=np.nan
    if m['route_length_sd']==0:m['route_length_skew']=np.nan
    np.testing.assert_allclose(rt.P_k.sum(),1)
    np.testing.assert_allclose(rt.P_k@rt.L_k_m,m['route_length_mean'])
    np.testing.assert_allclose(edges.mean_length_contribution_m.sum(),m['route_length_mean'])
    np.testing.assert_allclose(rt.P_k@((rt.L_k_m-m['route_length_mean'])**2),m['route_length_sd']**2,rtol=1e-9,atol=1e-7)
    np.testing.assert_allclose(nodes.loc[nodes.out_degree.eq(0),'visit_probability'],1)
    return m,nodes,edges,rt


def context(full,source,sink,domain_length):
    nodes,edges=allocation_tables(full);r=nodes.set_index('node').visit_probability.to_dict();out={}
    for e in edges.itertuples():out.setdefault(e.u,[]).append(e)
    moment={int(nodes.iloc[0].node):0.}
    for u in nodes.node:
        for e in out.get(u,[]):moment[e.v]=moment.get(e.v,0.)+e.p_outgoing*(moment.get(u,0.)+r[u]*e.L_e_m)
    downstream={}
    for u in reversed(nodes.node.tolist()):downstream[u]=sum(e.p_outgoing*(e.L_e_m+downstream[e.v]) for e in out.get(u,[]))
    return dict(upstream_distance=moment[source]/r[source],downstream_distance=downstream[sink],
        network_entry_exposure=r[source],modified_length_fraction=domain_length/full.link_length_m.sum())


def main():
    allrows=build_current_analysis_frame(['02','46']);cases={};values=[];hashes={}
    for label,ex,sid in [('Simple','02','S013_unit_20'),('Small compound','46','CB006_bubble_101')]:
        row=allrows[(allrows.example_id==ex)&(allrows.state_id==sid)].iloc[0]
        parent,child,domain=load_domain_pair(row);tag=ex+'_'+sid
        row.to_json(OUT/(tag+'_record.json'))
        for state,frame in [('parent',parent),('child',child)]:
            m,n,e,rt=descriptors(frame)
            if state=='parent':m.update(context(pd.read_parquet(row.parent_parameters_parquet),domain['source'],domain['sink'],frame.link_length_m.sum()))
            for kind,data in [('reaches',frame),('nodes',n),('edges',e),('routes',rt)]:data.to_csv(OUT/f'{tag}_{state}_{kind}.csv',index=False)
            values += [dict(case=label,state=state,metric=k,value=v) for k,v in m.items()]
        cases[label]=dict(tag=tag,example=ex,state_id=sid,**domain)
        for p in [row.parent_parameters_parquet,row.parameters_parquet,domain['membership_path']]:hashes[p]=hashlib.sha256(Path(p).read_bytes()).hexdigest()
    values=pd.DataFrame(values);values.to_csv(OUT/'metric_values.csv',index=False)
    (OUT/'cases.json').write_text(json.dumps(cases,indent=2))
    (OUT/'provenance.json').write_text(json.dumps(dict(source_hashes=hashes,scope='Two real current collapse domains; no routing, calibration, or hydraulic resistance solving.',checks='Single source/sink; no external interior crossings; matching anchors; current endpoint equality; route/edge moments agree; sink allocation equals one.'),indent=2))
    cells=[]
    def md(s):cells.append(nb.v4.new_markdown_cell(s))
    def code(s):cells.append(nb.v4.new_code_cell(s))
    md(r'''# Geometric metrics on two real collapsed domains

This replaces the synthetic two-diamond example. The earlier toy notebook is archived with the output snapshot. Every geometric metric uses **current physical reach attributes** and a validated before/after domain. No routing or recalibration is performed.

Cases: **02 / S013_unit_20**, a simple two-reach unit; **46 / CB006_bubble_101**, a small compound with 11 reaches and five complete routes. Both children have one replacement reach. This compound is a real interconnected unit, not two separate diamonds joined by arbitrary centreline segments.''')
    code(f'''from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
ROOT=Path({str(ROOT)!r});OUT=Path({str(OUT)!r})
sys.path.insert(0,str(ROOT/'RAPID/src'))
from rapid_tools.current_review import parent_child_network_figure
cases=json.loads((OUT/'cases.json').read_text())
values=pd.read_csv(OUT/'metric_values.csv')
def table(case,state,kind):return pd.read_csv(OUT/f"{{cases[case]['tag']}}_{{state}}_{{kind}}.csv")
display(pd.DataFrame(cases).T[['example','state_id','source','sink','parent_links','child_links','domain_type']])''')
    md(r'''## Notation: reach, route and domain are different objects

- $e$ identifies **one physical reach/link**. $L_e$ is that reach's length in metres; $W_e$ (or $w_e$) is its measured/derived hydrographic width in metres. **$L_e$ is not a sum of path lengths.**
- $k$ identifies **one complete source-to-sink route**. Its length is $L_k=\sum_{e\in k}L_e$.
- $\sum_eL_e$ sums each unique reach in the chosen domain once. It is different from both a route length and mean travelled distance.
- $p_{u,e}=W_e/\sum_{j\in out(u)}W_j$ is a local outgoing allocation fraction. It is a width proxy, not observed discharge.
- $P_k$ is the probability of one complete route, obtained by multiplying its local split fractions.
- $r_u$ and $r_e$ sum the allocation arriving at a node or passing through a reach. They include all contributing route histories.

Width columns in the reach tables refer to the same hydrographic widths used by current routing preparation. Routing subdivisions are not treated as new physical reaches.''')
    md(r'''## The domain answer

**For the local collapse predictors here, yes: the metrics are calculated for the collapsed comparison domain, before and after the collapse, with matched physical inlet/outlet boundaries.** Simple: nodes 40→39, parent links 58/59 and child link 79. Compound: nodes 25→23, parent 11 links and child link 541.

The first foundation dataset instead computed whole-network summaries. Those remain valid **network-context** quantities, but are not local-domain measurements. This notebook now supplies real local-domain examples explicitly; it does not claim that the full 2,172-case local feature table has been rebuilt.

For a member edit inside a larger compound, distinguish the edited closed domain from the containing compound before/after. If the edit is not closed, a larger enclosing matched domain may be necessary and must be labelled. For these two full-unit collapses, the validated comparison domain is the collapsed unit itself.

Exceptions are clearly contextual: upstream/downstream distances, entry exposure, and modified-length fraction need the whole parent network as reference. Their values below describe the parent domain's position/exposure, not intrinsic width or length within the domain.''')
    for case in cases:
        md(f'## {case}: actual parent and child geometry\n\nThe map is zoomed to the validated collapse domain; surrounding reaches are grey. The following reach and route tables are the numerical inputs for all worked metric values.')
        code(f'''case={case!r}
record=pd.read_json(OUT/(cases[case]['tag']+'_record.json'),typ='series')
record['example_id']=str(record['example_id']).zfill(2)
parent_child_network_figure(record,zoom_to_domain=True).show()
for state in ['parent','child']:
    print(state)
    display(table(case,state,'edges').round(6))
    display(table(case,state,'routes').round(6))''')
    md(r'''## Confluences restore allocation; complete-route probabilities do not reset

The recursion is
$$r_v=\sum_{e:u\to v}r_up_{u,e},\qquad r_e=r_up_{u,e}.$$
Every incoming contribution is **added** before outgoing allocation is applied. For a 90:10 split that fully rejoins, the downstream node receives $0.9+0.1=1$. Repeating such splits does not reduce the allocation on a common downstream reach to $0.9^n$. That product describes only the particular route that chose the 90% branch every time.

In the real compound, node 26 receives contributions from nodes 27 and 28. Their sum restores its allocation to about 0.817238, equal to the share that entered the indirect branch at node 25. At the final confluence, node 23 also receives the direct branch (link 34), bringing the total to **1.0**.

There is no probability renormalization trick: all path histories are included. A confluence with an additional outgoing split accumulates first, then redistributes the sum.''')
    code('''nodes=table('Small compound','parent','nodes');edges=table('Small compound','parent','edges')
display(nodes.round(6))
display(edges[edges.v.isin([26,23])][['id_link','u','v','p_outgoing','r_edge']].round(6))
for node in [26,23]:
    incoming=edges.loc[edges.v.eq(node),'r_edge'].sum()
    expected=nodes.loc[nodes.node.eq(node),'visit_probability'].iloc[0]
    print(f'Node {node}: sum incoming = {incoming:.9f}; node allocation = {expected:.9f}')
    assert np.isclose(incoming,expected)
assert np.isclose(nodes.loc[nodes.node.eq(23),'visit_probability'].iloc[0],1.)''')
    md(r'''## Repeated shared reaches: why expected length is different from unweighted summation

$$\mu_L=\sum_kP_kL_k=\sum_er_eL_e.$$
Shared reaches may appear in several route descriptions; that is correct. Their contributions are weighted by the route probabilities and add to their total allocation $r_e$. A reach traversed by all routes has $r_e=1$, not a multiplicity equal to the number of routes. Summing complete-path widths as if all routes were independent parallel channels lacks this property.

The next check computes each real mean independently from the explicit route list and the edge allocations. Production graphs use DAG recursions; enumeration is only used here because these examples have two and five routes.''')
    code('''for case in cases:
    rt=table(case,'parent','routes');ed=table(case,'parent','edges')
    a=(rt.P_k*rt.L_k_m).sum();b=ed.mean_length_contribution_m.sum()
    print(case, 'route probabilities sum =',rt.P_k.sum(),'mean from routes =',a,'m; from reaches =',b,'m')
    assert np.isclose(a,b) and np.isclose(rt.P_k.sum(),1.)''')
    md(r'''## Each metric: real values, formula and interpretation

All lengths are metres; area/variance terms are square metres; CV, entropy, counts, allocation and elongation are dimensionless; bifurcation density is per metre. Each table shows parent, child and change where defined. The child has one route, so route-length SD/CV are zero. A mean conditional on encountering a split is **undefined** when there are no splits; those cells are NaN, rather than falsely implying zero dominance or zero branch width.

Route skewness is the standard third standardized central moment $E[(L-\mu_L)^3]/\sigma_L^3$, not an invented exponent. It is undefined for a single deterministic route. [NIST definition](https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/skewness.pdf).''')
    catalog=pd.read_csv(FOUNDATION/'geometry_dictionary.csv')
    notes={
        'route_length_mean':'For the simple parent, multiply the two printed reach lengths by their outgoing width fractions. For the compound, use all five P_k values, or the r_e L_e contributions. This is not sum of unique lengths.',
        'route_length_sd':'Spread between complete allocated routes, not spread among individual link lengths. Complete collapse to one route removes this geometric spread.',
        'width_mean':'Sum L_e W_e divided by sum L_e over unique physical reaches. This differs from flow-weighted width.',
        'width_harmonic':'Sum L_e divided by sum L_e/W_e. Narrow reaches matter strongly. This aggregate is not equivalent nonlinear network resistance.',
        'branch_length_contrast_mean':'At each split, compare expected remaining distance through each outgoing branch to the same domain sink; weight local contrasts by node visits.',
        'branch_length_variance_sum':'Equals route-length SD squared. This is a redundant alternative representation, not another independent predictor.',
        'cycle_rank':'E−V+C counts undirected independent loops. Parallel physical reaches are separate edges. The flow graph remains directed and acyclic.',
        'junction_excess':'Sum max(outdegree−1,0). Equals cycle rank for these closed single-sink graphs.',
        'log1p_paths':'The simple parent has two complete routes: ln(3). The compound has five: ln(6). A single child route gives ln(2), not zero.',
        'split_exposure_mean':'Conditional visit probability averaged over split nodes. Confluences have already accumulated all incoming contributions.',
        'network_entry_exposure':'Whole-network allocation arriving at the domain inlet. Within-domain calculations reset entry allocation to one, so this separate factor is needed for network exposure.',
        'modified_length_fraction':'For these complete unit collapses, sum of unique parent-domain reach lengths divided by unique whole-parent network length. Not a sum of path lengths.',
        'route_length_quantiles':'Inverse discrete CDF of the listed route probabilities. Values are P10/P50/P90; there is no interpolation between distinct route lengths.',
        'path_harmonic_width_min':'Minimum of the listed real route harmonic widths. An extreme path width is not an allocation fraction.',
        'path_harmonic_width_max':'Maximum of the listed real route harmonic widths. Never add these overlapping path widths as independent channel capacities.',
    }
    for r in catalog.itertuples():
        md(f'### {r.name}\n\n**Family:** {r.family}. **Equation:** `{r.equation}`.\n\n{r.description}\n\n{notes.get(r.name,r.redundancy_note)}')
        code(f'''metric={r.name!r}
shown=values[values.metric.eq(metric)].pivot(index='case',columns='state',values='value').reindex(cases)
for state in ['parent','child']:
    if state not in shown:shown[state]=np.nan
if metric!='route_length_quantiles':
    shown=shown.apply(pd.to_numeric,errors='coerce');shown['child_minus_parent']=shown.child-shown.parent
display(shown[['parent','child']+(['child_minus_parent'] if 'child_minus_parent' in shown else [])])''')
    md(r'''## Resistance stays distinct

The existing `resistance_log = -log10(K_child/K_parent)` is not equivalent to mean/harmonic width. The matched domain is now identified and checked for these examples, but this notebook does **not** promote an older stored K value as a newly calculated resistance under current routing parameters. Current geometric examples use only connectivity, length and width. Depth/roughness policy must be stated when adding resistance. No hydraulic resistance solver or routing run has been launched here.

## Scope of this update

The review map now highlights validated domain links and honors `ZOOM_TO_DOMAIN=True`. These two examples compute local before/after descriptors from current attributes. The original whole-network feature snapshot remains unchanged. Broader local feature extraction still needs per-state domain validation, especially edited members requiring enclosing domains.''')
    p=ROOT/'RAPID/notebooks/geometry_metric_worked_examples_20261012.ipynb'
    backup=OUT/'geometry_metric_worked_examples_toy_archive.ipynb'
    if p.exists() and not backup.exists():shutil.copy2(p,backup)
    nb.write(nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}}),p)
    print(p)


if __name__=='__main__':main()
