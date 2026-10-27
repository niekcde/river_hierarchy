"""Read-only audit of published 46/CB006; write isolated review artifacts."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat as nb
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src')]
from rapid_tools.current_review import build_current_analysis_frame,load_pair
from rapid_tools.current_domains import load_domain_pair
from rapid_tools.cunge import route,topology
OUT=ROOT/'RAPID/outputs/cb006_hydrograph_audit_20261012'
OUT.mkdir(exist_ok=True)
r=build_current_analysis_frame(['46']).query('state_id == "CB006_bubble_101"').iloc[0]
p,c,domain=load_domain_pair(r)
pw,cw,forcing=load_pair(r)
full=[pd.read_parquet(r[k]) for k in ['parent_parameters_parquet','parameters_parquet']]
for key,f in zip(['parent_parameters_parquet','parameters_parquet'],full):
 state=Path(r[key]).parents[3]
 current=pd.read_parquet(state/'hydraulic_context/hydraulic_frame.parquet').set_index('id_link')
 indexed=f.set_index('id_link')
 for column in ['link_length_m','width','depth_m','head_us','head_ds','slope']:
  np.testing.assert_array_equal(indexed[column],current.loc[indexed.index,column])
 manifest=json.loads((state/'hydraulic_context/hydraulic_context_manifest.json').read_text())
 geometry=gpd.read_file(manifest['input_paths']['directed_links']).set_index('id_link')
 np.testing.assert_array_equal(indexed.width,geometry.loc[indexed.index,'wid_adj_routing'])
for f in full:
 np.testing.assert_allclose(f.slope,(f.head_us-f.head_ds)/f.link_length_m,rtol=1e-10)
 np.testing.assert_allclose(f.rapid_celerity_mps_raw,(5/3)*f.depth_m**(2/3)*np.sqrt(f.slope)/f.n_manning)
common=full[0].set_index('id_link').index.intersection(full[1].set_index('id_link').index)
cols=['link_length_m','width','depth_m','slope','head_us','head_ds','n_manning','rapid_k','rapid_x','count']
np.testing.assert_allclose(full[0].set_index('id_link').loc[common,cols],full[1].set_index('id_link').loc[common,cols],rtol=0,atol=0)
for name,f in [('parent',p),('child',c)]:f.to_csv(OUT/f'{name}_domain_parameters.csv',index=False)
def moments(f):
 order,edges,root,end=topology(f.reset_index(drop=True));f=f.reset_index(drop=True)
 # Zero-frequency derivatives of the exact routing recurrence, not event moments.
 mass={root:1.};m1={root:0.};m2={root:0.}
 for u in order:
  for i in edges.get(u,[]):
   e=f.iloc[i];mean=e['count']*e.rapid_k;var=e['count']*e.rapid_k**2*(1-2*e.rapid_x)
   v=e.v;w=e.split_fraction
   mass[v]=mass.get(v,0)+w*mass[u]
   m1[v]=m1.get(v,0)+w*(m1[u]+mass[u]*mean)
   m2[v]=m2.get(v,0)+w*(m2[u]+2*mean*m1[u]+mass[u]*(var+mean**2))
 return dict(gain=mass[end],mean_minutes=m1[end]/mass[end]/60,sd_minutes=np.sqrt(m2[end]/mass[end]-(m1[end]/mass[end])**2)/60)
rows=[]
for scope,frames in [('local',[p,c]),('whole_network',full)]:
 for label,f in zip(['parent','child'],frames): rows.append(dict(scope=scope,state=label,**moments(f)))
metrics=pd.DataFrame(rows);metrics.to_csv(OUT/'routing_transfer_moments.csv',index=False)
# Explicit local routes to avoid comparing child length with a sum of parallel lengths.
by={}
for e in p.itertuples():by.setdefault(e.u,[]).append(e)
routes=[]
def walk(u,ids,prob,L,T):
 if u==domain['sink']:
  routes.append(dict(links=' → '.join(map(str,ids)),probability=prob,length_m=L,effective_slope=(c.head_us.iloc[0]-c.head_ds.iloc[0])/L,advective_minutes=T/60));return
 for e in by[u]:walk(e.v,ids+[e.id_link],prob*e.split_fraction,L+e.link_length_m,T+e.link_length_m/e.rapid_celerity_mps)
walk(domain['source'],[],1.,0.,0.)
pd.DataFrame(routes).to_csv(OUT/'parent_routes.csv',index=False)
fig,ax=plt.subplots(2,2,figsize=(12,7))
t0=pd.Timestamp(r.parent_peak_utc)
for name,w,color in [('Parent',pw,'#2563eb'),('Collapsed',cw,'#dc2626')]:
 t=(w.time_utc-t0).dt.total_seconds()/3600;q=w.q_outlet_cms
 for a,window in [(ax[0,0],240),(ax[0,1],24)]:
  sel=t.abs()<=window;a.plot(t[sel],q[sel],label=name,color=color)
 sel=t.abs()<=.5;ax[1,0].plot(t[sel]*60,(q-q.max())[sel],'.-',label=name,color=color)
ax[0,0].set(title='Outlet event context',xlabel='Hours from parent peak',ylabel='Discharge (m³/s)')
ax[0,1].set(title='Crest and limbs',xlabel='Hours from parent peak',ylabel='Discharge (m³/s)')
ax[1,0].set(title='Each curve minus its own maximum',xlabel='Minutes from parent peak',ylabel='Q − own maximum (m³/s)')
t=(pw.time_utc-t0).dt.total_seconds()/3600;sel=t.abs()<=120
ax[1,1].plot(t[sel],(cw.q_outlet_cms-pw.q_outlet_cms)[sel],color='#7c3aed');ax[1,1].axhline(0,color='gray',lw=.8)
ax[1,1].set(title='Shape change: child minus parent',xlabel='Hours from parent peak',ylabel='ΔQ (m³/s)')
for a in ax.flat:
 a.grid(alpha=.2)
 if a!=ax[1,1]:a.legend()
fig.tight_layout();fig.savefig(OUT/'hydrograph_comparison.png',dpi=170);plt.close(fig)
print(metrics.to_string(index=False),flush=True)
print(pd.DataFrame(routes).to_string(index=False),flush=True)
# Independent DAG implementation, and a halved-clock sensitivity without refitting.
checks=[]
for dt in [5.,2.5]:
 for label,f,w in zip(['parent','child'],full,[pw,cw]):
  nsub=int(60/dt)
  native,audit=route(f,forcing.forcing_cms.to_numpy(),nsub,dt,return_native=True)
  # Match production trapezoidal interval reporting, with held interval forcing.
  bins=native.reshape(-1,nsub)
  initial=np.r_[forcing.forcing_cms.iloc[0],bins[:-1,-1]]
  q=bins.mean(axis=1)+(initial-bins[:,-1])/(2*nsub)
  i=int(q.argmax());result=dict(state=label,dt_seconds=dt,peak_utc=str(w.time_utc.iloc[i]),peak_cms=q[i],max_difference_from_published_cms=float(abs(q-w.q_outlet_cms.to_numpy()[:len(q)]).max()),comparison_intervals=len(q),**audit)
  checks.append(result);pd.DataFrame(checks).to_csv(OUT/'independent_routing_checks.csv',index=False)
  print(result,flush=True)
checks_frame=pd.DataFrame(checks)
assert checks_frame.loc[checks_frame.dt_seconds.eq(5),'max_difference_from_published_cms'].max()<1e-8
for dt,group in checks_frame.groupby('dt_seconds'):
 times=group.set_index('state').peak_utc.map(pd.Timestamp)
 assert (times['child']-times['parent']).total_seconds()==300
(OUT/'provenance.json').write_text(json.dumps(dict(record=r.to_dict(),domain=domain,unchanged_common_links=len(common),note='No production edits or refitting. Transfer moments are model diagnostics, not baseline-separated event moments.'),indent=2,default=str))

# Add a self-contained real case without changing the existing metric definitions.
book=ROOT/'RAPID/notebooks/hydrograph_response_worked_examples_20261012.ipynb'
n=nb.read(book,as_version=4)
tag='cb006_audit_20261012'
n.cells=[cell for cell in n.cells if cell.metadata.get('audit_section')!=tag]
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(s,metadata={'audit_section':tag}))
def code(s):cells.append(nb.v4.new_code_cell(s,metadata={'audit_section':tag}))
md(r'''## Real case: 46 / CB006_bubble_101 — later crest, faster model travel

The requested label was `45 / CB006_bubble_101`. That state is in **network 46**; network 45's sixth compound is `CB006_bubble_26`. The audit below follows the exact `CB006_bubble_101` state in the active published registry.

This notebook describes response variables, not an established categorical classification of hydrograph types. Keep **crest timing**, **height**, **bulk timing**, **widening** and **asymmetry** separate. This real case explains why a later crest must not automatically be labelled a slower network.

Parent maximum: **2024-09-03 12:15:30 UTC**, **1883.352286 m³/s**. Child maximum: **12:20:30 UTC**, **1888.382382 m³/s**. Thus sampled and reviewed crest shifts are **+300 s** and total peak change is **+0.267082%**. The reported values are 60-second interval means; both published runs use a 5-second routing clock. The crests are extremely flat, but the delay survives a halved clock.

The following image includes an own-maximum-subtracted zoom. Subtracting each maximum is for visualizing timing only; it is not a baseline choice for excess-volume analysis.''')
code("""from pathlib import Path
import pandas as pd
from IPython.display import display, Image
AUDIT_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'RAPID').is_dir())
AUDIT = AUDIT_ROOT/'RAPID/outputs/cb006_hydrograph_audit_20261012'
display(Image(filename=str(AUDIT/'hydrograph_comparison.png')))
""")
md(r'''### Are widths and slopes consistent?

The matched domain is node **25 → 23**, with 11 parent reaches replaced by child reach **541**. Both retain the same boundary heads, **220.638556 → 220.530458 m**, a **0.108098 m** drop. Each saved slope agrees with `(head_us − head_ds) / link_length_m`. Hydraulic parameters match the current hydraulic frames; widths match the geometry's `wid_adj_routing` field. All 530 surviving reaches retain identical lengths, widths, depths, heads, slopes, roughness, K, X and subdivision counts. Neither parent-domain nor replacement celerities are capped; four caps elsewhere are common to both states.

The replacement width is **918.774803 m**. Its source is the trimmed two-bank **wet** width, not the **2020.683718 m total corridor width** that includes dry space. It uses 437 trimmed samples and no routing width correction. This verifies provenance and implementation consistency; it is not independent field validation of every measured width or fitted head.

The child width exceeds most parent reach widths, but not link 43 (**1058.088602 m**). There is no single parent width obtained by adding every reach: many are in series or shared between routes.

The replacement length is **4693.697291 m**, so its slope is **2.303052×10⁻⁵ m/m**. The dominant parent route (50.30% width allocation) is **4536.038149 m**, hence its end-to-end slope is **2.383099×10⁻⁵ m/m**, slightly steeper than the child. A collapse is not guaranteed to be shorter or steeper than every parent route. The tables retain the actual per-reach slopes and complete-route comparisons.''')
code("""audit_parent=pd.read_csv(AUDIT/'parent_domain_parameters.csv')
audit_child=pd.read_csv(AUDIT/'child_domain_parameters.csv')
columns=['id_link','id_us_node','id_ds_node','link_length_m','width','depth_m','head_us','head_ds','slope','rapid_celerity_mps']
print('Parent physical reaches');display(audit_parent[columns])
print('Replacement reach');display(audit_child[columns])
print('Complete parent routes; probabilities use the routing width allocation')
display(pd.read_csv(AUDIT/'parent_routes.csv'))
""")
md(r'''### Why later peak does not imply slower transport

For this model, `c = (5/3) depth^(2/3) sqrt(slope) / n`, and the replacement celerity is **1.064278 m/s**. Its travel-time parameter `L/c` is **73.503649 min**, compared with **83.88–288.22 min** along individual parent routes and a width-allocation-weighted parent mean of **134.084848 min**. In this sense, the child is faster than every parent route.

Cunge routing also changes shape. It uses diffusivity `D = Q_ref/(2 width slope)` and `X = 0.5[1 − Q_ref/(width slope c Δx)]`, rather than routing a rigidly translated hydrograph. [USACE's Cunge description](https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/channel-flow/muskingum-cunge-model) distinguishes propagation and diffusion. Our implementation freezes a reference state; it does not update the coefficients at each timestep.

The **model transfer function** has mean delay `Σ K` along a route and second cumulant `Σ K²(1−2X)` over its numerical cells. At splits, combine routes using their allocation probabilities, including between-route variance. The local mean changes **134.085 → 73.504 min**, and its transfer spread changes **844.873 → 238.841 min**. The whole-network mean is also **60.581 min earlier**. This change in spread and tail weighting means the event is reshaped, not simply translated; the mode (crest) can move later even when the mean model delay decreases. It is not a contradiction.

**These are routing diagnostics, not measured water-particle travel times or event-excess centroid/spread responses.** Negative previous-inflow coefficients occur, but the checked local impulse-response weights are nonnegative: coefficient signs alone do not determine kernel signs. The normalized model kernel is still distinct from an observed event-excess distribution; its algebraic transfer moments remain separate diagnostics. The real forcing starts and ends at different flows; no real-event excess baseline is chosen here. A model-transfer spread is also not proof of a particular change in half-height event width.''')
code("display(pd.read_csv(AUDIT/'routing_transfer_moments.csv'))")
md(r'''### Verification and limits

An independent DAG cascade reproduces the published parent and child outlet series to **less than 3×10⁻¹⁰ m³/s** over the **31,680 supplied forcing intervals**, including the peak. It uses held interval forcing and the production trapezoidal reporting convention. The additional published settled tail is not rerun by this check. Halving the native clock to **2.5 seconds** keeps the same reported parent and child peak bins and therefore the same **+5 min** shift. This checks time discretization, not spatial-grid convergence or the physical accuracy of the reference hydraulics.

The hydraulic-context manifest still carries older `requires_regeneration`/`historical` status text; this audit therefore checks the actual current frames and completed run products directly rather than relying on those status labels. The saved parameters match the current frames exactly.

Use **“later, slightly higher crest”** for the observed hydrograph response. Use **“shorter model mean delay”** for the routing diagnostic. Avoid one undifferentiated faster/slower label. Geometry consistency and time-step reproducibility do not establish universal physical correctness of the depth–width law, fitted heads, diffusion assumptions, or spatial discretization.

All checks write to `RAPID/outputs/cb006_hydrograph_audit_20261012`. No production parameter, calibration, or published hydrograph was changed. Reproduce with `RAPID/investigations/cb006_hydrograph_audit_20261012.py`.''')
code("display(pd.read_csv(AUDIT/'independent_routing_checks.csv'))")
for cell in cells:
 if cell.cell_type=='code':compile(cell.source,'<audit notebook cell>','exec')
n.cells.extend(cells);nb.validate(n);nb.write(n,book)
print('Updated',book,flush=True)
