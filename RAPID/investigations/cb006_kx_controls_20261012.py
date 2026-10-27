"""Diagnostic K/X interventions on fixed parent topology; no physical refit."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src')]
from rapid_tools.cunge import route,topology
OUT=ROOT/'RAPID/outputs/cb006_hydrograph_audit_20261012'
meta=json.loads((OUT/'provenance.json').read_text());r=meta['record'];domain=meta['domain']
parent=pd.read_parquet(r['parent_parameters_parquet']);child=pd.read_parquet(r['parameters_parquet'])
local=parent[parent.id_link.isin(domain['parent_links'])]
mom=pd.read_csv(OUT/'routing_transfer_moments.csv').query('scope == "local"').set_index('state')
paths=pd.read_csv(OUT/'parent_routes.csv');mean=mom.loc['parent','mean_minutes'];childmean=mom.loc['child','mean_minutes']
vbetween=float(paths.probability@((paths.advective_minutes-mean)**2));vwithin=mom.loc['parent','sd_minutes']**2-vbetween
ratio=childmean/mean
# After K scaling, choose X scaling to reproduce child transfer variance.
scale=(mom.loc['child','sd_minutes']**2/ratio**2-vbetween)/vwithin
settings=dict(k_factor=ratio,x_diffusion_factor=scale,parent_between_route_variance_min2=vbetween,parent_within_route_variance_min2=vwithin)
(OUT/'kx_control_settings.json').write_text(json.dumps(settings,indent=2));print(settings,flush=True)
forcing=pd.read_parquet(r['forcing_parquet']);flow=forcing.forcing_cms.to_numpy();dt=5.;ns=12
results=[];waves=pd.DataFrame({'time_utc':forcing.time_utc})
for name,kfactor,xfactor in [('parent',1.,1.),('K_only',ratio,1.),('X_only',1.,scale),('K_and_X',ratio,scale)]:
 f=parent.copy();sel=f.id_link.isin(domain['parent_links'])
 f.loc[sel,'rapid_k']*=kfactor
 f.loc[sel,'rapid_x']=.5+xfactor*(f.loc[sel,'rapid_x']-.5)
 native,audit=route(f,flow,ns,dt,return_native=True);bins=native.reshape(-1,ns)
 q=bins.mean(axis=1)+(np.r_[flow[0],bins[:-1,-1]]-bins[:,-1])/(2*ns)
 waves[name]=q;i=int(q.argmax());base=pd.Timestamp(r['parent_peak_utc'])
 results.append(dict(control=name,local_K_multiplier=kfactor,local_X_diffusion_multiplier=xfactor,local_mean_minutes=mean*kfactor,local_spread_minutes=kfactor*np.sqrt(vbetween+xfactor*vwithin),peak_utc=str(forcing.time_utc.iloc[i]),peak_shift_minutes=(forcing.time_utc.iloc[i]-base).total_seconds()/60,peak_cms=q[i],**audit))
 pd.DataFrame(results).to_csv(OUT/'kx_control_results.csv',index=False);print(results[-1],flush=True)
 if name=='parent':
  published=pd.read_parquet(r['parent_outlet_parquet']);np.testing.assert_allclose(q,published.q_outlet_cms.iloc[:len(q)],rtol=0,atol=1e-8)
actual=pd.read_parquet(r['outlet_parquet']);waves['actual_child']=actual.q_outlet_cms.iloc[:len(waves)].to_numpy()
waves.to_parquet(OUT/'kx_control_hydrographs.parquet',index=False)
print('Completed diagnostic controls',flush=True)

# Exact unit-step response illustrates prompt transmission versus a long tail.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(11,4));kernel=[]
for label,color,f in [('parent','#2563eb',local),('child','#dc2626',child[child.id_link.isin(domain['child_links'])])]:
 f=f.reset_index(drop=True);k=f.rapid_k.to_numpy();x=f.rapid_x.to_numpy();den=dt+2*k*(1-x)
 aa=(dt-2*k*x)/den;bb=(dt+2*k*x)/den;dd=(2*k*(1-x)-dt)/den
 assert (aa>=0).all() and (dd>=0).all() and (bb+dd*aa>=0).all()
 pulse=np.r_[0.,np.ones(30*24*60)]
 native,_=route(f,pulse,ns,dt,return_native=True)
 cdf=native[ns:];hours=np.arange(len(cdf))*dt/3600
 for ax in axes:ax.plot(hours[::12],cdf[::12],color=color,label=label)
 kernel.append(dict(state=label,prompt_fraction=cdf[0],max_cell_memory_hours=float((-dt/np.log(dd)/3600).max()),tail_fraction_after_24h=float(1-cdf[int(24*3600/dt)])))
axes[0].set(xlim=(0,12),ylim=(.80,1),title='Target-domain unit-step response: first 12 h',xlabel='Hours after input step',ylabel='Fraction of final response')
axes[1].set(xlim=(0,240),ylim=(.96,1.0005),title='Tail: next ten days',xlabel='Hours after input step',ylabel='Fraction of final response')
for ax in axes:ax.grid(alpha=.2);ax.legend()
fig.tight_layout();fig.savefig(OUT/'kx_prompt_and_tail.png',dpi=160);plt.close(fig)
pd.DataFrame(kernel).to_csv(OUT/'kx_kernel_diagnostics.csv',index=False)
print(pd.DataFrame(kernel).to_string(index=False),flush=True)

import nbformat as nb
book=ROOT/'RAPID/notebooks/hydrograph_response_worked_examples_20261012.ipynb';n=nb.read(book,as_version=4);tag='cb006_kx_explanation'
n.cells=[cell for cell in n.cells if cell.metadata.get('audit_section')!=tag]
def md(s):return nb.v4.new_markdown_cell(s,metadata={'audit_section':tag})
def code(s):return nb.v4.new_code_cell(s,metadata={'audit_section':tag})
cells=[md(r'''## Why K and X allow an earlier mean but a later crest

**Yes: the K/X changes reproduce the direction of the observed effect.** Geometry affects both propagation and spreading; a crest maximum is a result of the entire routing response, not just length divided by celerity.

For each numerical cell in the actual code:

$$K=\Delta x/c,\qquad X=\tfrac12\left[1-\frac{Q_{ref}}{W S c\Delta x}\right].$$

K is in seconds **per numerical cell**, not per physical reach. Each local reach in this case has eight cells. The child has **K = 551.277 s per cell**, giving **8K = 73.504 min**, and **X = −41.734**. The parent has different K and X for each reach; it is incorrect to compare the child with an unweighted average of parent X values.

The model mean delay along a route is `sum(K)`; its second central moment is `sum[K²(1−2X)]`, summed over all numerical cells. Mixing parent routes also adds between-route variance. Thus **K affects both delay and spreading, while X affects spreading and the full timing response**. X is not an extra travel time, and K-only is not a pure translation.

[USACE's Cunge equations](https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/channel-flow/muskingum-cunge-model) give these K/X relationships. This project uses frozen-reference coefficients; the cited HEC implementation also describes time-varying coefficients.'''),md(r'''### Controlled experiment: only the target-domain K and X are altered

Keep the **parent topology, split fractions, exterior parameters, forcing, timestep and reporting convention**. Apply these diagnostic overrides only to its 11 target reaches:

- **K-only:** multiply K by **0.548188**, matching the actual child's local mean delay.
- **X-only:** replace X by `0.5 + 0.259939 (X − 0.5)`, reducing the spreading factor `1−2X`.
- **K and X:** apply both. The two scaling factors were chosen from the actual child's **model mean and variance**, not fitted to its peak time. Together they match those two transfer moments exactly, including the parent's between-route variance.

These are causal interventions in the numerical recurrence, **not geometrically self-consistent alternative river states**. Width, slope and celerity metadata are not reinterpreted as physically generating the overridden coefficients.

| Run | Outlet crest shift versus parent |
|---|---:|
| K-only | **−16 min** |
| X-only | **+41 min** |
| K and X | **+4 min** |
| Actual collapsed network | **+5 min** |

The effects are not additive: `−16 + 41` is not the combined result. Peak extraction is nonlinear, and K and X interact. The remaining one-minute difference and the height difference show that matching two moments does not reproduce every detail of the actual changed topology and response kernel. This is strong evidence for the proposed K/X mechanism, not an exact decomposition of the actual collapse into independent K and X contributions.'''),code('''from pathlib import Path
import pandas as pd
from IPython.display import display, Image
KX_ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'RAPID').is_dir())
KX=KX_ROOT/'RAPID/outputs/cb006_hydrograph_audit_20261012'
display(pd.read_csv(KX/'kx_control_results.csv')[['control','local_mean_minutes','local_spread_minutes','peak_shift_minutes','peak_cms']])
'''),md(r'''### The concrete mechanism: less prompt response, but a shorter long tail

For one cell the recurrence is `out[t] = A in[t] + B in[t−1] + D out[t−1]`. Its impulse-response weight at the current step is A; later weights are `(B + D A) D^(j−1)`. In this target domain all these impulse weights are nonnegative, even though B is negative. **A negative previous-inflow coefficient does not by itself imply a negative impulse response.**

The complete parent domain transmits **87.52%** of a small input step in the same native update; the actual child transmits **82.74%**. The remainder is distributed over later updates. Some parent cell memories extend to an **87.22-hour** exponential time constant, versus **6.54 hours** per child cell. The parent therefore combines a stronger prompt component with a smaller but much longer tail. The child reduces that tail enough to lower its mean delay, while its weaker prompt component can move this event's crest later.

The step-response plot shows this directly: the parent initially responds more; the child subsequently approaches its final response sooner. The K/X controls above establish the corresponding peak effect for the actual event. A unit step by itself does not predict the crest direction for every possible input.

**“Same native update” is a property of this numerical routing operator, not water physically crossing kilometres in five seconds.** These very negative-X coefficients require care in physical interpretation. This audit explains the saved model outcome; it does not establish spatial-grid convergence or validate its prompt response against observations. The model kernel and its moments remain separate from baseline-separated event response variables.'''),code("display(pd.read_csv(KX/'kx_kernel_diagnostics.csv'))\ndisplay(Image(filename=str(KX/'kx_prompt_and_tail.png')))")]
n.cells[1:1]=cells
old='Negative previous-inflow coefficients exist in these Cunge runs, so do not interpret their discrete impulse weights as a probability density. The algebraic transfer moments remain well defined.'
new='Negative previous-inflow coefficients occur, but the checked local impulse-response weights are nonnegative: coefficient signs alone do not determine kernel signs. The normalized model kernel is still distinct from an observed event-excess distribution; its algebraic transfer moments remain separate diagnostics.'
for cell in n.cells:
 if cell.cell_type=='markdown':cell.source=cell.source.replace(old,new)
nb.validate(n);nb.write(n,book)
# Keep the earlier audit generator consistent with the corrected kernel statement.
source=ROOT/'RAPID/investigations/cb006_hydrograph_audit_20261012.py'
source.write_text(source.read_text().replace(old,new))
print('Added K/X explanation and corrected impulse-weight wording',flush=True)
