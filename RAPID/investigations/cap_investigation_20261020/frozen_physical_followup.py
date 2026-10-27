"""Exact hydraulic accounting and saved-wave translation controls; no fits/reroutes."""
from pathlib import Path
import sys,json,hashlib,itertools,math
import numpy as np,pandas as pd
import networkx as nx
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'RAPID/src'))
from rapid_tools.current_domains import allocation_tables
OUT=ROOT/'RAPID/outputs/uncapped_event_window_20261020'
EXT=ROOT/'RAPID/outputs/hydrograph_closed_unit_extension_20261019'
SC=ROOT/'RAPID/outputs/hydrograph_predictor_scopes_20261014'
DEST=OUT/'frozen_physical_followup_20261026';DEST.mkdir(exist_ok=True)
SOURCES={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 p=Path(p);SOURCES[str(p)]=sha(p)
 if p.suffix=='.json':return json.loads(p.read_text())
 return pd.read_csv(p,dtype={'example':str}) if p.suffix=='.csv' else pd.read_parquet(p)
freeze=read(OUT/'frozen_failure_audit_20261026/manifest.json')['protected_sha256']
for p,h in freeze.items():assert sha(p)==h
D=read(OUT/'frozen_failure_audit_20261026/all_errors.parquet');D=D[D.model.eq('depth_informed_10')].set_index(['example','state'])
keys=read(OUT/'frozen_failure_audit_20261026/physical_context.csv')[['example','state']]
keys=pd.concat([keys,pd.DataFrame([dict(example='13',state='all_closed_units_collapsed')])]).drop_duplicates()
jobs={}
for p in [ROOT/'RAPID/outputs/event_window_plan_20261013/plan.json',EXT/'routing_plan.json',OUT/'routing_plan.json']:
 for c in read(p)['cases']:
  for j in c['jobs']:
   for name in {j['state_id'],*j.get('aliases',[])}:jobs[c['example'],name]=j
meta=read(SC/'geometry_responses.parquet').set_index(['example','state']);graphs=read(SC/'domain_graphs.csv').set_index(['example','state','scope'])
st=read(EXT/'state_metrics.parquet');new=read(OUT/'review/uncapped_state_metrics.parquet');st=pd.concat([st[~st.example.isin(new.example)],new]);st=st[st.window.eq('0.5%')].set_index(['example','state'])
solver=read(SC/'resistance_solver_audit.csv');solver=solver[np.isclose(solver.alpha,.65)].set_index('graph')
rows=[];terms=[];edges=[];waves=[]
def factors(f,w):
 h=f.depth_m.to_numpy()**(-2/3);s=f.slope.to_numpy()**(-.5);L=w.sum();H=np.average(h,weights=w);S=np.average(s,weights=w);C=np.average(h*s,weights=w)/(H*S)
 return np.array([L,H,S,C])
def shapley(a,b,n):
 ans=[]
 for k in range(4):
  value=0
  other=[i for i in range(4) if i!=k]
  for size in range(4):
   for sub in itertools.combinations(other,size):
    x=a.copy();x[list(sub)]=b[list(sub)];y=x.copy();y[k]=b[k]
    value+=(np.prod(y)-np.prod(x))*math.factorial(size)*math.factorial(3-size)/24
  ans.append(value*.6*n/3600)
 return ans
# Integrate piecewise linear waveform exactly using midpoint Simpson (t*q quadratic).
def centroid(t,q):
 x=t-t[0];h=np.diff(x);mass=np.sum(h*(q[:-1]+q[1:])/2)
 moment=np.sum(h/6*(x[:-1]*q[:-1]+4*(x[:-1]+h/2)*(q[:-1]+q[1:])/2+x[1:]*q[1:]))
 return t[0]+moment/mass,mass
def moments(f):
 graph=nx.DiGraph();graph.add_edges_from(zip(f.u,f.v));order=list(nx.topological_sort(graph));root=next(u for u in order if graph.in_degree(u)==0);end=next(u for u in order if graph.out_degree(u)==0)
 prob={root:1.};m1={root:0.};m2={root:0.};diff={root:0.}
 for u in order:
  for rr in f[f.u.eq(u)].itertuples():
   w=rr.split_fraction;mu=rr.count*rr.rapid_k/3600;v=rr.count*rr.rapid_k**2*(1-2*rr.rapid_x)/3600**2
   prob[rr.v]=prob.get(rr.v,0)+prob[u]*w
   m1[rr.v]=m1.get(rr.v,0)+w*(m1[u]+prob[u]*mu)
   m2[rr.v]=m2.get(rr.v,0)+w*(m2[u]+2*mu*m1[u]+prob[u]*mu**2)
   diff[rr.v]=diff.get(rr.v,0)+w*(diff[u]+prob[u]*v)
 assert abs(prob[end]-1)<1e-10
 return m1[end],m2[end]-m1[end]**2,diff[end]
for ex,state in keys.itertuples(index=False,name=None):
 r=D.loc[(ex,state)];fs={side:read(jobs[ex,name]['parameters']).set_index('id_link',drop=False) for side,name in [('parent','S000_base'),('child',state)]}
 n=float(fs['parent'].n_manning.iloc[0]);assert np.allclose(fs['child'].n_manning,n)
 if r.cohort=='segmented':
  line=pd.read_csv(EXT/f'variants/{ex}/{state}/lineage.csv');SOURCES[str(EXT/f'variants/{ex}/{state}/lineage.csv')]=sha(EXT/f'variants/{ex}/{state}/lineage.csv');line=line[line.role.eq('collapsed_block')]
  selected={side:fs[side].loc[line[side+'_link_id'].unique()] for side in fs}
  weights={side:f.reference_flow_fraction.to_numpy()*f.link_length_m.to_numpy() for side,f in selected.items()}
 else:
  g=graphs.loc[(ex,state,meta.loc[(ex,state),'B_scope'])];selected={};weights={}
  for side in fs:
   geom=read(SC/'graphs'/f'{g[side+"_graph"]}.parquet');alloc=allocation_tables(geom)[1].set_index('id_link');selected[side]=fs[side].loc[alloc.index];weights[side]=alloc.mean_length_contribution_m.to_numpy()
 rec=dict(example=ex,state=state,cohort=r.cohort,target_hours=r.centroid_shift_hours,predicted_hours=r.predicted_hours,error_hours=r.error_hours,n=n,delta_G_inverse_depth=r.delta_G_inverse_depth,resistance=r.resistance_explicit_depth)
 for scope in ['full','descriptor']:
  fac={};T={}
  for side in fs:
   f=fs[side] if scope=='full' else selected[side];w=(f.reference_flow_fraction*f.link_length_m).to_numpy() if scope=='full' else weights[side]
   assert not f.rapid_celerity_capped.any()
   a=factors(f,w);fac[side]=a;T[side]=.6*n*np.prod(a)/3600
   np.testing.assert_allclose(T[side],np.sum(w/f.rapid_celerity_mps)/3600,rtol=1e-10)
   gh=np.sum(w/f.depth_m/np.sqrt(f.slope));correction=np.sum(w/f.depth_m/np.sqrt(f.slope)*f.depth_m**(1/3))/gh
   terms.append(dict(example=ex,state=state,scope=scope,side=side,travel_hours=T[side],length_km=a[0]/1000,effective_depth_m=a[1]**(-1.5),effective_slope=a[2]**-2,covariance_factor=a[3],Gh=gh,depth_exponent_correction=correction))
   rec[f'{scope}_{side}_travel_h']=T[side]
  delta=T['child']-T['parent'];rec[scope+'_delta_h']=delta
  contributions=shapley(fac['parent'],fac['child'],n);np.testing.assert_allclose(sum(contributions),delta,atol=1e-10)
  for label,val in zip(['length','depth','slope','covariance'],contributions):rec[f'{scope}_{label}_h']=val
  if scope=='descriptor':
   for side in fs:
    f=selected[side];fullw=f.reference_flow_fraction*f.link_length_m
    rec[f'{side}_descriptor_fullweight_h']=float((fullw/f.rapid_celerity_mps).sum()/3600)
   rec['descriptor_actual_delta_h']=rec['child_descriptor_fullweight_h']-rec['parent_descriptor_fullweight_h']
   rec['outside_descriptor_delta_h']=rec['full_delta_h']-rec['descriptor_actual_delta_h']
 # Exact network transfer moments: route mixing variance plus Cunge diffusion variance.
 for side,f in fs.items():
  mean,mix,diff=moments(f);np.testing.assert_allclose(mean,rec['full_'+side+'_travel_h'],rtol=1e-10)
  rec[side+'_path_variance_h2']=mix;rec[side+'_diffusion_variance_h2']=diff;rec[side+'_transfer_sd_h']=float(np.sqrt(max(0,mix+diff)))
 # Segmented equivalent resistance dilution by unchanged serial connectors.
 if r.cohort=='segmented':
  parent,child=fs['parent'],fs['child'];ids=child.index.difference(selected['child'].index)
  assert set(ids)<=set(parent.index)
  np.testing.assert_allclose(child.loc[ids,['link_length_m','width','depth_m']],parent.loc[ids,['link_length_m','width','depth_m']],rtol=1e-10)
  np.testing.assert_allclose(child.loc[ids,'reference_flow_fraction'],1,atol=1e-10)
  # Every retained link carries all flow, hence is in series with the edited blocks.
  a=(parent.depth_m/parent.width**.65).iloc[0];pg=graphs.loc[(ex,'fully_collapsed','N'),'parent_graph'];sol=solver.loc[pg]
  assert bool(sol.success) and sha(SC/'graphs'/f'{pg}.parquet')==sol.graph_sha256
  SOURCES[str(SC/'graphs'/f'{pg}.parquet')]=sol.graph_sha256
  kp=float(sol.K)/a**3;kc=float((child.link_length_m/(child.width**2*child.depth_m**3)).sum());conn=child.loc[ids];ku=float((conn.link_length_m/(conn.width**2*conn.depth_m**3)).sum())
  np.testing.assert_allclose(-np.log10(kc/kp),r.resistance_explicit_depth,atol=1e-10)
  rec['connector_parent_resistance_fraction']=ku/kp;rec['connector_child_resistance_fraction']=ku/kc;rec['edited_resistance_log']=-np.log10((kc-ku)/(kp-ku))
 # Largest full-network edge contributions; distinguish identical links from changed ones.
 p,c=fs['parent'],fs['child'];shared=p.index.intersection(c.index)
 for side,f,sign in [('parent',p,-1),('child',c,1)]:
  z=sign*f.reference_flow_fraction*f.link_length_m/f.rapid_celerity_mps/3600
  for idx in abs(z).nlargest(5).index:
   rr=f.loc[idx];edges.append(dict(example=ex,state=state,side=side,id_link=int(idx),signed_travel_h=z.loc[idx],length_km=rr.link_length_m/1000,depth_m=rr.depth_m,slope=rr.slope,flow_fraction=rr.reference_flow_fraction,celerity=rr.rapid_celerity_mps,in_descriptor=idx in selected[side].index))
 # Pure translation at physical mean change, retaining observed parent waveform and fixed window.
 pr=st.loc[(ex,'S000_base')];cr=st.loc[(ex,state)];a,b=pr.rise_seconds,pr.fall_seconds;base=pr.baseline_cms
 data={}
 for side,rr in [('parent',pr),('child',cr)]:
  ff=read(rr.outlet);assert sha(rr.outlet)==rr.outlet_sha256
  data[side]=(ff.time_utc.astype('int64').to_numpy()/1e9,ff.q_outlet_cms.to_numpy())
 tp,qp=data['parent'];tc,qc=data['child'];delay=rec['full_delta_h']*3600
 # Include all translated breakpoints for exact integration; constant padding flagged.
 t=np.unique(np.r_[a,b,tp[(tp>a)&(tp<b)],tc[(tc>a)&(tc<b)],(tp+delay)[(tp+delay>a)&(tp+delay<b)]])
 parent=np.interp(t,tp,qp)-base;child=np.interp(t,tc,qc)-base;shift=np.interp(t-delay,tp,qp)-base
 pm,pv=centroid(t,parent);cm,cv=centroid(t,child);sm,sv=centroid(t,shift)
 rec['translation_sensitivity_at_zero']=1+((a-pm)*parent[0]-(b-pm)*parent[-1])/pv
 rec['window_start_excess']=float(parent[0]);rec['window_end_excess']=float(parent[-1])
 rec['window_start_boundary_term']=(pm-a)*parent[0]/pv
 rec['window_end_boundary_term']=(b-pm)*parent[-1]/pv
 np.testing.assert_allclose((cm-pm)/3600,r.centroid_shift_hours,atol=1e-8)
 rec.update(translation_centroid_h=(sm-pm)/3600,window_translation_correction_h=(sm-pm)/3600-rec['full_delta_h'],shape_residual_h=(cm-sm)/3600,translation_padded=bool(t[0]-delay<tp[0] or t[-1]-delay>tp[-1]),parent_initial_excess=float(qp[0]-base),parent_final_excess=float(qp[-1]-base),volume_change_pct=100*(cv/pv-1))
 np.testing.assert_allclose(rec['full_delta_h']+rec['window_translation_correction_h']+rec['shape_residual_h'],r.centroid_shift_hours,atol=1e-8)
 rec['physical_h_per_million_G']=rec['full_delta_h']/r.delta_G_inverse_depth*1e6
 rec['response_retention']=r.centroid_shift_hours/rec['full_delta_h']
 rows.append(rec)
 if (ex,state) in [('15','CB001_bubble_4'),('22','CB001_bubble_9'),('55','all_closed_units_collapsed'),('41','all_closed_units_collapsed'),('46','CB001_bubble_90'),('45','all_closed_units_collapsed')]:
  waves.append(pd.DataFrame(dict(example=ex,state=state,time_hours=(t-a)/3600,parent=parent,child=child,translated_parent=shift)))
pd.DataFrame(rows).to_csv(DEST/'decomposition.csv',index=False);pd.DataFrame(terms).to_csv(DEST/'hydraulic_terms.csv',index=False);pd.DataFrame(edges).to_csv(DEST/'dominant_links.csv',index=False);pd.concat(waves).to_parquet(DEST/'wave_comparisons.parquet',index=False)
# Reuse only saved LOEO SHAP with predictions verified against the frozen files.
sh=read(OUT/'validation_shap_20261024/values.parquet');sh=sh[sh.strategy.eq('leave_one_network_out')].copy()
original=read(OUT/'frozen_failure_audit_20261026/all_errors.parquet')
check=sh.merge(original[['example','state','model','predicted_hours']],on=['example','state','model'],validate='one_to_one');assert len(check)==4370
np.testing.assert_allclose(check.predicted_minutes/60,check.predicted_hours,rtol=1e-10,atol=1e-10)
shcols=[c for c in sh if c.startswith('shap__')]
np.testing.assert_allclose(sh.base_minutes+sh[shcols].sum(axis=1),sh.predicted_minutes,atol=1e-7)
check.merge(keys,on=['example','state'],validate='many_to_one').to_csv(DEST/'saved_shap_cases.csv',index=False)
# Plot actual versus translated hydrographs for the principal cases and controls.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
for ax,wave in zip(axes.flat,waves):
 ex=wave.example.iloc[0];state=wave.state.iloc[0]
 for col,label,color,ls in [('parent','Parent','#555555','-'),('child','Actual child','#0072B2','-'),('translated_parent','Parent shifted by physical mean change','#D55E00','--')]:
  ax.plot(wave.time_hours,wave[col],label=label,color=color,linestyle=ls,linewidth=1.1)
 ax.set_title(ex+' / '+state,fontsize=10);ax.set_xlabel('Hours from fixed window start');ax.set_ylabel('Excess discharge (m³/s)');ax.grid(alpha=.2)
axes.flat[0].legend(fontsize=7)
fig.savefig(DEST/'translation_controls.png',dpi=160);plt.close(fig)
for p,h in {**freeze,**SOURCES}.items():assert sha(p)==h
for source in [Path(__file__),ROOT/'RAPID/src/rapid_tools/current_domains.py',ROOT/'RAPID/src/rapid_tools/cunge.py']:
 SOURCES[str(source)]=sha(source)
(DEST/'manifest.json').write_text(json.dumps(dict(fits=0,reroutes=0,protected_sha256=freeze,sources=SOURCES,method='Exact full-network mean-time accounting; descriptor-domain Shapley algebra over length, inverse-depth, inverse-slope, covariance; fixed-window parent translation by full mean-time change. Algebraic contributions are not feasible geometry interventions.'),indent=2)+'\n')
print(pd.DataFrame(rows).to_string(index=False))
