"""Parent-child predictor exploration using saved Cunge parameters only.

No routing, hydraulic solving, calibration, lineage inference, or production writes.
Closed-domain path moments are computed by DAG recursions, never path enumeration.
"""
from pathlib import Path
from datetime import datetime
import json, hashlib, argparse
import numpy as np
import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'RAPID/outputs/cunge_fixed_comparison_plus37_20261006'
MODELS = ROOT/'RAPID/outputs/cunge_fixed_extra_trees_plus37_20261006'
AUDIT = ROOT/'RAPID/outputs/prediction_audit_20261006T164258'
# Name -> (length exponent, time exponent, family, interpretation).
METRICS = {
 'total_length':(1,0,'length','Sum of unique physical reach lengths'),
 'area_proxy':(2,0,'width','Sum of length times width; channel-area proxy, not polygon union'),
 'width_mean':(1,0,'width','Length-weighted reach width mean'),
 'width_sd':(1,0,'width','Length-weighted reach width standard deviation'),
 'width_cv':(0,0,'width','Length-weighted width SD/mean'),
 'width_harmonic':(1,0,'width','Total length / sum(length/width)'),
 'width_p10':(1,0,'width','Length-weighted width 10th percentile'),
 'width_p90':(1,0,'width','Length-weighted width 90th percentile'),
 'bifurcations':(0,0,'topology','Number of physical branching vertices; not routing cells'),
 'cycle_rank':(0,0,'topology','E-V+C on the underlying undirected multigraph'),
 'junction_excess':(0,0,'topology','Sum(max(out_degree-1,0))'),
 'bifurcation_density':(-1,0,'topology','Bifurcations per metre of unique physical channel'),
 'route_length_mean':(1,0,'length','Expected complete route length under width/reference split allocation'),
 'route_length_sd':(1,0,'length','SD of allocated route length'),
 'route_length_cv':(0,0,'length','Allocated route length SD/mean'),
 'route_length_min':(1,0,'length','Shortest directed source-to-sink physical length'),
 'route_length_max':(1,0,'length','Longest directed source-to-sink physical length'),
 'route_length_skew':(0,0,'length','Third central route-length moment divided by SD cubed; zero for deterministic routes'),
 'log1p_paths':(0,0,'topology','Natural log(1+complete directed path count); explicit complexity only'),
 'route_entropy':(0,0,'splits','Expected accumulated -log branch probability; not a bounded local imbalance'),
 'expected_splits':(0,0,'topology','Expected number of branching vertices encountered by a particle'),
 'dominance_mean':(0,0,'splits','Visit-weighted mean largest outgoing fraction at branching vertices; zero if none'),
 'dominance_sd':(0,0,'splits','Visit-weighted SD of largest outgoing fraction; zero if none'),
 'effective_branches_mean':(0,0,'splits','Visit-weighted mean 1/sum(p_j squared) at branching vertices'),
 'effective_branches_sd':(0,0,'splits','Visit-weighted SD effective branch count'),
 'split_entropy_mean':(0,0,'splits','Visit-weighted mean local entropy / log(out_degree)'),
 'split_entropy_sd':(0,0,'splits','Visit-weighted SD normalized local split entropy'),
 'split_exposure_mean':(0,0,'exposure','Unweighted mean conditional node visit probability across bifurcations'),
 'split_exposure_sd':(0,0,'exposure','SD conditional node visit probability across bifurcations'),
 'split_impact_mean':(0,0,'splits','Mean across bifurcations of visit_probability*(1-sum(p_j squared))'),
 'split_impact_sd':(0,0,'splits','SD across bifurcations of visit_probability*(1-sum(p_j squared))'),
 'split_impact_sum':(0,0,'splits','Sum of visit_probability*(1-sum(p_j squared)); repeated splitting can exceed one'),
 'branch_length_contrast_mean':(1,0,'split_contrast','Visit-weighted mean SD among branch expected remaining distances'),
 'branch_length_contrast_sd':(1,0,'split_contrast','Visit-weighted SD of branch distance contrast'),
 'branch_length_variance_sum':(2,0,'split_contrast','Sum visit_probability*Var(branch expected remaining distance); equals route-length variance'),
 'area_elongation':(0,0,'shape','Allocated mean route length squared / unique-edge area proxy; hypothesis only'),
 'sinuosity_proxy':(0,0,'shape','Allocated mean route length / shortest directed route; not map sinuosity'),
 'slope_mean':(0,0,'hydraulics','Length-weighted saved reference slope'),
 'slope_cv':(0,0,'hydraulics','Length-weighted saved slope CV'),
 'depth_mean':(1,0,'hydraulics','Length-weighted saved reference depth'),
 'celerity_mean':(1,-1,'hydraulics','Length-weighted saved calibrated Cunge celerity'),
 'celerity_cv':(0,0,'hydraulics','Length-weighted saved celerity CV'),
 'diffusivity_mean':(2,-1,'hydraulics','Length-weighted saved Cunge diffusivity'),
 'route_time_mean':(0,1,'hydraulics','Expected sum physical_length/celerity along route; parameter-derived, not child outlet lag'),
 'route_time_sd':(0,1,'hydraulics','SD of route advective travel times, excluding within-reach dispersion'),
 'route_time_cv':(0,0,'hydraulics','SD/mean of route advective travel time'),
 'route_time_skew':(0,0,'hydraulics','Third central route-time moment / SD cubed'),
 'branch_time_contrast_mean':(0,1,'hydraulics','Visit-weighted mean SD among branch expected remaining travel times'),
 'branch_time_contrast_sd':(0,1,'hydraulics','Visit-weighted SD of branch time contrast'),
 'branch_time_variance_sum':(0,2,'hydraulics','Sum visit*branch time-mean variance; equals between-route travel-time variance'),
 'within_route_dispersion':(0,2,'hydraulics','Expected sum 2*diffusivity*length/celerity cubed; low-frequency dispersion proxy'),
 'total_time_spread':(0,1,'hydraulics','sqrt(between-route time variance + within-route dispersion proxy)'),
 'capped_length_fraction':(0,0,'hydraulics','Physical length fraction with saved celerity cap active'),
}
HYDRAULIC = {k for k,v in METRICS.items() if v[2]=='hydraulics'}


def weighted_stats(x,w):
    x,w=np.asarray(x,float),np.asarray(w,float)
    if not len(x) or w.sum()==0:return 0.,0.
    mean=float(np.average(x,weights=w));return mean,float(np.sqrt(np.average((x-mean)**2,weights=w)))


def weighted_quantile(x,w,q):
    order=np.argsort(x,kind='stable');xx=np.asarray(x)[order];ww=np.asarray(w)[order]
    return float(xx[min(np.searchsorted(np.cumsum(ww),q*ww.sum()),len(xx)-1)])


def describe(frame, allow_hydraulics=True):
    """Aggregate a physical graph; route summaries require one closed inlet/outlet.

    Node probabilities are conditional on entry to this domain. Multiply by
    separately recorded network inlet exposure when comparing domains.
    """
    f=frame.reset_index(drop=True);L=f.link_length_m.to_numpy(float);W=f.width.to_numpy(float)
    assert len(f)>0 and np.all(L>0) and np.all(W>0) and not f.id_link.duplicated().any()
    g=nx.MultiDiGraph();out={}
    for i,r in enumerate(f.itertuples()):
        u,v=int(r.id_us_node),int(r.id_ds_node);g.add_edge(u,v,key=int(r.id_link));out.setdefault(u,[]).append((i,v))
    result={k:np.nan for k in METRICS};bifs=[u for u in g if g.out_degree(u)>1]
    result.update(total_length=L.sum(),area_proxy=np.dot(L,W),width_mean=weighted_stats(W,L)[0],width_sd=weighted_stats(W,L)[1],width_harmonic=L.sum()/np.sum(L/W),width_p10=weighted_quantile(W,L,.1),width_p90=weighted_quantile(W,L,.9),bifurcations=len(bifs),cycle_rank=g.number_of_edges()-len(g)+nx.number_weakly_connected_components(g),junction_excess=sum(max(0,g.out_degree(u)-1) for u in g),bifurcation_density=len(bifs)/L.sum())
    result['width_cv']=result['width_sd']/result['width_mean']
    if allow_hydraulics:
        for col,key in [('slope','slope'),('depth_m','depth'),('rapid_celerity_mps','celerity'),('hydraulic_diffusivity_m2_s','diffusivity')]:
            mean,sd=weighted_stats(f[col],L);result[key+'_mean']=mean
            if key+'_cv' in result:result[key+'_cv']=sd/mean
        result['capped_length_fraction']=np.average(f.rapid_celerity_capped.astype(float),weights=L)
    sources=[u for u in g if g.in_degree(u)==0];sinks=[u for u in g if g.out_degree(u)==0]
    if len(sources)!=1 or len(sinks)!=1 or not nx.is_directed_acyclic_graph(g):
        return result,dict(status='not_single_source_sink_DAG',edges=len(f),nodes=len(g))
    source,sink=sources[0],sinks[0];order=list(nx.topological_sort(g));visit={source:1.};edge_visit=np.zeros(len(f));probs={}
    for u in order:
        ee=out.get(u,[])
        if not ee:continue
        indexes=[i for i,v in ee];p=W[indexes]/W[indexes].sum();probs[u]=p
        for q,(i,v) in zip(p,ee):edge_visit[i]=visit[u]*q;visit[v]=visit.get(v,0.)+edge_visit[i]
    assert np.isclose(visit[sink],1,atol=1e-12)
    counts={sink:1};lo={sink:0.};hi={sink:0.}
    for u in reversed(order[:-1]):
        ee=out[u];counts[u]=sum(counts[v] for i,v in ee);lo[u]=min(L[i]+lo[v] for i,v in ee);hi[u]=max(L[i]+hi[v] for i,v in ee)
    result.update(log1p_paths=np.log1p(float(counts[source])),route_length_min=lo[source],route_length_max=hi[source])
    def moments(values):
        means={sink:0.};variances={sink:0.};thirds={sink:0.};contrast={}
        for u in reversed(order[:-1]):
            ee=out[u];p=probs[u];branches=np.array([values[i]+means[v] for i,v in ee]);mu=float(p@branches);dev=branches-mu
            means[u]=mu;variances[u]=float(sum(q*(variances[v]+dx**2) for q,dx,(i,v) in zip(p,dev,ee)))
            thirds[u]=float(sum(q*(thirds[v]+3*dx*variances[v]+dx**3) for q,dx,(i,v) in zip(p,dev,ee)))
            contrast[u]=float(np.sqrt(p@(dev**2)))
        sd=np.sqrt(max(variances[source],0));skew=thirds[source]/sd**3 if sd>1e-10*max(1,means[source]) else 0.
        total=sum(visit[u]*contrast[u]**2 for u in bifs)
        np.testing.assert_allclose(total,variances[source],rtol=1e-9,atol=1e-5)
        bw=[visit[u] for u in bifs];bm,bs=weighted_stats([contrast[u] for u in bifs],bw)
        return means[source],sd,skew,bm,bs,total
    mu,sd,skew,bm,bs,var= moments(L)
    result.update(route_length_mean=mu,route_length_sd=sd,route_length_cv=sd/mu,route_length_skew=skew,branch_length_contrast_mean=bm,branch_length_contrast_sd=bs,branch_length_variance_sum=var,area_elongation=mu**2/result['area_proxy'],sinuosity_proxy=mu/lo[source])
    bw=np.array([visit[u] for u in bifs]);dominance=[];eff=[];entropy=[];impacts=[];entropy_sum=0.
    for u in bifs:
        p=probs[u];h=float(-p@np.log(p));dominance.append(p.max());eff.append(1/(p@p));entropy.append(h/np.log(len(p)));impacts.append(visit[u]*(1-p@p));entropy_sum+=visit[u]*h
    for key,values in [('dominance',dominance),('effective_branches',eff),('split_entropy',entropy)]:
        result[key+'_mean'],result[key+'_sd']=weighted_stats(values,bw)
    result['split_exposure_mean'],result['split_exposure_sd']=weighted_stats(bw,np.ones(len(bw)))
    result['split_impact_mean'],result['split_impact_sd']=weighted_stats(impacts,np.ones(len(impacts)))
    result.update(expected_splits=bw.sum(),route_entropy=entropy_sum,split_impact_sum=sum(impacts))
    if allow_hydraulics:
        c=f.rapid_celerity_mps.to_numpy(float);diff=f.hydraulic_diffusivity_m2_s.to_numpy(float)
        mu,sd,skew,bm,bs,var=moments(L/c);disp=float(edge_visit@(2*diff*L/c**3))
        result.update(route_time_mean=mu,route_time_sd=sd,route_time_cv=sd/mu,route_time_skew=skew,branch_time_contrast_mean=bm,branch_time_contrast_sd=bs,branch_time_variance_sum=var,within_route_dispersion=disp,total_time_spread=np.sqrt(var+disp))
    return result,dict(status='ok',edges=len(f),nodes=len(g),source=int(source),sink=int(sink),max_outdegree=max(g.out_degree(u) for u in g),cycle_identity_error=result['cycle_rank']-result['junction_excess'])


def parse_ids(value):return {int(float(x)) for x in str(value).split(',') if x.strip()}


def dimensional_basis():
    names=['delta_t','event_T','L_parent','L_child','W_parent','W_child','c_parent','c_child','D_parent','D_child']
    D=np.array([[0,0,1,1,1,1,1,1,2,2],[1,1,0,0,0,0,-1,-1,-1,-1]],float)
    expressions={'delta_t/event_T':{0:1,1:-1},'L_child/L_parent':{3:1,2:-1},'W_parent/L_parent':{4:1,2:-1},'W_child/L_parent':{5:1,2:-1},'c_parent*event_T/L_parent':{6:1,1:1,2:-1},'c_child*event_T/L_parent':{7:1,1:1,2:-1},'D_parent*event_T/L_parent^2':{8:1,1:1,2:-2},'D_child*event_T/L_parent^2':{9:1,1:1,2:-2}}
    B=np.array([[powers.get(j,0) for j in range(len(names))] for powers in expressions.values()]).T
    assert np.linalg.matrix_rank(D)==2 and np.linalg.matrix_rank(B)==8 and np.allclose(D@B,0)
    return pd.DataFrame(D,index=['length_exponent','time_exponent'],columns=names),pd.DataFrame(B,index=names,columns=expressions)


def checks():
    # Analytic two-branch route distribution; tests topology, mean/SD, weighting,
    # splitting invariance, homothetic geometry scaling and dimension basis.
    f=pd.DataFrame(dict(id_link=[0,1,2],id_us_node=[0,0,1],id_ds_node=[1,1,2],link_length_m=[100.,200.,50.],width=[3.,1.,4.]))
    a,meta=describe(f,False);np.testing.assert_allclose([a['route_length_mean'],a['route_length_sd'],a['dominance_mean'],a['expected_splits'],a['cycle_rank']],[175.,np.sqrt(1875),.75,1.,1.])
    split=[]
    for r in f.itertuples():
        for j,(u,v) in enumerate([(r.id_us_node,100+r.id_link),(100+r.id_link,r.id_ds_node)]):split.append(dict(id_link=2*r.id_link+j,id_us_node=u,id_ds_node=v,link_length_m=r.link_length_m/2,width=r.width))
    b,_=describe(pd.DataFrame(split),False)
    for key in METRICS:
        if key not in HYDRAULIC:np.testing.assert_allclose(a[key],b[key],rtol=1e-10,atol=1e-8,equal_nan=True)
    scaled=f.copy();scaled[['link_length_m','width']]*=1000;c,_=describe(scaled,False)
    for key,(le,te,fam,desc) in METRICS.items():
        if key not in HYDRAULIC:np.testing.assert_allclose(c[key],a[key]*1000**le,rtol=1e-9,atol=1e-6,equal_nan=True)
    one=f.iloc[[0,2]].copy();o,_=describe(one,False);assert o['bifurcations']==0 and o['dominance_mean']==0 and o['route_length_sd']==0
    dimensional_basis()
    return dict(analytic_branch_moments=True,degree_two_subdivision=True,geometric_unit_scaling=True,no_split_convention=True,buckingham_nullspace=True)


def build(out=None):
    out=Path(out) if out else ROOT/'RAPID/outputs'/('parent_child_predictors_'+datetime.now().strftime('%Y%m%dT%H%M%S'))
    out.mkdir(parents=True,exist_ok=False);hashes={}
    def tracked(path):
        path=Path(path);hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();return path
    def read(path):
        path=tracked(path);return pd.read_parquet(path) if path.suffix=='.parquet' else pd.read_csv(path,dtype={'example_id':str})
    tracked(__file__);test=checks();(out/'checks.json').write_text(json.dumps(test,indent=2))
    raw=read(SOURCE/'paired_results.parquet');base_rows=raw[raw.method.eq('cunge')].copy();groups=read(AUDIT/'groups.csv');foot=read(AUDIT/'modification_footprint_audit.csv')
    responses=read(MODELS/'matched_features_responses.parquet')
    targetcols=['example_id','fingerprint','method','peak_excess_change_pct','event_duration_change_pct']
    responses=responses.merge(raw[targetcols],on=['example_id','fingerprint','method'],validate='one_to_one')
    sources=json.loads(tracked(SOURCE/'provenance.json').read_text())['sources']
    features=[];states=[];geometry_audit=[];forcing=[];dictionary={};network_scales={}
    for src in sources:
        ex=src['example'];root=Path(src['source']);base=read(root/'states'/src['base_fingerprint']/'parameters.parquet')
        net_p,net_meta=describe(base);assert net_meta['status']=='ok';wave=read(root/'common_forcing.parquet')
        t=wave.time_seconds.to_numpy(float);q=wave.q_inflow_cms.to_numpy(float);excess=np.maximum(q-q[0],0);area=np.trapz(excess,t);assert area>0
        center=np.trapz(t*excess,t)/area;event_T=float(np.sqrt(np.trapz((t-center)**2*excess,t)/area));assert event_T>0
        onset=np.flatnonzero(excess>.01*excess.max())[0];rise=float(t[q.argmax()]-t[onset])
        forcing.append(dict(example_id=ex,event_T_s=event_T,event_rise_s=rise,forcing_peak=q.max(),forcing_base=q[0],forcing_excess_volume=area,forcing_sigma_definition='SD of excess-inflow volume in time above initial flow',network_route_length_m=net_p['route_length_mean'],network_total_length_m=net_p['total_length']))
        Lstar=net_p['route_length_mean'];network_scales[ex]=(Lstar,event_T);group=base_rows[base_rows.example_id.eq(ex)];pcache={}
        source_paths={}
        for r in group.itertuples():
            childpath=root/'states'/r.fingerprint/'parameters.parquet';child=read(childpath)
            parent_ids=parse_ids(r.matched_base_link_ids);child_ids=parse_ids(r.child_link_ids)
            assert parent_ids<=set(base.id_link) and child_ids<=set(child.id_link)
            key=tuple(sorted(parent_ids))
            if key not in pcache:pcache[key]=describe(base[base.id_link.isin(parent_ids)])
            dom_p,mp=pcache[key];dom_c,mc=describe(child[child.id_link.isin(child_ids)]);net_c,nc=describe(child)
            assert mp['status']==mc['status']==nc['status']=='ok'
            gr=groups[(groups.example_id==ex)&(groups.fingerprint==r.fingerprint)].iloc[0]
            support=parse_ids(gr.footprint_ids);sel,sm=describe(base[base.id_link.isin(support)],False)
            # Actual exposure from global reference allocation, not forced to one.
            def exposure(f,ids,meta):
                return float(f[f.id_link.isin(ids)&f.id_us_node.eq(meta['source'])].reference_flow_fraction.sum())
            ep=exposure(base,parent_ids,mp);ec=exposure(child,child_ids,mc)
            row=dict(example_id=ex,fingerprint=r.fingerprint,state_id=r.state_id,group=gr.group,unit_type=r.target_type,domain_type=r.hydraulic_domain_type,event_T_s=event_T,network_route_length_m=Lstar,selection_support_length_fraction=sel['total_length']/net_p['total_length'],domain_length_fraction=dom_p['total_length']/net_p['total_length'],domain_inlet_exposure_parent=ep,domain_inlet_exposure_child=ec,delta_domain_inlet_exposure=ec-ep,resistance_log=r.resistance_log,selection_domain_ratio=sel['total_length']/dom_p['total_length'],upstream_distance_over_network=r.upstream_shortest_length_m/Lstar,downstream_distance_over_network=r.downstream_shortest_length_m/Lstar)
            for scope,pp,cc in [('domain',dom_p,dom_c),('network',net_p,net_c)]:
                for key,(le,te,family,definition) in METRICS.items():
                    scale=Lstar**le*event_T**te
                    for role,value in [('parent',pp[key]),('child',cc[key]),('delta',cc[key]-pp[key])]:
                        name=f'{scope}_{role}_{key}';row[name]=value
                        dictionary[name]=dict(feature=name,scope=scope,role=role,family=family,kind='hydraulic_Cunge_reference' if key in HYDRAULIC else 'geometry_with_width_allocation' if key.startswith(('route_','branch_','split_','dominance','effective_','expected_','area_elongation','sinuosity')) else 'geometry',L_exponent=le,T_exponent=te,dimensionless=le==te==0,definition=definition if role!='delta' else 'Child minus parent: '+definition,normalization='none',default_candidate=role!='child' and (scope=='domain' or role=='delta'))
                        if le!=0 or te!=0:
                            nd=name+'_nd';row[nd]=value/scale;dictionary[nd]={**dictionary[name],'feature':nd,'L_exponent':0,'T_exponent':0,'dimensionless':True,'normalization':f'base_network_route_length^{le} * excess_inflow_sigma^{te}','default_candidate':dictionary[name]['default_candidate']};dictionary[name]['default_candidate']=False
                    # Ratios are available for exploration, but not alongside delta+parent by default.
                    if abs(pp[key])>1e-14:
                        name=f'{scope}_ratio_{key}';row[name]=cc[key]/pp[key]
                    else:name=f'{scope}_ratio_{key}';row[name]=np.nan
                    dictionary[name]={**dictionary[f'{scope}_parent_{key}'],'feature':name,'role':'ratio','L_exponent':0,'T_exponent':0,'dimensionless':True,'normalization':'child/parent; undefined when parent zero','default_candidate':False}
            # Selection support has no certified child correspondence: parent geometric scope only.
            for key in ['total_length','bifurcations','cycle_rank','width_cv','dominance_mean','dominance_sd','split_impact_mean','split_impact_sd','route_length_cv']:
                name='selection_parent_'+key;row[name]=sel[key];le,te,family,definition=METRICS[key]
                dictionary[name]=dict(feature=name,scope='selected_parent_support',role='parent',family=family,kind='selection_proxy_not_actual_edit',L_exponent=le,T_exponent=te,dimensionless=le==te==0,definition=definition,normalization='none',default_candidate=False)
            row['exposed_delta_route_length']=ep*(dom_c['route_length_mean']-dom_p['route_length_mean'])/Lstar
            row['exposed_delta_route_time']=ep*(dom_c['route_time_mean']-dom_p['route_time_mean'])/event_T
            row['exposed_delta_route_time_variance']=ep*(dom_c['route_time_sd']**2-dom_p['route_time_sd']**2)/event_T**2
            row['exposed_delta_within_dispersion']=ep*(dom_c['within_route_dispersion']-dom_p['within_route_dispersion'])/event_T**2
            row['parent_advection_event_ratio']=dom_p['route_time_mean']/event_T
            # Chosen effective quantities define an inspectable domain-scale Pe proxy.
            cp=dom_p['route_length_mean']/dom_p['route_time_mean'];dp=dom_p['diffusivity_mean'];lp=dom_p['route_length_mean']
            row['parent_peclet_proxy']=cp*lp/dp
            row['parent_fourier_proxy']=dp*event_T/lp**2
            row['parent_courant_event_proxy']=cp*event_T/lp # event scale, NOT numerical Courant
            np.testing.assert_allclose(row['parent_peclet_proxy']*row['parent_fourier_proxy'],row['parent_courant_event_proxy'])
            features.append(row)
            for role,f,meta in [('parent',dom_p,mp),('child',dom_c,mc)]:states.append(dict(example_id=ex,fingerprint=r.fingerprint,state_id=r.state_id,role=role,**f,**meta))
            geometry_audit.append(dict(example_id=ex,state_id=r.state_id,parent_domain_links=len(parent_ids),child_domain_links=len(child_ids),parent_exposure=ep,child_exposure=ec,selection_status=sm['status'],parent_cycle_identity_error=mp['cycle_identity_error'],child_cycle_identity_error=mc['cycle_identity_error']))
        print('Extracted',ex,len(group),'parent-child pairs',flush=True)
    f=pd.DataFrame(features);assert len(f)==552 and not f.duplicated(['example_id','fingerprint']).any()
    # Register all standalone candidates and physical context scales.
    for name in f.columns:
        if name in dictionary or name in ['example_id','fingerprint','state_id','group','unit_type','domain_type']:continue
        dim=(0,1) if name=='event_T_s' else (1,0) if name=='network_route_length_m' else (0,0)
        hydraulic=any(s in name for s in ['time','dispersion','advection','peclet','fourier','courant','resistance'])
        dictionary[name]=dict(feature=name,scope='paired_context',role='context',family='hydraulics' if hydraulic else 'exposure_position',kind='hydraulic_Cunge_reference' if hydraulic else 'geometry_with_width_allocation',L_exponent=dim[0],T_exponent=dim[1],dimensionless=dim==(0,0),definition={'parent_peclet_proxy':'c_eff*L_domain / length-weighted diffusivity; c_eff=L_route/tau_route','parent_fourier_proxy':'length-weighted diffusivity*event_T/L_domain squared','parent_courant_event_proxy':'c_eff*event_T/L_domain; event transport ratio, not numerical Courant','exposed_delta_route_length':'parent domain inlet exposure * delta mean route length / base-network mean route length','exposed_delta_route_time':'parent domain inlet exposure * delta mean route advective time / event_T','exposed_delta_route_time_variance':'parent domain inlet exposure * delta between-route time variance / event_T squared','exposed_delta_within_dispersion':'parent domain inlet exposure * delta within-route dispersion / event_T squared'}.get(name,name.replace('_',' ')),normalization='see definition / scales',default_candidate=dim==(0,0) and name not in ['parent_fourier_proxy','parent_courant_event_proxy','parent_advection_event_ratio','domain_inlet_exposure_child'])
    catalog=pd.DataFrame(dictionary.values());assert set(catalog.feature)==set(f.columns)-{'example_id','fingerprint','state_id','group','unit_type','domain_type'}
    response_cols=['example_id','fingerprint','method','shift_h','relative_shift_pct','parent_lag_h','n_supplied','peak_excess_change_pct','event_duration_change_pct']
    data=f.merge(responses[response_cols],on=['example_id','fingerprint'],validate='one_to_many');assert len(data)==2208
    data['shift_over_event_T']=data.shift_h*3600/data.event_T_s
    # Hydraulic predictors use Cunge reference states, never substitute them as
    # fixed-X hydraulic predictors. Geometry-only cross-method comparisons remain valid.
    data.to_parquet(out/'paired_candidates.parquet',index=False);catalog.to_csv(out/'feature_catalog.csv',index=False)
    pd.DataFrame(states).to_parquet(out/'matched_domain_states.parquet',index=False);pd.DataFrame(geometry_audit).to_csv(out/'domain_audit.csv',index=False);pd.DataFrame(forcing).to_csv(out/'forcing_scales.csv',index=False)
    dim,basis=dimensional_basis();dim.to_csv(out/'dimension_matrix.csv');basis.to_csv(out/'pi_basis.csv')
    assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in hashes.items())
    (out/'provenance.json').write_text(json.dumps(dict(source_hashes=hashes,source_unchanged=True,pairs=552,routing_methods=4,features=len(catalog),scope='Saved parameters and responses only; no rerouting, calibration, production or prior snapshot writes. Cunge hydraulic descriptors are conditional mechanistic diagnostics. Ratios with zero parent are missing, not regularized.',checks=test),indent=2))
    print('COMPLETE',out,flush=True);return out

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out');args=parser.parse_args();build(args.out)
