"""Reference-flow coordinates of completed real base networks; no rerouting."""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import networkx as nx
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'RAPID/src'))
from rapid_tools.refreshed_analysis import DEFAULT_REFRESH_ROOT
OUT=ROOT/'RAPID/validation/current_hydraulic_domain_20261005'
SYNTH=ROOT/'synthetic_experiments/outputs/hydraulic_alpha065_20260930/hydraulic_applicability'


def coordinates():
    rows=[];sources=[]
    for folder in sorted(DEFAULT_REFRESH_ROOT.iterdir()):
        if not (folder/'inventory.csv').exists():continue
        inv=pd.read_csv(folder/'inventory.csv');r=inv[(inv.state=='S000_base') & inv.status.eq('ready')].iloc[0]
        f=pd.read_parquet(r.frame);run=folder/r.fingerprint/'run'
        h=pd.read_csv(run/'outlet_hydrograph.csv');m=json.loads((run/'hydrograph_metrics.json').read_text())
        t=h.time_seconds.to_numpy();q=h.q_inflow_cms.to_numpy();peak=int(q.argmax());level=q[0]+.01*(q.max()-q[0])
        j=int(np.flatnonzero(q[:peak+1]>level)[0]);assert j>0
        onset=t[j-1]+(t[j]-t[j-1])*(level-q[j-1])/(q[j]-q[j-1])
        rise=t[peak]-onset;local_rise=m['inflow_peak_time_seconds']-m['event_start_time_seconds'];assert rise>0 and local_rise>0
        graph=nx.DiGraph();graph.add_edges_from(zip(f.id_us_node,f.id_ds_node));roots=[n for n in graph if graph.in_degree(n)==0];ends=[n for n in graph if graph.out_degree(n)==0]
        assert len(roots)==len(ends)==1 and nx.is_directed_acyclic_graph(graph)
        flow={n:0. for n in graph};flow[roots[0]]=1.;shares={}
        for node in nx.topological_sort(graph):
            out=f[f.id_us_node.eq(node)]
            for edge in out.itertuples():
                share=flow[node]*edge.width/out.width.sum();shares[edge.id_link]=share;flow[edge.id_ds_node]+=share
        assert np.isclose(flow[ends[0]],1)
        np.testing.assert_allclose(f.rapid_celerity_mps_raw,(5/3)*f.depth_m**(2/3)*np.sqrt(f.slope)/.035,rtol=1e-12)
        sources.append(dict(example=folder.name,fingerprint=r.fingerprint,frame=r.frame,hydrograph=str(run/'outlet_hydrograph.csv'),rise_hours=rise/3600,legacy_local_rise_hours=local_rise/3600,base_q_cms=q[0],peak_q_cms=q.max()))
        for edge in f.itertuples():
            for scenario,qref in [('base',q[0]),('peak',q.max())]:
                allocated=shares[edge.id_link]*qref;area=edge.width*edge.depth_m;um=edge.rapid_celerity_mps_raw*3/5
                row=dict(example=folder.name,id_link=edge.id_link,reference_flow=scenario,length_m=edge.link_length_m,width_m=edge.width,depth_m=edge.depth_m,slope=edge.slope,flow_share=shares[edge.id_link],allocated_q_cms=allocated,raw_celerity_mps=edge.rapid_celerity_mps_raw,used_celerity_mps=edge.rapid_celerity_mps,capped=not np.isclose(edge.rapid_celerity_mps_raw,edge.rapid_celerity_mps),rise_hours=rise/3600,legacy_local_rise_hours=local_rise/3600,q_manning_over_allocated=um*area/allocated)
                for mode,u in [('allocated',allocated/area),('manning',um)]:
                    row[mode+'_velocity']=u;row[mode+'_f2']=u*u/(9.80665*edge.depth_m)
                    row[mode+'_tplus']=rise*u*edge.slope/edge.depth_m
                    row[mode+'_local_tplus']=local_rise*u*edge.slope/edge.depth_m
                rows.append(row)
    return pd.DataFrame(rows),sources


def main():
    OUT.mkdir(parents=True,exist_ok=True);d,sources=coordinates();assert np.isfinite(d.select_dtypes('number')).all().all()
    d.to_parquet(OUT/'reaches.parquet',index=False);summary=[]
    for cohort,g in [('all',d),*list(d.groupby('example'))]:
        for flow,z in g.groupby('reference_flow'):
            for mode in ['allocated','manning']:
                for proxy,col in [('first_1pct_to_peak',mode+'_tplus'),('legacy_local_min_to_peak',mode+'_local_tplus')]:
                    passed=z[col].ge(80)&z[mode+'_f2'].le(.9)
                    summary.append(dict(example=cohort,reference_flow=flow,velocity=mode,time_proxy=proxy,links=len(z),median_tplus=z[col].median(),median_f2=z[mode+'_f2'].median(),passing_links=int(passed.sum()),passing_length_fraction=z.loc[passed,'length_m'].sum()/z.length_m.sum(),median_q_manning_over_allocated=z.q_manning_over_allocated.median()))
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
    (OUT/'provenance.json').write_text(json.dumps(dict(reference='CURRENT-HYDRAULIC-DOMAIN-2026-10-05',sources=sources,scope='16 completed real base networks, canonical physical links; no child-domain or observed-flow validation',velocity='Qwidth/(W*h), separately raw Manning c*3/5; never capped c as water velocity',depth='Production BASED-anchored alpha=.65 depth; n=.035; saved final slope',time='First forcing 1% departure to first peak; legacy local-min-to-peak sensitivity. Neither is necessarily the perturbation period in the literature.',screen='Provisional repository Tplus>=80, F2<=.9, not full published boundary',old_new='No matched previous-parameterization reconstruction; historical gauge overlay is a different cohort and cannot establish improvement.',synthetic_source=str(SYNTH)),indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=pd.read_parquet(SYNTH/'reaches.parquet');fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    for i,scenario in enumerate(['base','peak']):
        for j,mode in enumerate(['allocated','manning']):
            ax=axes[i,j];a=s[s.reference_flow.eq(scenario)].drop_duplicates([mode+'_tplus',mode+'_f2']);b=d[d.reference_flow.eq(scenario)]
            ax.scatter(a[mode+'_tplus'],a[mode+'_f2'],s=7,alpha=.14,color='#2563eb',label='Selected synthetic grid')
            ax.scatter(b[mode+'_tplus'],b[mode+'_f2'],s=8,alpha=.3,color='#d97706',label='Current real base links')
            ax.set(xscale='log',yscale='log',xlabel='T⁺ (rise-time proxy)',ylabel='F²',title=f'{scenario.capitalize()} reference · {mode} velocity');ax.axvline(80,c='#15803d',ls='--');ax.axhline(.9,c='#15803d',ls=':')
            xmin,xmax=ax.get_xlim();ymin,ymax=ax.get_ylim();xmax=max(xmax,800);ymax=max(ymax,1.8);ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,ymax);ax.fill_between([80,xmax],ymin,.9,color='#dcfce7',zorder=0)
            ax.grid(alpha=.12);ax.legend(fontsize=8)
    fig.suptitle('Current hydraulic coordinates · provisional screen, not a model-validity boundary')
    fig.savefig(OUT/'current_domain.png',dpi=140);plt.close(fig)
    print(pd.DataFrame(summary).query("example=='all'").to_string(index=False))

if __name__=='__main__':main()
