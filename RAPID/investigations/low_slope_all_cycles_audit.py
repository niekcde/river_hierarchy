"""Read-only cycle support audit of all current parent links below 1e-5."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import linregress, theilslopes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'RAPID/assets/low_slope_all_cycles_20261002'; OUT.mkdir(parents=True,exist_ok=True)
rows=[]; windows=[]; provenance={}
for ex in ['11','13','15','21','22','41','46']:
    root=ROOT/'network_variants/output'/f'{ex}_indep{"_direction130" if ex=="15" else ""}__anchored/states/S000_base/hydraulic_context'
    m=json.loads((root/'hydraulic_context_manifest.json').read_text())
    p=Path(m['slope_refit_promotion']['reference_profile_dir']); provenance[ex]=str(p)
    f=pd.read_parquet(p/'reference_profile.parquet').sort_values('reference_section_dist_out_m')
    o=pd.read_parquet(p/'reference_profile_observations.parquet')
    n=pd.read_csv(root/'node_hydraulic_context.csv').set_index('id_node')
    d=pd.read_csv(root/'link_hydraulic_context.csv'); d=d[d.slope_used<1e-5]
    x=f.reference_section_dist_out_m.to_numpy();y=f.fitted_wse_m.to_numpy()
    requests=[]
    for _,r in d.iterrows():
        a,b=sorted([n.loc[r.id_us_node,'reference_section_dist_out_m'],n.loc[r.id_ds_node,'reference_section_dist_out_m']])
        for kind,lo,hi in [('exact',a,b),('expanded_1km_each_end',max(x.min(),a-1000),min(x.max(),b+1000))]:
            requests.append(dict(example=ex,link=int(r.id_link),window=kind,lo=lo,hi=hi,link_slope=r.slope_used,ownership=r.ownership_kind,physical_length_m=r.actual_length_m))
    if ex=='11':
        for a,b in [(0,44000),(0,40000),(0,20000),(20000,40000),(40000,44000)]:
            requests.append(dict(example=ex,link=-1,window=f'case11_{a//1000}_{b//1000}km',lo=a,hi=b,link_slope=np.nan,ownership='longitudinal_diagnostic',physical_length_m=np.nan))
    for w in requests:
        lo,hi=w['lo'],w['hi'];w['profile_endpoint_slope']=(np.interp(hi,x,y)-np.interp(lo,x,y))/(hi-lo) if hi>lo else np.nan
        windows.append(w)
        z=o[o.reference_section_dist_out_m.between(lo,hi)]
        for aid,g in z.groupby('acquisition_id',sort=False):
            q=g.groupby('reference_section_dist_out_m',as_index=False).wse_m.mean()
            if len(q)<3:continue
            xx=q.reference_section_dist_out_m.to_numpy(); yy=q.wse_m.to_numpy()
            fit=linregress(xx,yy); pred=np.interp(xx,x,y);rob=theilslopes(yy,xx).slope
            # Matched positions isolate profile-versus-observation differences from coverage.
            pf=linregress(xx,pred).slope; pr=theilslopes(pred,xx).slope
            coverage=np.ptp(xx)/(hi-lo) if hi>lo else 0
            rows.append(dict(**w,acquisition_id=aid,cycle=g.cycle.iloc[0],orbit_pass=g['pass'].iloc[0],n_positions=len(q),coverage=coverage,eligible=len(q)>=4 and coverage>=.5,ols=fit.slope,theil=rob,profile_ols=pf,profile_theil=pr,delta_ols=fit.slope-pf,delta_theil=rob-pr,r2=fit.rvalue**2))
    print(ex,'done',flush=True)
    if ex=='11':
        fig,axs=plt.subplots(2,1,figsize=(12,8),constrained_layout=True)
        for aid,g in o.groupby('acquisition_id',sort=False):
            q=g[g.reference_section_dist_out_m.between(0,44000)].sort_values('reference_section_dist_out_m')
            cyc=int(g.cycle.iloc[0]);lw=2.2 if cyc in [5,7] else .8
            axs[0].plot(q.reference_section_dist_out_m/1000,q.stage_aligned_wse_m,lw=lw,label=f'cycle {cyc}',alpha=.85)
            axs[1].plot(q.reference_section_dist_out_m/1000,q.wse_m-q.wse_m.mean(),lw=lw,label=f'cycle {cyc}',alpha=.85)
        q=f[f.reference_section_dist_out_m.between(0,44000)]
        axs[0].plot(q.reference_section_dist_out_m/1000,q.fitted_wse_m,'k',lw=3,label='Current profile')
        axs[1].plot(q.reference_section_dist_out_m/1000,q.fitted_wse_m-q.fitted_wse_m.mean(),'k',lw=3)
        axs[0].set(ylabel='Stage-aligned WSE (m)',title='Case 11: full downstream section (0–44 km)');axs[0].legend(ncol=4,fontsize=8)
        axs[1].set(ylabel='WSE minus trace mean (m)',title='Vertical offsets removed; slopes unchanged')
        for ax in axs:ax.set(xlabel='Reference-section distance to outlet (km)');ax.grid(alpha=.2)
        fig.savefig(OUT/'case11_0_44km.png',dpi=160);plt.close(fig)
r=pd.DataFrame(rows);r.to_csv(OUT/'acquisition_slopes.csv',index=False)
w=pd.DataFrame(windows);summ=[]
for _,v in w.iterrows():
    g=r[(r.example==v.example)&(r.link==v.link)&(r.window==v.window)&r.eligible]
    rec=v.to_dict();rec['n_eligible_acquisitions']=len(g);rec['n_cycles']=g.cycle.nunique()
    # Equal-cycle summaries prevent multiple orbit passes in one cycle dominating.
    g=g.groupby('cycle')[['ols','theil','profile_ols','profile_theil','delta_ols','delta_theil']].median()
    for name in ['ols','theil','profile_ols','profile_theil','delta_ols','delta_theil']:
        rec[name+'_median']=g[name].median();rec[name+'_q25']=g[name].quantile(.25);rec[name+'_q75']=g[name].quantile(.75)
    rec['fraction_both_steeper']=((g.delta_ols>1e-10)&(g.delta_theil>1e-10)).mean()
    rec['fraction_both_positive']=((g.ols>0)&(g.theil>0)).mean()
    rec['classification']='insufficient coverage' if len(g)<3 else ('consistent steeper support' if rec['fraction_both_steeper']>=.75 and rec['fraction_both_positive']>=.75 else 'mixed or no steeper support')
    summ.append(rec)
s=pd.DataFrame(summ);s.to_csv(OUT/'section_summary.csv',index=False)
(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
for ex,g in s[s.link!=-1].groupby('example'):
    g=g[g.window=='exact'];fig,ax=plt.subplots(figsize=(10,max(3,.3*len(g)+1.5)),constrained_layout=True)
    for i,(_,v) in enumerate(g.iterrows()):
        ax.plot([v.theil_q25*1e5,v.theil_q75*1e5],[i,i],color='tab:blue',lw=3)
        ax.plot(v.theil_median*1e5,i,'o',color='tab:blue')
        ax.plot(v.profile_theil_median*1e5,i,'x',color='black')
    ax.set(yticks=range(len(g)),yticklabels=[f'link {int(v.link)} (n={int(v.n_cycles)})' for _,v in g.iterrows()],xlabel='Projected longitudinal slope (cm/km): equal-cycle median/IQR blue; matched profile black',title=f'Case {ex}: exact projected spans of flagged links')
    ax.axvline(0,color='gray',lw=.8);ax.grid(axis='x',alpha=.2)
    fig.savefig(OUT/f'case{ex}_sections.png',dpi=150);plt.close(fig)
print(s.groupby(['window','classification']).size().to_string())
