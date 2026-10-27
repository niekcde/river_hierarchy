"""Read-only notebook views for current hydraulic comparison outputs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .hydraulics import LABELS
from .networks import make_states

ROOT=Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT=ROOT/'synthetic_experiments/outputs/depth_methods_20261027'


def load(output=DEFAULT_OUTPUT):
    output=Path(output)
    if not (output/'completion.json').exists():raise FileNotFoundError('Run the comparison first; no complete results at '+str(output))
    manifest=json.loads((output/'manifest.json').read_text());done=json.loads((output/'completion.json').read_text())
    if done['source']!=manifest['source']:raise ValueError('Results/source mismatch')
    df=pd.read_csv(output/'responses.csv')
    if len(df)!=done['n_comparisons']:raise ValueError('Incomplete response table')
    return manifest,df


def comparison_figure(results, family='simple', child='simple/full'):
    selected=results[(results.family==family)&(results.child_state==child)]
    if selected.empty:raise ValueError('No selected comparison')
    if selected['case'].nunique()!=1:raise ValueError('Select one case before plotting')
    fig,axes=plt.subplots(1,2,figsize=(11,3.7),constrained_layout=True)
    for method,g in selected.groupby('method',sort=False):
        g=g.sort_values('branch_width_factor')
        for ax,column,label in zip(axes,['centroid_shift_minutes','peak_shift_minutes'],['Full-event centroid shift (min)','Peak shift (min)']):
            ax.plot(g.branch_width_factor,g[column],'o-',label=LABELS[method]);ax.set(xlabel='Total branch-width factor',ylabel=label)
            ax.axhline(0,color='gray',lw=.7);ax.grid(alpha=.15)
    axes[1].legend(fontsize=8);fig.suptitle(f'{child}: child minus parent; negative = earlier')
    return fig


def hydrograph_figure(output,case='equal_split',width_factor=1.2,parent='simple/original',child='simple/full'):
    manifest,_=load(output);fig,axes=plt.subplots(1,3,figsize=(14,3.5),constrained_layout=True)
    jobs=[j for j in manifest['jobs'] if j['case']==case and np.isclose(j['parameters']['branch_width_factor'],width_factor)]
    for ax,job in zip(axes,jobs):
        g=pd.read_parquet(Path(output)/'groups'/job['id']/'hydrographs.parquet')
        s=job['settings'];end=s['spinup_hours']+s['rise_hours']+s['fall_hours']+48;g=g[g.time_seconds<=end*3600]
        ax.plot(g.time_seconds/3600,g.inflow_cms,color='gray',ls=':',label='Input')
        ax.plot(g.time_seconds/3600,g[parent],label='Parent');ax.plot(g.time_seconds/3600,g[child],ls='--',label='Child')
        ax.set(title=LABELS[s['method']],xlabel='Time since simulation start (h)',ylabel='Discharge (m³/s)');ax.legend(fontsize=8)
    return fig


def geometry_figure(parameters,family='simple',child='simple/full'):
    states,_=make_states(parameters);fig,axes=plt.subplots(1,2,figsize=(12,3),constrained_layout=True)
    for ax,key in zip(axes,[family+'/original',child]):
        g=states[key]
        for u,v,k,d in g.edges(keys=True,data=True):
            x=np.array(d['plot_x'])/1000;y=np.array(d['plot_y'])
            # Endpoint-only loop traces overlap: give parallel edges schematic offsets.
            parallel=list(g[u][v]);offset=(parallel.index(k)-(len(parallel)-1)/2)*1.3
            xx=np.linspace(x[0],x[-1],50);yy=np.linspace(y[0],y[-1],50)+offset*np.sin(np.linspace(0,np.pi,50))
            ax.plot(xx,yy,lw=1+6*d['w']/parameters['total_width_m'])
            ax.annotate(f"{d['w']:.0f} m",(xx[25],yy[25]),xytext=(0,5),textcoords='offset points',ha='center',fontsize=8)
        L=parameters['total_length_m']/1000
        ax.axvspan(0,L*.25,color='#f1f5f9',zorder=-1);ax.axvspan(L*.75,L,color='#f1f5f9',zorder=-1)
        ax.set(title=key,xlabel='Downstream position (km)',yticks=[])
        ax.margins(y=.25)
    fig.suptitle(f"Reference inlet/exit width {parameters['total_width_m']:g} m; test-section factor {parameters.get('branch_width_factor',1):g}")
    return fig
