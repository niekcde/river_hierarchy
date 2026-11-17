"""Plot saved outlier diagnostics and actual parent/child outlet records."""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from investigate_j_outlier_groups import DEST, SOURCE

COLORS={'G1 same-sign aggregate':'#d18c16','G2 same-sign member':'#1978b5','G3 opposite-sign local':'#bb3b78'}


def display_indices(x,y):
    """Keep 10-minute display samples plus both extrema of each 10-minute bin.

    All retained values are original samples; this does not affect any metrics.
    """
    bins=np.floor(np.asarray(x)*6).astype(int)
    f=pd.DataFrame({'bin':bins,'y':np.asarray(y)})
    idx=np.unique(np.r_[0,len(f)-1,f.groupby('bin',sort=False).head(1).index,
                        f.groupby('bin',sort=False).y.idxmin(),f.groupby('bin',sort=False).y.idxmax()])
    return idx


def overview():
    d=pd.read_parquet(DEST/'population_diagnostics.parquet')
    fig=go.Figure()
    groups=[('Not flagged','#bfc5cc'),*COLORS.items()]
    for group,color in groups:
        q=d[d.group.eq(group)]
        custom=q[['example','state','delta_J_minutes','centroid_shift_minutes','parent_transfer_sd_hours','transfer_sd_hours']].to_numpy()
        fig.add_trace(go.Scatter(x=q.sd_ratio,y=q.residual_z,mode='markers',name=group,
            marker=dict(color=color,size=5 if group=='Not flagged' else 10,opacity=.45 if group=='Not flagged' else .95),customdata=custom,
            hovertemplate='%{customdata[0]} / %{customdata[1]}<br>Spread ratio %{x:.3f}<br>Residual z %{y:.2f}<br>ΔJ %{customdata[2]:.2f} min; centroid %{customdata[3]:.2f} min<br>Transfer SD %{customdata[4]:.2f} → %{customdata[5]:.2f} h<extra>%{fullData.name}</extra>'))
    for y in [-3,3]:fig.add_hline(y=y,line_dash='dash',line_color='#777')
    fig.add_vline(x=1,line_dash='dot',line_color='#777')
    fig.update_layout(template='plotly_white',height=560,title='Manning ΔJ outliers: change in model transfer spread',
        xaxis_title='Child / parent analytic transfer standard deviation',yaxis_title='Residual z relative to the existing pooled ΔJ line',legend=dict(orientation='h',y=1.13))
    return fig


def wave_inspector():
    d=pd.read_csv(DEST/'outlier_diagnostics.csv',dtype={'example':str})
    # Start at a representative opposite-sign example, then preserve group order.
    d=d.sort_values(['group','example','state'])
    first=d.index[(d.example=='37')&(d.state=='island_37_282')][0]
    d=pd.concat([d.loc[[first]],d.drop(index=first)],ignore_index=True)
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.63,.37],vertical_spacing=.08)
    buttons=[]
    for i,r in enumerate(d.itertuples()):
        p=pd.read_parquet(SOURCE/'states'/r.example/'S000_base/outlet.parquet')
        c=pd.read_parquet(SOURCE/'states'/r.example/r.state/'outlet.parquet')
        onset=pd.Timestamp(r.onset_utc);end=pd.Timestamp(r.end_utc);duration=(end-onset).total_seconds()/3600
        tp=(p.time_utc-onset).dt.total_seconds().to_numpy()/3600;tc=(c.time_utc-onset).dt.total_seconds().to_numpy()/3600
        # Full saved duration, reduced for display only; no translation or timing recalculation.
        pi=display_indices(tp,p.q_outlet_cms);ci=display_indices(tc,c.q_outlet_cms)
        fig.add_trace(go.Scatter(x=tp[pi],y=p.q_outlet_cms.iloc[pi],name='Parent',line=dict(color='#333',width=2),visible=i==0),row=1,col=1)
        fig.add_trace(go.Scatter(x=tc[ci],y=c.q_outlet_cms.iloc[ci],name='Child',line=dict(color=COLORS[r.group],width=2),visible=i==0),row=1,col=1)
        valid=(tc>=tp.min())&(tc<=tp.max())
        dq=c.q_outlet_cms.to_numpy()[valid]-np.interp(tc[valid],tp,p.q_outlet_cms)
        di=display_indices(tc[valid],dq)
        fig.add_trace(go.Scatter(x=tc[valid][di],y=dq[di],name='Child − parent',line=dict(color=COLORS[r.group]),visible=i==0),row=2,col=1)
        shapes=[dict(type='rect',xref='x',yref='paper',x0=0,x1=duration,y0=0,y1=1,fillcolor='#eeeeee',opacity=.4,line_width=0,layer='below')]
        for center,color in [(r.parent_centroid_seconds,'#333'),(r.child_centroid_seconds,COLORS[r.group])]:
            x=(center-onset.timestamp())/3600
            shapes.append(dict(type='line',xref='x',yref='paper',x0=x,x1=x,y0=.43,y1=1,line=dict(color=color,dash='dot',width=1)))
        title=f'{r.example} / {r.state} — ΔJ {r.delta_J_minutes:+.2f} min; centroid {r.centroid_shift_minutes:+.2f} min<br><sup>{r.group}; transfer SD {r.parent_transfer_sd_hours:.2f} → {r.transfer_sd_hours:.2f} h; grey = response window</sup>'
        layout={'title.text':title,'shapes':shapes,'xaxis.range':[-6,duration+6],'xaxis2.range':[-6,duration+6],'yaxis.autorange':True,'yaxis2.autorange':True}
        visible=[False]*(3*len(d));visible[3*i:3*i+3]=[True]*3
        buttons.append(dict(label=f'{r.example} / {r.state}',method='update',args=[{'visible':visible},layout]))
        if i==0:fig.update_layout(**layout)
    fig.add_hline(y=0,line_width=1,line_color='#aaa',row=2,col=1)
    fig.update_layout(template='plotly_white',height=730,margin=dict(t=160),legend=dict(orientation='h',y=1.04),
        updatemenus=[dict(buttons=buttons,x=0,y=1.24,xanchor='left',yanchor='top')])
    fig.update_yaxes(title_text='Outlet discharge (m³/s)',row=1,col=1)
    fig.update_yaxes(title_text='Difference (m³/s)',row=2,col=1)
    fig.update_xaxes(title_text='Hours since the original event-window onset (double-click to show full saved record)',row=2,col=1)
    return fig


def main():
    overview().write_html(DEST/'outlier_spread.html',include_plotlyjs=True)
    wave_inspector().write_html(DEST/'outlier_hydrographs.html',include_plotlyjs=True)
    print('Saved both interactive figures to',DEST)

if __name__=='__main__':main()
