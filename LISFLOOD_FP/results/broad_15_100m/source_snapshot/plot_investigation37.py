"""Current MC, legacy-bed local runs and corrected-bed local runs for example 37."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import broad_study as b
from audit_current_predictor import MC,DIAG,OUT

def main():
    ex='37';cal=json.loads((b.OUT/ex/'calibration.json').read_text());tr=json.loads((b.OUT/ex/'transfer.json').read_text());local_base=pd.read_parquet(Path(cal['final']['folder'])/'wave.parquet');mc_base=pd.read_parquet(MC/'states'/ex/'S000_base'/'outlet.parquet');diag=pd.read_parquet(DIAG).set_index(['example','state'])
    fig=make_subplots(rows=2,cols=2,subplot_titles=['Smaller collapse: hydrographs','Smaller collapse: discharge change','Larger collapse: hydrographs','Larger collapse: discharge change'],vertical_spacing=.13)
    def add(w,name,color,dash,row,col):
        indices={0,len(w)-1};values=w.q_outlet_cms.to_numpy()
        for i in range(0,len(w),10):indices.update([i+int(np.argmin(values[i:i+10])),i+int(np.argmax(values[i:i+10]))])
        z=w.iloc[sorted(indices)];fig.add_trace(go.Scatter(x=z.time_utc,y=z.q_outlet_cms,mode='lines',name=name,legendgroup=name,showlegend=row==1,line=dict(color=color,dash=dash),hovertemplate='%{x}<br>%{y:.3f} m³/s<extra>%{fullData.name}</extra>'),row=row,col=col)
    def difference(a,z):
        x=a.time_utc.astype('int64').to_numpy();t=z.time_utc.astype('int64').to_numpy();mask=(t>=x[0])&(t<=x[-1]);d=z.loc[mask,['time_utc','q_outlet_cms']].copy();d['q_outlet_cms']-=np.interp(t[mask],x,a.q_outlet_cms);return d
    for row,state in enumerate(['S256_unit_229','S274_unit_280'],1):
        old=next(r for r in tr['runs'] if r['state']==state);local_old=pd.read_parquet(Path(old['folder'])/'wave.parquet');mc=pd.read_parquet(MC/'states'/ex/state/'outlet.parquet')
        add(mc_base,'Current MC original','#777777','dash',row,1);add(local_base,'Local original','#006bb6','solid',row,1)
        add(mc,'Current MC collapse','#222222','dash',row,1);add(local_old,'Local collapse: legacy bed','#c78135','dot',row,1)
        add(difference(mc_base,mc),'Current MC change','#222222','dash',row,2);add(difference(local_base,local_old),'Local change: legacy bed','#c78135','dot',row,2)
        path=OUT/('37_current_beds_'+state)/'result.json'
        if path.exists():
            r=json.loads(path.read_text())
            if r['valid']:
                w=pd.read_parquet(Path(r['folder'])/'wave.parquet');add(w,'Local collapse: current bed','#b72436','solid',row,1);add(difference(local_base,w),'Local change: current bed','#b72436','solid',row,2)
        d=diag.loc[(ex,state)]
        for col in [1,2]:
            fig.add_vrect(x0=d.onset_utc,x1=d.end_utc,fillcolor='gray',opacity=.08,line_width=0,row=row,col=col)
            fig.update_xaxes(range=[d.onset_utc,pd.Timestamp(d.end_utc)+pd.Timedelta(hours=36)],row=row,col=col)
        fig.add_hline(y=0,line_color='#aaaaaa',row=row,col=2)
    fig.update_yaxes(title_text='Discharge (m³/s)',col=1);fig.update_yaxes(title_text='Collapse − original (m³/s)',col=2);fig.update_xaxes(title_text='Time (UTC)')
    fig.update_layout(title='Example 37: separating MC version, bed profile and solver response',template='plotly_white',height=1100,legend=dict(orientation='h',y=-.22),margin=dict(b=220),hovermode='x unified')
    fig.add_annotation(text='Shading: predictor’s fixed event window. Corrected-bed curves appear only after their numerical run completes. All local runs retain the same fitted n.',x=0,y=-.14,xref='paper',yref='paper',showarrow=False)
    fig.write_html(OUT/'37_diagnostic_plotly.html',include_plotlyjs=True)

if __name__=='__main__':main()
