"""Plot saved out-of-fold predictions; no refitting or repeat averaging."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'RAPID/outputs/uncapped_event_window_20261020/hydrography_rf_compound_20261029'
OUT = SOURCE/'figures'
OUT.mkdir(exist_ok=True)
p = pd.read_parquet(SOURCE/'predictions.parquet')
p = p[p.model.eq('geometry_normalized_9') & p.cohort.eq('original')].copy()
assert len(p) == 2170*5 and not p.duplicated(['repeat','example','state']).any()
p['modelled_min'] = p.centroid_shift_hours*60
p['predicted_min'] = p.predicted_hours*60
s = pd.read_csv(SOURCE/'summary.csv')
s = s[s.model.eq('geometry_normalized_9') & s.cohort.eq('original')].iloc[0]
def transform(x):
    x = np.asarray(x)
    return np.sign(x)*np.log10(1+np.abs(x))
x,y = transform(p.modelled_min),transform(p.predicted_min)
lo,hi = min(x.min(),y.min()),max(x.max(),y.max())
pad = (hi-lo)*.05
limits = (lo-pad,hi+pad)
ticks = np.array([-500,-200,-100,-50,-20,-10,-5,-2,-.5,0,.5,2,5,10,20,50,100,200,500])
ticks = ticks[(transform(ticks)>=limits[0]) & (transform(ticks)<=limits[1])]
fig,ax = plt.subplots(figsize=(8.5,8.2))
ax.scatter(x,y,s=9,c='#243b53',alpha=.18,linewidths=0,rasterized=True)
ax.plot(limits,limits,'--',color='#777777',lw=1.2,zorder=0)
ax.axhline(0,color='#dddddd',lw=.8,zorder=0);ax.axvline(0,color='#dddddd',lw=.8,zorder=0)
ax.set(xlim=limits,ylim=limits,xlabel='RAPID-modelled centroid shift (minutes)',ylabel='RF-predicted centroid shift (minutes)')
ax.set_xticks(transform(ticks),[f'{t:g}' for t in ticks],rotation=45)
ax.set_yticks(transform(ticks),[f'{t:g}' for t in ticks])
ax.set_aspect('equal',adjustable='box')
ax.spines[['top','right']].set_visible(False)
ax.text(.035,.96,f'Mean across 5 repeats\n$R^2$ = {s.r2_mean:.3f}\nMAE = {s.mae_minutes_mean*60:.1f} seconds',transform=ax.transAxes,va='top',fontsize=11,bbox=dict(facecolor='white',alpha=.92,edgecolor='none',pad=7))
fig.suptitle('Hydrography-only RF — predicted versus modelled timing',fontsize=15,y=.97)
ax.set_title('Normalized spread change + entropy change · grouped-compound validation',fontsize=10,pad=14)
fig.text(.5,.035,'2,170 individual collapses; five held-out predictions per case (10,850 points).\nSigned-log axes; labels show minutes. Negative = advance; positive = delay.\nFixed-Manning responses, finite 0.5% event window. Dashed line = perfect prediction.',ha='center',fontsize=9,color='#444444')
fig.subplots_adjust(bottom=.21,top=.87,left=.13,right=.96)
fig.savefig(OUT/'hydrography_rf_prediction_vs_modelled.png',dpi=200)
fig.savefig(OUT/'hydrography_rf_prediction_vs_modelled.svg')
plt.close(fig)
interactive=go.Figure()
for rep,g in p.groupby('repeat'):
    custom=g[['example','state','edit_type','modelled_min','predicted_min']].to_numpy()
    interactive.add_trace(go.Scattergl(x=transform(g.modelled_min),y=transform(g.predicted_min),mode='markers',name=f'Repeat {rep+1}',marker=dict(size=5,color='#243b53',opacity=.22),customdata=custom,
        hovertemplate='Network %{customdata[0]} · %{customdata[1]}<br>%{customdata[2]}<br>RAPID: %{customdata[3]:.3f} min<br>RF: %{customdata[4]:.3f} min<extra>%{fullData.name}</extra>'))
interactive.add_shape(type='line',x0=limits[0],x1=limits[1],y0=limits[0],y1=limits[1],line=dict(color='grey',dash='dash'),layer='below')
axis=dict(tickmode='array',tickvals=transform(ticks),ticktext=[f'{t:g}' for t in ticks],range=limits,zeroline=True,zerolinecolor='#dddddd')
interactive.update_xaxes(**axis,title='RAPID-modelled centroid shift (minutes)')
interactive.update_yaxes(**axis,title='RF-predicted centroid shift (minutes)',scaleanchor='x',scaleratio=1)
interactive.update_layout(template='plotly_white',width=950,height=850,title=dict(text=f'Hydrography-only RF — grouped-compound validation<br><sup>2,170 individual collapses × 5 repeats; mean R² {s.r2_mean:.3f}, MAE {s.mae_minutes_mean*60:.1f} s. Signed-log axes.</sup>'),legend=dict(title='Click to hide; double-click to isolate'),margin=dict(b=110))
interactive.add_annotation(x=.5,y=-.13,xref='paper',yref='paper',showarrow=False,text='Finite 0.5% event-window centroid shifts · Negative = advance; positive = delay.<br>Five separate held-out predictions per case; no ensemble averaging. Scores are means across repeats.')
interactive.write_html(OUT/'hydrography_rf_prediction_vs_modelled.html',include_plotlyjs=True)
(OUT/'plot_metadata.json').write_text(json.dumps(dict(model='geometry_normalized_9',cohort='original',cases=2170,points=len(p),repeats=5,averaged_predictions=False,r2_mean=s.r2_mean,mae_minutes_mean=s.mae_minutes_mean),indent=2)+'\n')
print(OUT)
