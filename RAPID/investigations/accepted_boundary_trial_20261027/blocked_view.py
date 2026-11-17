from pathlib import Path
import sys,json,base64,io
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import pandas as pd,pyogrio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from network_variants.reviewed_geometry import ReviewedGeometry
r=ReviewedGeometry();OUT=ROOT/'RAPID/outputs/accepted_boundary_trial_20261027_v4';fig,axes=plt.subplots(1,2,figsize=(10,4.5),constrained_layout=True)
rows=[]
for ax,state,lid in [(axes[0],'S000_base',434),(axes[1],'S289_unit_143',979)]:
 rec=r.lookup('37','indep',state);f=pd.read_parquet(rec['original_frame']).set_index('id_link');g=pyogrio.read_dataframe(rec['links']).set_index('id_link');ns=pyogrio.read_dataframe(rec['nodes']).set_index('id_node');mapping=dict(zip(f.u,f.id_us_node))|dict(zip(f.v,f.id_ds_node));q=f.loc[lid];origin=ns.loc[mapping[286]].geometry;xx0,yy0=origin.x,origin.y
 for i,t in f[f.u.isin([286,287])|f.v.isin([286,287])].iterrows():
  line=g.loc[i].geometry;xx,yy=line.xy;color='#c04b39' if i==lid else '#a9b6c0';ax.plot([x-xx0 for x in xx],[y-yy0 for y in yy],c=color,lw=2.5 if i==lid else 1)
  if i==lid:
   us=ns.loc[mapping[t.u]].geometry;rev=us.distance(line.interpolate(0))>us.distance(line.interpolate(1,normalized=True));a=line.interpolate(.7 if rev else .3,normalized=True);b=line.interpolate(.3 if rev else .7,normalized=True);ax.annotate('',xy=(b.x-xx0,b.y-yy0),xytext=(a.x-xx0,a.y-yy0),arrowprops=dict(arrowstyle='->',lw=2,color=color))
 for v in [286,287]:
  pt=ns.loc[mapping[v]].geometry;ax.scatter(pt.x-xx0,pt.y-yy0,c='#222');ax.annotate(str(v),(pt.x-xx0,pt.y-yy0),xytext=(5,5),textcoords='offset points')
 ax.set_xlim(-85,85);ax.set_ylim(-85,85);ax.set_aspect('equal');ax.set_title(f'{"Reference" if state=="S000_base" else "Current"}: link {lid}, {int(q.u)} → {int(q.v)}');ax.set_xlabel('Local easting (m)');ax.set_ylabel('Local northing (m)')
buf=io.BytesIO();fig.savefig(buf,format='png',dpi=120);plt.close(fig);(OUT/'unit143_additional_reversal.png').write_bytes(buf.getvalue())
page='''<!doctype html><meta charset="utf-8"><title>Unit 143: additional exterior reversal</title><style>body{font:16px system-ui;max-width:1100px;margin:35px;line-height:1.5}img{width:100%}</style><h1>37 / unit 143: additional exterior reversal</h1><p>The accepted restoration of merged link 989 has been applied in the isolated geometry candidate. This is a different connection, outside that approved repair.</p><img src="data:image/png;base64,'''+base64.b64encode(buf.getvalue()).decode()+'''"><p>Original exterior reach 434 runs <b>287 → 286</b>. Current reach 979 runs <b>286 → 287</b> on the same 48.805 m line. Keeping reference endpoint heads gives a directed drop of <b>−0.039021 m</b> in the current direction.</p><p>The trial does not silently accept this extra exterior direction change or expand the released-node set. It retains the corrected geometry candidate but does not publish fitted heads/slopes for this case.</p><p>A follow-up option is to treat this connector like the two explicitly accepted reversed connectors and test minimal boundary flexibility. That option has not been authorised or fitted here. Alternatively, preserving the original direction requires checking the surrounding collapsed topology; flipping a connector in isolation is not assumed safe.</p><p><a href="review_results.html">Return to complete trial report</a></p>'''
(OUT/'unit143_additional_reversal.html').write_text(page)
