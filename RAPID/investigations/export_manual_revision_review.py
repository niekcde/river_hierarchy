"""Export current versus revised geometry views and a small review landing page."""
from pathlib import Path
import sys,json
import pandas as pd
import geopandas as gpd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'RAPID/src'))
from rapid_tools import collapse_creation_review as review
from rapid_tools.anchor_repair_trial import node_mapping
from rapid_tools.boundary_contract_review import plot_comparison
OUT=ROOT/'RAPID/outputs/boundary_contract_repair_20261008/review_revision'

def run():
 figs=OUT/'figures';figs.mkdir(exist_ok=True);items=[]
 folders=list(OUT.glob('[0-9][0-9]/*/trial_links.parquet'))+list((OUT/'island_boundary_trials').glob('*/trial_links.parquet'))
 for path in folders:
  st=path.parent.name;ex='41' if st.startswith('island_') else path.parent.parent.name;row=review.case_row(ex,st);co=json.loads((path.parent/'contract.json').read_text());f=pd.read_parquet(row.frame_path)
  before=review.vectors(row.child_links_path,'links')[['id_link','geometry']].merge(f[['id_link','id_us_node','id_ds_node','u','v']],on='id_link');nodes=review.vectors(row.child_nodes_path,'nodes').copy();nodes['parent_identity']=nodes.id_node.map(node_mapping(f))
  after=gpd.read_parquet(path);an=gpd.read_parquet(path.parent/'trial_nodes.parquet');p=review.vectors(row.base_links_path,'links');pn=review.vectors(row.base_nodes_path,'nodes');required=co.get('reviewed_required_boundary_nodes',co['boundary_nodes'])
  result_path=path.parent/'result.json';result=json.loads(result_path.read_text()) if result_path.exists() else {}
  label='BLOCKED recovery candidate' if result.get('overall_geometry_pass') is False else 'revised trial'
  fig=plot_comparison(before,after,p,nodes,an,pn,required,co['selected_parent_links'],f'{ex} / {st}: parent · production · {label}',context_padding_m=150)
  target=figs/f'{ex}_{st}.html';fig.write_html(target,include_plotlyjs='directory');items.append((ex,st,target.name,label));print(ex,st,'exported',flush=True)
 body=''.join(f'<li><a href="figures/{name}">{ex} / {st}</a> — {label}</li>' for ex,st,name,label in items)
 (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Reviewed boundary revision</title><style>body{font:16px system-ui;max-width:1050px;margin:40px auto;line-height:1.6}li{margin:8px}</style><h1>Reviewed boundary revision</h1><p>Isolated trial outputs. Production geometry and RAPID results are unchanged. Geometry checks do not establish routing readiness.</p><p>The current notebook automatically loads the revised trials. Internal and provisional parent nodes are excluded from its required-boundary list, with the original list retained as evidence.</p><ul>'+body+'</ul><p><a href="summary.json">Outcome summary</a> · <a href="connection_flag_triage.csv">Nine connection flags</a> · <a href="island_creation_triage.csv">Island stage comparison</a> · <a href="island_duplicate_resolution_plan.csv">22 duplicate resolution decisions</a></p>')
if __name__=='__main__':run()
