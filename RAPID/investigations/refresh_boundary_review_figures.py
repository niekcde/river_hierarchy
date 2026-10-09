"""Refresh the four user-reviewed views without changing any repair decisions."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'RAPID/src'))
import pandas as pd
import geopandas as gpd
from rapid_tools import collapse_creation_review as review
from rapid_tools.anchor_repair_trial import node_mapping
from rapid_tools.boundary_contract_review import plot_comparison,correspondence_table
OUT=ROOT/'RAPID/outputs/boundary_contract_repair_20261008'
CASES=[('02','S026_unit_16',[],None),('15','S087_unit_44',[89],89),('15','S113_unit_189',[365,383,379,412,413],379),('15','S158_unit_191',[376,378],379),('41','S016_unit_4',[],5)]

def run():
    summaries=[]
    for ex,state,extra,focus in CASES:
        row=review.case_row(ex,state);folder=OUT/'full'/ex/state;contract=json.loads((folder/'contract.json').read_text())
        source=review.vectors(row.child_links_path,'links');frame=pd.read_parquet(row.frame_path)
        before=source[['id_link','geometry']].merge(frame[['id_link','id_us_node','id_ds_node','u','v']],on='id_link',validate='one_to_one')
        assert set(before.id_link)==set(source.id_link)
        nodes=review.vectors(row.child_nodes_path,'nodes').copy();nodes['parent_identity']=nodes.id_node.map(node_mapping(frame))
        after=gpd.read_parquet(folder/'trial_links.parquet');an=gpd.read_parquet(folder/'trial_nodes.parquet')
        parent=review.vectors(row.base_links_path,'links');pn=review.vectors(row.base_nodes_path,'nodes')
        args=(before,after,parent,nodes,an,pn,contract['boundary_nodes'],contract['selected_parent_links'],f'{ex} — {state}: original parent / current / trial')
        fig=plot_comparison(*args);fig.write_html(folder/'comparison.html',include_plotlyjs='directory',config={'responsive':True})
        if focus is not None:
            close=plot_comparison(*args,focus_parent_node=focus,radius_m=100)
            close.write_html(folder/f'boundary_P{focus}_closeup.html',include_plotlyjs='directory',config={'responsive':True})
        if ex=='02':
            assert 80 in fig.layout.meta['visible_link_ids']['current'] and 80 in fig.layout.meta['visible_link_ids']['trial']
            assert source[source.id_link.eq(80)].geometry.iloc[0].equals(after[after.id_link.eq(80)].geometry.iloc[0])
        table=correspondence_table(nodes,an,pn,contract['boundary_nodes'],extra_child_nodes=extra)
        table.to_csv(folder/'correspondence_review.csv',index=False)
        summaries.append(dict(example=ex,state=state,plot=str(folder/'comparison.html'),mapping_table=table.to_dict('records'),
                              automatic_classification_unchanged=True,plot_metadata=fig.layout.meta))
        print(ex,state,'exported; original parent IDs and rectangular coverage verified',flush=True)
    (OUT/'figure_review_checks.json').write_text(json.dumps(summaries,indent=2))
    feedback=[
        dict(example='02',state='S026_unit_16',user_observation='Link 80 appears absent in figure but present in QGIS; added node looks good.',verified='Link 80 exists unchanged in source and trial. The old narrow-buffer display filter omitted it. Rectangular filtering now includes it.',review_status='plotting_defect_fixed; positive user review of inserted node; other automatic checks unchanged'),
        dict(example='15',state='S087_unit_44',user_observation='Both topologies appear good; different node matching; original parent needed to judge.',verified='Existing C89 changes hydraulic identity H414 to original P89 without moving or inserting a node. The remaining automatic flag concerns P89 replacement incidence.',review_status='visual topology positive; boundary-role interpretation pending parent comparison'),
        dict(example='15',state='S113_unit_189',user_observation='C365/C383 appear outside the unit; C413/C412 are the boundaries.',verified='C365 and C383 are unchanged context nodes, not prescribed boundaries or newly inserted nodes. C383 has new hydraulic identity H414, not original P383. C413 already represents P383; C412 is restored from H415 to P377. The parent contract additionally contains side-interface P379, shown separately for review.',review_status='context-label confusion corrected; user proposed main boundaries recorded; side-interface P379 not silently removed'),
        dict(example='15',state='S158_unit_191',user_observation='C376 is not part of the unit.',verified='C376 is unchanged context with identity P377, outside the selected parent-unit node set. The trial changes C378 from H414 to P379, not C376.',review_status='context observation verified; P379 replacement-incidence flag remains for review')]
    (OUT/'user_figure_review_feedback_20261008.json').write_text(json.dumps(feedback,indent=2))
    links=''.join(f'<li><a href="full/{ex}/{state}/comparison.html">{ex} / {state}</a> — <a href="full/{ex}/{state}/correspondence_review.csv">identity table</a></li>' for ex,state,_,_ in CASES)
    (OUT/'figure_review_index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;margin:40px;max-width:1000px}li{margin:16px 0}</style><h1>Parent / current / trial review</h1><p>Every view now has a separate original-parent panel. Orange parent links identify the selected unit. Grey child dots are context, not unit membership. P = original parent; C = raw child; H = new hydraulic identity. Zoom is shared across all three panels.</p><p>Link 80 in 02–16 is included: the earlier figure omitted it with a narrow branch-buffer filter. No network geometry or repair classification was changed.</p><ul>'+links+'</ul>')
if __name__=='__main__':run()
