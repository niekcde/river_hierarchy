import pandas as pd
from rapid_tools.master_workflow import inventory_summary


def test_inventory_counts_alias_entries_separately_from_unique_states():
    frame=pd.DataFrame(dict(example=['02']*5+['11'],family=['indep','compound_bubbles','island','island','indep','indep'],
      status=['ready','ready','ready','excluded_duplicate','excluded_topology_review','ready'],fingerprint=['base','base','island','island','bad','eleven']))
    table=inventory_summary(frame,['02'])
    assert table.loc['02'].to_dict()==dict(indep=1,compound_bubbles=1,island=1,total_entries=3,unique_states=2)
