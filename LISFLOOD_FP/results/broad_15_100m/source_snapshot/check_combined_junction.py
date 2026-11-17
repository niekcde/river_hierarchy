"""Independent uniform-flow diagnostic for a simultaneous merge/split node.
Compare identical parallel channels with separate middle nodes versus a shared
middle node. Widths, bed slope, total discharge and physical storage are equal.
No solver/geometry changes are made in this diagnostic.
"""
import numpy as np
import pandas as pd
import broad_study as b

def network(shared):
 if shared:u=[0,1,1,2,2,3];v=[1,2,2,3,3,4]
 else:u=[0,1,1,2,3,4];v=[1,2,3,4,4,5]
 return pd.DataFrame(dict(id_link=range(6),u=u,v=v,width=[200.,100.,100.,100.,100.,200.],link_length_m=[200.]*6,head_us=[3.,2.8,2.8,2.6,2.6,2.4],head_ds=[2.8,2.6,2.6,2.4,2.4,2.2]))

if __name__=='__main__':
 rows=[];n=.025;depth=2.;unit=depth**(5/3)*np.sqrt(.001)/n;Q=200*unit
 for shared in [False,True]:
  f=network(shared)
  configs=[(length,.35) for length in [200,100,50,25]]+([(100,.175)] if shared else [])
  for length,cfl in configs:
   cells=b.v.grid(f,f'max{length}');m=b.build(cells,n);q=np.full(len(m['a']),unit)
   if shared:
    for k,(a,bb) in enumerate(zip(m['a'],m['b'])):
     if bb<len(cells) and cells.id_link.iloc[a] in [1,2] and cells.id_link.iloc[bb] in [3,4]:q[k]=unit/2
   balance=np.zeros(len(cells));np.add.at(balance,m['a'],-q*m['wf']);inside=m['b']<len(cells);np.add.at(balance,m['b'][inside],(q*m['wf'])[inside]);balance[m['source']]+=Q
   initial=(np.full(len(cells),depth),q)
   r=b.v.solve(m,[0,86400],[Q,Q],initial,cfl=cfl,max_dt=150)
   row=dict(shared_middle_node=shared,max_length_m=length,cfl=cfl,initial_maximum_continuity_imbalance_cms=float(abs(balance).max()),maximum_final_depth_departure_m=float(abs(r['h']-depth).max()),final_outlet_departure_cms=float(abs(r['wave'][-1,1]-Q)),uniform_equilibrium_preserved=bool(abs(r['h']-depth).max()<1e-9),audit=r['audit'])
   rows.append(row);print(shared,length,row['maximum_final_depth_departure_m'],flush=True)
 b.save(b.OUT/'combined_junction_check.json',dict(description=__doc__,runs=rows,interpretation='The shared-node pairwise min-width interfaces do not preserve this ideal uniform equilibrium at finite resolution. This is inherited from the interface rule, not a mass-conservation failure. Test spatial refinement before attributing real-network differences to it; it is not established as the cause of the observed limiter failures.'))
