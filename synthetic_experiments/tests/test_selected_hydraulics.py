"""Scientific invariants for the versioned selected hydraulic adapter."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'synthetic_experiments/investigations'))
from width_depth_scaling import adjusted,predictor,make_states,canonical_network,routing
from segmentation_sensitivity import run_frame
from selected_hydraulic_workflow import clock_for

PARAMS=dict(total_width_m=500,reference_slope=1e-4,outer_a_fraction=.8,inner_fraction=.5,length_multiplier=1.25,total_length_m=20000,internal_location='central')


def test_reference_anchor_and_allocation_preserved():
    states,_=make_states(PARAMS);net=canonical_network(states['nested/original'])
    s=routing.Settings(kb=144.2974472721433,n_manning=.035)
    frame=routing.prepare(net,s);new=adjusted(frame,s,500,.65)
    assert np.array_equal(frame.width,new.rapid_branch_width_m)
    reference=new.width==500
    assert np.allclose(new.loc[reference,'rapid_k'],frame.loc[reference,'rapid_k'])
    # Independent local depth formula; narrowing no longer implies linear depth reduction.
    expected=(500/s.kb)*(new.width/500)**.65
    assert np.allclose(new.rapid_kb_value,new.width/expected)
    assert (expected[new.width<500]>(new.width/s.kb)[new.width<500]).all()


def test_alpha_one_reproduces_routing():
    states,_=make_states(PARAMS);net=canonical_network(states['simple/original']);s=routing.Settings(n_manning=.035,kb=144.2974472721433)
    clock=routing.schedule([net],s)
    h0,_,frame=routing.route(net,s,clock)
    h1,_,_=run_frame(net,adjusted(frame,s,500,1.),s,clock)
    assert np.allclose(h0.outlet_cms,h1.outlet_cms,rtol=1e-10,atol=1e-8)


def test_simple_resistance_matches_parallel_power_law():
    graphs,pairs=make_states(PARAMS);pair=next(p for p in pairs if p['child_state']=='simple/full')
    s=routing.Settings(kb=144.2974472721433,n_manning=.035)
    result=predictor(graphs[pair['parent_state']],graphs[pair['child_state']],pair,s,500,.65)
    # K_i = common scale * L_i / W_i^(2+3alpha); parallel Q=sum sqrt(dH/K_i).
    power=2+3*.65
    parent=graphs[pair['parent_state']];child=graphs[pair['child_state']]
    a,b=pair['selection_start_m'],pair['selection_end_m']
    def edges(g):return [d for u,v,k,d in g.edges(keys=True,data=True) if g.nodes[u]['x']>=a and g.nodes[v]['x']<=b]
    kp=[d['length_m']/d['w']**power for d in edges(parent)]
    kc=sum(d['length_m']/d['w']**power for d in edges(child))
    equivalent=1/sum(1/np.sqrt(k) for k in kp)**2
    assert len(kp)==2
    assert np.isclose(result['Lambda_matched'],kc/equivalent,rtol=1e-7)


def test_selected_long_narrow_retains_200m_integer_clock():
    params=dict(PARAMS,outer_a_fraction=.05,length_multiplier=3.)
    states,_=make_states(params);s=routing.Settings(kb=144.2974472721433,n_manning=.035)
    clock=clock_for([canonical_network(g) for g in states.values()],s,500)
    assert clock['dt_seconds']==315
    assert clock['dt_min_seconds']<315<330<clock['dt_max_seconds']
