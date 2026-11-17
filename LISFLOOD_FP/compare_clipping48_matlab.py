"""Compare actual MATLAB replay with the instrumented production kernel."""
from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat
root=Path(__file__).resolve().parent/'results/broad_15_100m/investigation/first_clipping48'
p=dict(np.load(root/'first_failure.npz'));result={}
for label,folder in [('one_step',root/'one_step'),('full_prefix',root)]:
    m=loadmat(folder/'matlab_first_failure.mat',squeeze_me=True)
    result[label]=dict(time_s=float(m['t']),dt_s=float(m['dt']),steps=int(m['steps']),first_cell_zero_based=int(np.argmin(m['trial'])),minimum_trial_depth_m=float(np.min(m['trial'])),max_pre_depth_difference_m=float(np.max(abs(m['h']-p['pre_h']))),max_unit_discharge_difference=float(np.max(abs(m['q']-p['new_q']))),max_trial_depth_difference_m=float(np.max(abs(m['trial']-p['trial']))))
result['python']=dict(time_s=float(p['step'][0]),dt_s=float(p['step'][1]),steps=int(p['step'][3]),first_cell_zero_based=int(np.argmin(p['trial'])),minimum_trial_depth_m=float(np.min(p['trial'])))
result['scope']='Actual MATLAB CPU adaptation: shared geometry, initialization, normal-flow outlet and step accounting; not the unchanged supplied GPU program.'
(root/'matlab_comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
d=root/'diagnosis.json';j=json.loads(d.read_text());j['matlab_status']='Completed; see matlab_comparison.json';j['matlab_comparison']=result;d.write_text(json.dumps(j,indent=2))
