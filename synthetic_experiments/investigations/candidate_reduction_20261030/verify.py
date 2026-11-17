"""Integrity checks for the completed descriptive outputs (no statistical fitting)."""
import json,itertools,platform
from pathlib import Path
import numpy as np
import pandas as pd
import scipy,matplotlib
import audit
O=audit.OUT
for name in ['provenance.json','shortlist_provenance.json']:
 d=json.loads((O/name).read_text())
 for p,h in d['sources'].items():assert audit.sha(p)==h,p
inv=pd.read_csv(O/'inventory.csv');ledger=pd.read_csv(O/'candidate_decisions.csv');short=pd.read_csv(O/'shortlist.csv');cov=pd.read_csv(O/'coverage_by_type.csv');u=pd.read_csv(O/'undefined_reasons.csv')
assert len(inv)==205 and len(ledger)==205 and inv.variable.nunique()==ledger.variable.nunique()==205
assert set(inv.variable)==set(ledger.variable)
assert len(short)==19 and short.variable.nunique()==19 and short.minimum_type_coverage.eq(100).all()
assert short.tier.value_counts().to_dict()=={'primary':14,'alternative':3,'context reserve':2}
assert not u.reason.str.contains('unresolved').any()
assert int((cov.n-cov.finite).sum())==int(u.n.sum())
quality=pd.read_csv(O/'finite_but_undefined_audit.csv');assert not set(quality.variable)&set(short.variable)
pairs=pd.read_csv(O/'predictor_pairs.csv');assert len(pairs)==161*160//2
lookup={tuple(sorted([r.a,r.b])):r.conservative_similarity for r in pairs.itertuples()}
clusters=pd.read_csv(O/'clusters.csv')
for (threshold,k),g in clusters.groupby(['threshold','cluster']):
 for a,b in itertools.combinations(g.variable,2):assert lookup[tuple(sorted([a,b]))]>=threshold-1e-10
sensitivity=pd.read_csv(O/'predictor_tolerance_sensitivity.csv').pivot(index=['variable','child_state'],columns='tolerance_factor',values='varying');assert sensitivity.nunique(axis=1).max()==1
sources={str(Path(__file__)):audit.sha(__file__)}
outputs={str(p):audit.sha(p) for p in O.iterdir() if p.is_file() and p.name!='validation.json'}
receipt=dict(complete=True,checks=['source hashes','205 unique dispositions','19 complete shortlisted predictors','all missing entries explained','no undefined derived values in shortlist','all cluster members meet thresholds','constant flags invariant across three tolerances'],sources=sources,output_sha256=outputs,versions={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__})
(O/'validation.json').write_text(json.dumps(receipt,indent=2));print('PASS:',len(outputs),'outputs; 205 dispositions; 19 complete candidates; all integrity checks')
