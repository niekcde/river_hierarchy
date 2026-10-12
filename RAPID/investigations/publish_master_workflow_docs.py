"""Publish commands, inventory counts and software verification alongside the master run."""
from pathlib import Path
import sys,json,shutil,html
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src')]
import pandas as pd
from rapid_tools.run_manifest import digest
STAGE=ROOT/'RAPID/assets/master_workflow_documentation_20261009'
TEST=ROOT/'RAPID/outputs/master_workflow_verification_20261009'

def stage():
 pointer=json.loads((ROOT/'RAPID/config/production_run.json').read_text());manifest=Path(pointer['manifest']);m=json.loads(manifest.read_text());target=manifest.parent
 STAGE.mkdir(parents=True,exist_ok=True)
 f=pd.read_csv(TEST/'inventory_overview.csv',dtype={'example':str});f.to_csv(STAGE/'inventory_overview.csv',index=False)
 summary=dict(study_networks=15,family_entries={'indep':1761,'compound_bubbles':192,'island':1530},total_entries=3483,unique_hydraulic_states=2187,method_jobs=2187,methods=m['methods'],
  base_entries=dict(indep=15,compound_bubbles=15,island=0),excluded_example='11',distinct_state_alias_savings=1296)
 (STAGE/'inventory_overview.json').write_text(json.dumps(summary,indent=2))
 code=[ROOT/'RAPID/src/rapid_tools'/n for n in ['master_workflow.py','run_manifest.py','prep.py','engine.py','cunge.py','common_grid.py']]
 evidence=dict(software_ready_for_master_prep_run=True,full_master_batch_started=False,tests_passed=15,smoke=json.loads((TEST/'verification.json').read_text()),
  resume=json.loads((TEST/'resume_verification.json').read_text()),source_sha256={str(p):digest(p) for p in code},run_manifest_sha256=digest(manifest))
 (STAGE/'software_integration.json').write_text(json.dumps(evidence,indent=2))
 guide='''MASTER RAPID PREP / RUN — 2026-10-09

The production entry points now consume run_manifest.json through the shared loader.
The selected master batch remains configured, not yet prepared or run.

Run these commands from /Users/6256481/Code/river-hierarchy using the project Python:

Inventory:
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python RAPID/investigations/cunge_full_hierarchy_reviewed.py --master-run inventory

Full PREP:
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python RAPID/investigations/cunge_full_hierarchy_reviewed.py --master-run prep

Full RUN (after PREP completes):
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python RAPID/investigations/cunge_full_hierarchy_reviewed.py --master-run run --workers 2

The default manifest is selected by RAPID/config/production_run.json. --manifest
can explicitly select another run. Historical runner modes remain available;
use --master-run for this production batch.

PREP checks the installed geometry release and hydraulic frame checksums, confirms
base hydraulics match calibration, and freezes per-state K/X/grid parameters with
shared native clocks across compared states. Base n/beta are inherited; no children
are fitted. Only calibrated Cunge is selected for production, retaining its selected
subdivisions. Forcing is the reviewed common-grid input.

Compact PREP stores hydraulic parameter tables and one forcing table per example.
RUN uses the independently validated Cunge DAG cascade and audits native discharges.
It retains outlet interval means, not all-reach/time NetCDF arrays. This avoids
large duplicate forcing arrays and transient all-reach storage. Public Python APIs:
rapid_tools.prep.prepare_master_run and rapid_tools.engine.run_master_run.

RUN verifies the PREP receipt and checksums, routes each physical state once per
method, then maps results to independent/compound/island entries. Shared held tails
are extended until every compared result stays within 0.1% of forcing amplitude of
the terminal forcing for at least two trailing days. This is post-event simulation,
not added antecedent observations. Observational feature scoring retains the
original downstream observation extent. Base replay deviation above 0.1% of input
amplitude stops the batch. Outputs retain scientific/source qualifications.

Interrupted RUNs resume completed examples after checksum checks. Incomplete
examples are recomputed; no completed global run is silently overwritten. PREP
completion is required before RUN. A new run manifest/directory is required for
changed inputs or a deliberately new preparation.

Historical two-method integration validation: 15 targeted tests; isolated PREP/RUN on 03 and 46, base plus one child,
8 method jobs. Cunge bases replayed exactly. Maximum single-X base discrepancy was
0.0279 m3/s (0.00171% of input amplitude) on the tightened shared native clock.
Completed-example resume was tested. This is software integration validation,
not proof that every full-batch state will pass its numerical preflight.

Inventory: 15 study networks; 3,483 eligible family entries; 2,187 unique physical
states after cross-family aliases; 2,187 Cunge jobs, including 15 shared base jobs.
Independent and compound counts each include 15 base entries. Example 11 remains
excluded. See reports/inventory_overview.html for counts by study network.
'''
 (STAGE/'README.txt').write_text(guide)
 table=f.rename(columns={'example':'Study network','indep':'Independent entries','compound_bubbles':'Compound entries','island':'Island entries','total_entries':'Total entries','unique_states':'Unique hydraulic states'})
 report='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Master routing inventory</title><style>body{font:16px/1.5 system-ui;max-width:1100px;margin:35px auto;padding:0 20px;color:#20303d}table{border-collapse:collapse;width:100%}td,th{padding:8px 12px;border-bottom:1px solid #ddd;text-align:right}h1{font-size:26px}a{color:#087b8a}</style><h1>Master routing inventory</h1><p><strong>15 study networks · 3,483 family entries · 2,187 unique hydraulic states.</strong></p><p>Each study network contains many independent, compound and island configurations. Entries from different families can refer to the same hydraulic state; those are routed once and reused. Cunge only gives 2,187 routing jobs, including 15 shared base jobs.</p>'''+table.to_html(index=False,border=0)+'''<p>Totals: independent 1,761; compound 192; island 1,530. Independent and compound each include 15 base entries. Unique-state counts remove cross-family aliases; they are not additional study networks.</p><p>Example 11 and the reviewed topology/duplicate exclusions are omitted. Full master preparation and routing have not started.</p><p><a href="15_network_overview.html">Calibrated hydrograph overview</a> · <a href="inventory_overview.csv">Inventory counts CSV</a> · <a href="../README.txt">PREP/RUN commands</a></p></html>'''
 (STAGE/'inventory_overview.html').write_text(report)
 targets={'README.txt':target/'README.txt','software_integration.json':target/'software_integration.json'}
 for n in ['inventory_overview.html','inventory_overview.csv','inventory_overview.json']:targets[n]=target/'reports'/n
 plan={n:dict(target=str(p),expected_previous=digest(p) if p.exists() else None,sha256=digest(STAGE/n)) for n,p in targets.items()}
 (STAGE/'publication_plan.json').write_text(json.dumps(plan,indent=2));print('Documentation staged')

def publish():
 plan=json.loads((STAGE/'publication_plan.json').read_text())
 for name,r in plan.items():
  p=Path(r['target']);assert (digest(p) if p.exists() else None)==r['expected_previous'];assert digest(STAGE/name)==r['sha256']
 for name,r in plan.items():
  p=Path(r['target']);p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(STAGE/name,p);assert digest(p)==r['sha256']
 print('Master workflow documentation and inventory report published.')

if __name__=='__main__':stage() if sys.argv[1]=='stage' else publish()
