MASTER RAPID PREP / RUN — 2026-10-09

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
