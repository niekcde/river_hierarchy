"""Re-run five selected collapses with the current MC bed profiles, frozen local n."""
from pathlib import Path
import broad_study as b
from investigate_junctions37 import variant,OUT
MC=Path('/Volumes/PhD/river_hierarchy/three_run_comparison_20261028/03_manning_midpoint/states')
if __name__=='__main__':
    b.v.OUT=OUT
    records=[]
    for ex,state in [('37','S256_unit_229'),('37','S274_unit_280'),('15','S171_unit_163'),('54','S055_unit_44'),('56','S015_unit_4')]:
        records.append(variant(state,ex,MC/ex/state/'parameters.parquet',False))
        b.save(OUT/'current_bed_controls.json',dict(runs=records,scope='Only bed profiles updated to the existing current MC source. Keep the original-network fitted local n, solver, widths, lengths, forcing and 100 m maximum. No junction partition change.'))
    print('CURRENT BED CONTROLS COMPLETE',flush=True)
