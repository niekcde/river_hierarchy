"""Refit only corrected target 45; preserve the completed first review snapshot."""
from pathlib import Path
import sys,json,shutil
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import cunge_calibration_reviewed as cal
import cunge_reviewed_verification as verification

def main():
 old=pd.read_csv(cal.OUT/'comparison.csv',dtype={'example':str});assert len(old)==32
 archive=cal.OUT/'before_case45_correction';archive.mkdir(exist_ok=True)
 for name in ['comparison.csv','targets_snapshot.json','prepared_verification.csv','old_new_calibration.csv','hierarchy_rerun_plan.csv','completion_checks.json','REPORT.txt']:
  if (cal.OUT/name).exists() and not (archive/name).exists():shutil.copy2(cal.OUT/name,archive/name)
 results=cal.run('45')
 new=pd.concat([old[old.example!='45'],pd.DataFrame(results)],ignore_index=True).sort_values(['example','method'])
 assert len(new)==32;new.to_csv(cal.OUT/'comparison.csv',index=False)
 result=verification.verify('45');v=pd.read_csv(cal.OUT/'prepared_verification.csv',dtype={'example':str});pd.concat([v[v.example!='45'],pd.DataFrame([result])],ignore_index=True).sort_values('example').to_csv(cal.OUT/'prepared_verification.csv',index=False)
 (cal.OUT/'targets_snapshot.json').write_bytes(cal.TARGETS_PATH.read_bytes())
 (cal.OUT/'target_revision_status.json').write_text(json.dumps(dict(changed_targets=['45'],requires_refit=[],case45_refit_complete=True,previous_snapshot=str(archive),other_targets_user_accepted=True),indent=2))
 print(new[new.example=='45'][['example','method','n_manning','error_h','within_sampling']].to_string(index=False))
if __name__=='__main__':main()
