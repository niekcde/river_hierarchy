"""Attach the completed 46 initialization qualification to both review figures."""
from pathlib import Path
import json,hashlib,re
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'RAPID/outputs/bounded_beta_calibration_20261009'
NOTE=('Example 46 initialization check: the displayed equilibrium-start fit is +0.5 minute from the target. '
      'Using 7/30 days of actual antecedent forcing instead gives +15.5/+14.5 minutes, with peak height '
      'less than 0.5% lower. This sensitivity remains a qualification; antecedent forcing has not been substituted in these panels.')

def main():
 changes={}
 for mode in ['full','crest']:
  path=OUT/f'15_panel_calibrated_{mode}.html';text=path.read_text();before=re.search(r'<script id="payload" type="application/octet-stream">(.*?)</script>',text).group(1)
  if 'id="initialization46"' not in text:
   text=text.replace('</header>',f'<p id="initialization46" style="background:#fff3d6;padding:12px">{NOTE}</p></header>',1)
   path.write_text(text)
  after=re.search(r'<script id="payload" type="application/octet-stream">(.*?)</script>',path.read_text()).group(1);assert before==after
  changes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
 (OUT/'publication_notes.json').write_text(json.dumps(dict(note=NOTE,numeric_payload_unchanged=True,html_sha256=changes,annotation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
 (OUT/'source_code'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 report=OUT/'REPORT.txt';text=report.read_text()
 if NOTE not in text:report.write_text(text+'\n'+NOTE+'\n')
if __name__=='__main__':main()
