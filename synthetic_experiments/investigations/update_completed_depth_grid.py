"""Build current full-grid analysis without rerouting."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'RAPID/src',ROOT/'synthetic_experiments/src'):sys.path.insert(0,str(p))
from synthetic_experiments.full_grid_review import build
if __name__=='__main__':
    tables=build()
    for name,frame in tables.items():print(name,len(frame),flush=True)
