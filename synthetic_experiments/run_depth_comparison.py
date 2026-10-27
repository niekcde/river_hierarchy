"""Current hydraulic comparison entry point. Legacy run_experiment.py is unchanged."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
for path in (root,root/'RAPID/src',root/'synthetic_experiments/src'):
    sys.path.insert(0,str(path))
from synthetic_experiments.depth_pipeline import main
if __name__=='__main__':main()
