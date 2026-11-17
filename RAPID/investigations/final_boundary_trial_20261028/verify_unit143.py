from pathlib import Path
import importlib.util
ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('final_verification',ROOT/'RAPID/investigations/final_boundary_trial_20261028/verify.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OUT=ROOT/'RAPID/outputs/unit143_completed_trial_20261028';m.main()
