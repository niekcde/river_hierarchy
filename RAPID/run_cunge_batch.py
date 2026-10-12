"""Run network families plus fully collapsed reference sections (Cunge only)."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src')]
from rapid_tools.centerline_batch import cli
if __name__=='__main__':cli()
