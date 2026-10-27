"""Plan or sequentially run the reviewed 15 input events in isolated output."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'RAPID/src')]
from rapid_tools.event_window_batch import cli
if __name__ == '__main__':
    cli()
