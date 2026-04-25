"""Compatibility wrapper for geometry-first synthetic enumeration."""

from pathlib import Path
import sys

_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from synthetic_runs.enumerate.geometry import *  # noqa: F401,F403
from synthetic_runs.enumerate.geometry import main as _phase7_main


if __name__ == "__main__":
    _phase7_main()
