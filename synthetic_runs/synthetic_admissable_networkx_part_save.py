"""Compatibility wrapper for the preserved legacy reference implementation."""

from pathlib import Path
import sys

_ROOT_DIR = Path(__file__).resolve().parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from legacy.reference.synthetic_admissable_networkx_part_save import *  # noqa: F401,F403
