#!/usr/bin/env python
"""Repository-local entry point; uses the existing scientific environment."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'synthetic_experiments'/'src', ROOT/'RAPID'/'src'):
    sys.path.insert(0,str(path))
from synthetic_experiments.pipeline import main

if __name__=='__main__':
    main()
