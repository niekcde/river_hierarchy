"""Resolve the explicitly selected synthetic analysis version, never silently mix runs."""
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / 'synthetic_experiments/configs/active_analysis.json'


def results_root():
    override = os.environ.get('SYNTHETIC_RESULTS_ROOT')
    if override:
        return Path(override).resolve()
    if REGISTRY.exists():
        record = json.loads(REGISTRY.read_text())
        return ROOT / record['results_root']
    return ROOT / 'synthetic_experiments/outputs'


RUN = results_root() / 'full_grid_island'
EXTENSION = results_root() / 'embedded_width_length_extension'
