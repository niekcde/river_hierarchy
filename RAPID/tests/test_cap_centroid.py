"""Check threshold crossings and all-crest integration in the exploratory metric."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('cap_check',ROOT/'investigations/check_cap_centroids_20261019.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def test_asymmetric_triangle_crossings():
    r=mod.cap_centroid([0,2,6],[0,5,0],.8)
    assert r['cap_centroid_seconds']==pytest.approx((1.6+2+2.8)/3)
    assert r['cap_volume_m3']==pytest.approx(.6)


def test_translation_and_amplitude_scaling():
    r=mod.cap_centroid(np.array([0,2,6])+120,np.array([0,5,0])*3,.8)
    assert r['cap_centroid_seconds']==pytest.approx(120+(1.6+2+2.8)/3)
    assert r['cap_volume_m3']==pytest.approx(1.8)


def test_disconnected_crests_both_contribute():
    r=mod.cap_centroid([0,2,6,10,12,16],[0,5,0,0,5,0],.8)
    assert r['cap_centroid_seconds']==pytest.approx(5+(1.6+2+2.8)/3)
    assert r['cap_volume_m3']==pytest.approx(1.2)
