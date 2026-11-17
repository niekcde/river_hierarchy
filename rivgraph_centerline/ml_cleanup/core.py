"""Native-resolution paired-mask preparation and leakage-free image features."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

SEED = 20261106
CORE = 192
HALO = 32
SIDE = CORE + 2 * HALO
CROP = np.s_[HALO:-HALO, HALO:-HALO]
ROOT = Path(__file__).resolve().parents[1] / 'outputs' / 'runs'
OUT = Path(__file__).resolve().parent / 'results'


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def aligned_offset(base, clean):
    if base.crs != clean.crs or not np.allclose(tuple(base.transform)[:2] + tuple(base.transform)[3:5], tuple(clean.transform)[:2] + tuple(clean.transform)[3:5], atol=1e-8, rtol=0):
        raise ValueError('CRS or pixel grid differs')
    col, row = ~base.transform * (clean.transform.c, clean.transform.f)
    if not np.allclose([col, row], np.round([col, row]), atol=1e-5, rtol=0):
        raise ValueError('Non-integer pixel offset')
    col, row = int(round(col)), int(round(row))
    if row < 0 or col < 0 or row + clean.height > base.height or col + clean.width > base.width:
        raise ValueError('Cleaned extent outside prepared extent')
    return col, row


def features(base, raw, pixel_size, include_raw=True):
    """Input-only spatial features; no target, coordinates, or river identity."""
    b = base.astype(np.float32)
    fs = [b]
    # Physical distance scales reduce sensitivity to different pixel resolutions.
    for metres in (15, 40, 100, 200):
        fs.append(ndi.gaussian_filter(b, max(0.6, metres / pixel_size)))
    fs += [np.minimum(ndi.distance_transform_edt(base) * pixel_size, 250) / 250,
           np.minimum(ndi.distance_transform_edt(~base) * pixel_size, 250) / 250]
    for val in (base, ~base):
        lab, _ = ndi.label(val, np.ones((3, 3)))
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        fs.append(np.log1p(sizes[lab]).astype(np.float32) / 12)
    if include_raw:
        fs += [raw.astype(np.float32)]
        for metres in (15, 50, 150):
            fs.append(ndi.gaussian_filter(raw.astype(np.float32), max(0.6, metres / pixel_size)))
    return np.stack(fs, axis=-1).astype(np.float32)


def cleanup(mask, min_water=16, max_hole=3):
    m = mask.copy().astype(bool)
    lab, _ = ndi.label(m, np.ones((3, 3)))
    sz = np.bincount(lab.ravel()); sz[0] = 0
    m &= sz[lab] >= min_water
    lab, _ = ndi.label(~m)  # dual connectivity: water 8, land 4
    sz = np.bincount(lab.ravel())
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    fill = sz <= max_hole; fill[0] = False; fill[border] = False
    m |= fill[lab]
    return m


def holes(m):
    lab, _ = ndi.label(~m)
    sz = np.bincount(lab.ravel())
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    keep = sz >= 4; keep[0] = False; keep[border] = False
    return keep[lab], int(keep.sum())


def counts(pred, target, base, pixel_size):
    p, y, b = pred.astype(bool), target.astype(bool), base.astype(bool)
    tp = int((p & y).sum()); fp = int((p & ~y).sum()); fn = int((~p & y).sum())
    pb = p ^ ndi.binary_erosion(p); yb = y ^ ndi.binary_erosion(y)
    # Tile edges are not physical river banks.
    for a in (pb, yb): a[:3] = False; a[-3:] = False; a[:, :3] = False; a[:, -3:] = False
    dy = ndi.distance_transform_edt(~yb) if yb.any() else np.full(y.shape, np.inf)
    dp = ndi.distance_transform_edt(~pb) if pb.any() else np.full(y.shape, np.inf)
    sp, sy = skeletonize(p), skeletonize(y)
    hp, np_h = holes(p); hy, ny_h = holes(y)
    remove = b & ~y; add = ~b & y; changed = p != b; needed = b != y
    return dict(tp=tp, fp=fp, fn=fn, pixels=int(y.size), errors=fp+fn,
        boundary_pred=int(pb.sum()), boundary_true=int(yb.sum()),
        boundary_pred_hit=int((pb & (dy <= 2)).sum()), boundary_true_hit=int((yb & (dp <= 2)).sum()),
        boundary_distance_sum_m=float((np.minimum(dy[pb], 50).sum()+np.minimum(dp[yb], 50).sum())*pixel_size),
        skel_pred=int(sp.sum()), skel_true=int(sy.sum()), skel_pred_hit=int((sp & y).sum()), skel_true_hit=int((sy & p).sum()),
        removed_needed=int(remove.sum()), removed_correct=int((remove & ~p).sum()),
        added_needed=int(add.sum()), added_correct=int((add & p).sum()),
        changes=int(changed.sum()), correct_changes=int((changed & needed).sum()),
        damage=int((changed & ~needed).sum()), unchanged=int((~needed).sum()),
        islands_pred=np_h, islands_true=ny_h, island_count_abs_error=abs(np_h-ny_h),
        island_tp=int((hp & hy).sum()), island_fp=int((hp & ~hy).sum()), island_fn=int((~hp & hy).sum()),
        components_pred=int(ndi.label(p,np.ones((3,3)))[1]), components_true=int(ndi.label(y,np.ones((3,3)))[1]))


def scores(c):
    def div(a,b): return a/b if b else None
    def f1(a,b):
        if a is None and b is None:return None
        if a is None or b is None or a+b==0:return 0.0
        return 2*a*b/(a+b)
    return dict(iou=div(c['tp'],c['tp']+c['fp']+c['fn']),
        dice=div(2*c['tp'],2*c['tp']+c['fp']+c['fn']),
        boundary_f1=f1(div(c['boundary_pred_hit'],c['boundary_pred']),div(c['boundary_true_hit'],c['boundary_true'])),
        capped_mean_boundary_distance_m=div(c['boundary_distance_sum_m'],c['boundary_pred']+c['boundary_true']),
        cldice=f1(div(c['skel_pred_hit'],c['skel_pred']),div(c['skel_true_hit'],c['skel_true'])),
        removal_recall=div(c['removed_correct'],c['removed_needed']), addition_recall=div(c['added_correct'],c['added_needed']),
        edit_precision=div(c['correct_changes'],c['changes']), damage_rate=div(c['damage'],c['unchanged']),
        island_iou=div(c['island_tp'],c['island_tp']+c['island_fp']+c['island_fn']),
        island_count_abs_error=c['island_count_abs_error'], errors=c['errors'])


def load_tile(rec):
    with np.load(OUT / rec['file']) as z: return {k:z[k] for k in z.files}
