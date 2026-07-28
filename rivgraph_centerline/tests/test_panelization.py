from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "rivgraph_centerline" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rivgraph_centerline.panelization import create_panel_breaks, plan_panels, prepare_panel_masks


def test_panel_breaks_and_plan_create_four_irregular_core_outputs(tmp_path: Path) -> None:
    mask = tmp_path / "mask.tif"
    with rasterio.open(mask, "w", driver="GTiff", height=100, width=100, count=1, dtype="uint8", crs="EPSG:32611", transform=from_origin(0, 100, 1, 1), nodata=0) as dst:
        dst.write(np.ones((100, 100), dtype="uint8"), 1)
    breaks = tmp_path / "breaks.gpkg"
    plan = tmp_path / "panel_plan.gpkg"
    assert create_panel_breaks(mask_path=mask, output_path=breaks, rows=2, columns=2)["break_count"] == 2
    summary = plan_panels(mask_path=mask, breaks_path=breaks, output_path=plan, overlap_m=5)
    assert summary["panel_count"] == 4
    assert summary["over_budget_panel_ids"] == []
    assert plan.exists()
    assert plan.with_suffix(".json").exists()
    prepared = prepare_panel_masks(mask_path=mask, plan_path=plan, output_dir=tmp_path / "panels", panel_ids={"P001"})
    assert prepared["prepared_panel_ids"] == ["P001"]
    assert (tmp_path / "panels" / "P001" / "mask.tif").exists()
