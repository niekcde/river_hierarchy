"""Add and execute only the new outlier-review cells, preserving existing cells."""
from pathlib import Path
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT=Path(__file__).resolve().parents[3]
TAG='delta-j-outlier-groups-investigation'

def main():
    path=ROOT/'RAPID/notebooks/rapid_hydrograph_review.ipynb'
    nb=nbformat.read(path,as_version=4)
    preserved=[c for c in nb.cells if TAG not in c.metadata.get('tags',[])]
    before=[nbformat.writes(nbformat.v4.new_notebook(cells=[c])) for c in preserved]
    markdown='''## Manning ΔJ outliers — three-group hydraulic investigation

The original **25 flags and pooled regression are unchanged**. This section uses completed fixed-Manning runs and analytic transfer moments; it launches no routing.

- **G1 (11):** nine all-closed-unit collapses plus two full compounds. All advance and reduce transfer spread; their proportional response differs from the pooled line.
- **G2 (4):** three delayed members increase dispersion; 46/S178_unit_44 instead resembles the aggregate advances.
- **G3 (10):** eight members and two islands. All increase transfer spread (1.24–5.47×); within-reach dispersion explains at least 99.88% of the added variance. Seven of the eight members have active slope-floor reaches. This is a model mechanism and an input-representation concern, not yet proof of a real-river effect.

**Transfer spread is not observed event width.** It includes potentially small, long-lived model tails. Matched non-outliers sometimes also increase spread. No opposite-sign case has direct saved time/grid refinement coverage.

The first plot places the existing residual z-scores against the change in analytic transfer spread. The dropdown below shows **actual saved parent and child curves**, followed by their difference. Grey marks the original response window; dotted lines mark the two window centroids. Double-click to view the full saved duration. For compact display, each ten-minute bin retains its first sample and both extrema; metrics use the original full-resolution records. Nothing is artificially shifted.

Full derivation, all 25 case notes, controls, source provenance and limitations: `RAPID/outputs/uncapped_event_window_20261020/three_run_review_20261029/incremental_storage/group_investigation/REPORT.md`.
'''
    code='''from pathlib import Path
import sys, importlib
import pandas as pd
from IPython.display import HTML, display
J_GROUP_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "RAPID/investigations/three_run_review_20261029").is_dir())
sys.path.insert(0, str(J_GROUP_ROOT / "RAPID/investigations/three_run_review_20261029"))
import plot_j_outlier_groups as j_group_plots
importlib.reload(j_group_plots)
J_GROUP_OUTPUT = j_group_plots.DEST
j_group_outliers = pd.read_csv(J_GROUP_OUTPUT / "outlier_diagnostics.csv", dtype={"example": str})
display(HTML(j_group_plots.overview().to_html(full_html=False, include_plotlyjs=True)))
display(j_group_outliers[["group", "example", "state", "delta_J_minutes", "centroid_shift_minutes", "residual_z", "sd_ratio", "checks"]].round(3))
'''
    inspector='''# Dropdown contains all 25 cases. Curves come directly from completed outlet.parquet records.
display(HTML(j_group_plots.wave_inspector().to_html(full_html=False, include_plotlyjs=True)))
'''
    cells=[nbformat.v4.new_markdown_cell(markdown),nbformat.v4.new_code_cell(code),nbformat.v4.new_code_cell(inspector)]
    shell=InteractiveShell.instance()
    for c in cells:
        c.metadata['tags']=[TAG]
        if c.cell_type!='code':continue
        with capture_output(stdout=True,stderr=True,display=True) as cap:
            result=shell.run_cell(c.source,store_history=True)
        if result.error_before_exec:raise result.error_before_exec
        if result.error_in_exec:raise result.error_in_exec
        c.execution_count=result.execution_count
        c.outputs=[nbformat.v4.new_output('display_data',data=o.data,metadata=o.metadata) for o in cap.outputs]
        if cap.stdout:c.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=cap.stdout))
        if cap.stderr:c.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=cap.stderr))
    nb.cells=preserved+cells
    nbformat.validate(nb)
    assert before==[nbformat.writes(nbformat.v4.new_notebook(cells=[c])) for c in nb.cells[:-len(cells)]]
    nbformat.write(nb,path)
    print('Preserved',len(preserved),'existing cells; added and executed',len(cells),'review cells.')

if __name__=='__main__':main()
