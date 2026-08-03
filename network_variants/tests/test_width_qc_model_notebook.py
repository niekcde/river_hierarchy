from __future__ import annotations

from pathlib import Path

import nbformat


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1] / "notebooks" / "width_qc_model_review.ipynb"
)


def test_width_qc_review_notebook_is_valid_and_code_cells_compile() -> None:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nbformat.validate(notebook)

    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"width-qc-notebook-cell-{index}", "exec")


def test_width_qc_review_notebook_covers_required_review_contract() -> None:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    contents = "\n".join(cell.source for cell in notebook.cells)
    required_topics = (
        "Manual-review counts and missingness",
        "Predictor distributions",
        "Pairwise relationships among all classifier and interpretation variables",
        "Correlation and collinearity",
        "Logistic coefficients",
        "Candidate-model comparison",
        "Held-out ROC and precision–recall",
        "Probability calibration",
        "Threshold trade-offs and split-sensitive risk",
        "Partial decision-surface diagnostics",
        "Manual/model disagreements",
        "Disagreement predictor plots",
        "Pairwise disagreement matrices",
        "Local geometry inspection",
        "Provisional recommendation",
    )
    for topic in required_topics:
        assert topic in contents
    assert "width_qc_training_data_v1.csv" in contents
    assert "width_qc_loeo_predictions_v1.csv" in contents
    assert "reviewed_links_snapshot.gpkg" in contents
    assert "production_authorized" in contents
    assert "scikit-learn version mismatch" in contents
    assert "portable_linear_surface" in contents
    assert "joblib.load" not in contents
    assert "disagreement_table_columns" in contents
    assert "classification_columns = list(features)" in contents
    assert "false_correction': '#E66101'" in contents
    assert "missed_correction': '#0571B0'" in contents
    assert "pairwise_review_matrix" in contents
    assert "important_context_variables" in contents
    assert "plotly.graph_objects" in contents
    assert "partial_logistic_boundary" in contents
    assert "Partial logistic boundary" in contents
    assert "network %{customdata[0]}<br>link %{customdata[1]}" in contents
