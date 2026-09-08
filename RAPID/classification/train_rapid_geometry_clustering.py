#!/usr/bin/env python3
"""Explore unsupervised geometry/resistance clusters for RAPID collapse states.

RAPID timing response is excluded from cluster fitting. It is joined only after
clustering to test whether geometry-derived groups have distinct responses.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, OPTICS, SpectralClustering
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_rapid_impact_classifier import Config, PRIMARY_Y_METRIC, load_and_filter


MODEL_VERSION = "rapid-geometry-clustering-v1"
RANDOM_STATE = 20260824
# Kept deliberately small and physically interpretable. Correlated candidates
# are pruned automatically before clustering to avoid double-weighting a metric.
CANDIDATE_FEATURES = (
    "len_fraction_width_weighted_log_lambda",
    "along_network_frac",
    "length_fraction",
    "width_fraction",
    "k_ratio_to_matched_base",
)
CORRELATION_PRUNE_THRESHOLD = 0.85
MIN_CLUSTER_SIZE = 5
MIN_CLUSTER_EXAMPLES = 2
MAX_NOISE_FRACTION_FOR_REVIEW = 0.30


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--example-ids", nargs="+", default=["02", "03", "05", "11", "21"], metavar="ID")
    parser.add_argument("--results-dir", type=Path, default=root / "RAPID" / "classification" / "results" / f"{MODEL_VERSION}-{stamp}")
    return parser.parse_args()


def select_features(frame: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    available = [column for column in CANDIDATE_FEATURES if column in frame and frame[column].notna().any()]
    numeric = frame.loc[:, available].apply(pd.to_numeric, errors="coerce")
    correlation = numeric.corr()
    retained: list[str] = []
    for feature in available:
        if not retained or correlation.loc[feature, retained].abs().max() < CORRELATION_PRUNE_THRESHOLD:
            retained.append(feature)
    if len(retained) < 2:
        raise ValueError(f"Too few non-collinear geometry features for clustering: {retained}")
    return retained, correlation


def score(labels: np.ndarray, values: np.ndarray, example_ids: pd.Series) -> dict[str, float | int | bool]:
    labels = np.asarray(labels)
    non_noise = labels != -1
    used_labels = labels[non_noise]
    clusters = np.unique(used_labels)
    sizes = pd.Series(used_labels).value_counts()
    support = pd.DataFrame({"cluster": labels[non_noise], "example_id": example_ids.to_numpy()[non_noise]}).groupby("cluster")["example_id"].nunique()
    result: dict[str, float | int | bool] = {
        "n_clusters": int(len(clusters)),
        "noise_fraction": float(1 - non_noise.mean()),
        "smallest_cluster_size": int(sizes.min()) if len(sizes) else 0,
        "min_cluster_example_support": int(support.min()) if len(support) else 0,
    }
    result["valid_exploratory_solution"] = bool(
        len(clusters) >= 2
        and result["smallest_cluster_size"] >= MIN_CLUSTER_SIZE
        and result["min_cluster_example_support"] >= MIN_CLUSTER_EXAMPLES
    )
    # A sparse component can be scientifically meaningful as an outlier, but
    # it is not evidence for a reproducible multi-example *regular* class.
    result["outlier_candidate"] = bool(
        len(clusters) >= 3
        and result["smallest_cluster_size"] < MIN_CLUSTER_SIZE
        and result["noise_fraction"] <= MAX_NOISE_FRACTION_FOR_REVIEW
    )
    if len(clusters) < 2 or non_noise.sum() <= len(clusters):
        return {**result, "silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}
    subset = values[non_noise]
    result["silhouette"] = float(silhouette_score(subset, used_labels))
    result["calinski_harabasz"] = float(calinski_harabasz_score(subset, used_labels))
    result["davies_bouldin"] = float(davies_bouldin_score(subset, used_labels))
    return result


def algorithms(values: np.ndarray) -> dict[str, np.ndarray]:
    found: dict[str, np.ndarray] = {}
    n_samples = len(values)
    for n_clusters in range(2, min(6, n_samples - 1) + 1):
        found[f"kmeans_k{n_clusters}"] = KMeans(n_clusters=n_clusters, n_init=30, random_state=RANDOM_STATE).fit_predict(values)
        found[f"gmm_full_k{n_clusters}"] = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=RANDOM_STATE).fit_predict(values)
        found[f"agglomerative_ward_k{n_clusters}"] = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(values)
        found[f"agglomerative_average_k{n_clusters}"] = AgglomerativeClustering(n_clusters=n_clusters, linkage="average").fit_predict(values)
        neighbors = min(10, n_samples - 1)
        found[f"spectral_k{n_clusters}"] = SpectralClustering(
            n_clusters=n_clusters, affinity="nearest_neighbors", n_neighbors=neighbors, assign_labels="kmeans", random_state=RANDOM_STATE
        ).fit_predict(values)
    # Density methods receive a range of data-derived neighbourhood scales.
    distances = NearestNeighbors(n_neighbors=min(6, n_samples)).fit(values).kneighbors(values)[0][:, -1]
    for quantile in (0.50, 0.70, 0.85):
        eps = float(np.quantile(distances, quantile))
        found[f"dbscan_q{int(quantile * 100)}"] = DBSCAN(eps=eps, min_samples=5).fit_predict(values)
    found["optics"] = OPTICS(min_samples=5, xi=0.05).fit_predict(values)
    return found


def main() -> None:
    args = parse_args()
    config = Config(example_ids=tuple(str(value).zfill(2) for value in args.example_ids))
    results_dir = args.results_dir.resolve()
    if results_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing result directory: {results_dir}")
    results_dir.mkdir(parents=True)
    frame, audit = load_and_filter(config, args.output_root.resolve())
    features, correlations = select_features(frame)
    transform = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    values = transform.fit_transform(frame.loc[:, features])
    summary_rows: list[dict[str, object]] = []
    assignments: list[pd.DataFrame] = []
    for name, labels in algorithms(values).items():
        metrics = score(labels, values, frame["example_id"])
        # This post-hoc summary intentionally occurs after clustering.
        response = frame.assign(cluster=labels).groupby("cluster", dropna=False)[PRIMARY_Y_METRIC].agg(["size", "mean", "median", "std"])
        response_range = float(response["mean"].max() - response["mean"].min()) if len(response) else np.nan
        summary_rows.append({"algorithm": name, **metrics, "posthoc_response_mean_range_pct_points": response_range})
        assignments.append(
            frame.loc[:, ["example_id", "state_id", *features, PRIMARY_Y_METRIC]].assign(algorithm=name, cluster=labels)
        )
    summary = pd.DataFrame(summary_rows)
    summary["review_status"] = np.select(
        [
            summary["valid_exploratory_solution"],
            summary["outlier_candidate"],
            summary["noise_fraction"] > MAX_NOISE_FRACTION_FOR_REVIEW,
        ],
        [
            "regular-structure candidate — inspect stability and interpretation",
            "outlier candidate — validate its geometry and RAPID response separately",
            "unsupported density partition — excessive noise fraction",
        ],
        default="unsupported partition — insufficient cluster support",
    )
    # This is a review queue, not a claim that silhouette identifies a winning
    # algorithm. Supported candidates appear first for human comparison.
    summary = summary.sort_values(["valid_exploratory_solution", "algorithm"], ascending=[False, True])
    pd.concat(assignments, ignore_index=True).to_csv(results_dir / "cluster_assignments.csv", index=False)
    summary.to_csv(results_dir / "clustering_quality_summary.csv", index=False)
    correlations.to_csv(results_dir / "candidate_feature_correlations.csv")
    (results_dir / "selected_features.json").write_text(json.dumps({"features": features, "correlation_prune_threshold": CORRELATION_PRUNE_THRESHOLD}, indent=2) + "\n")
    report = f"""# Geometry-only clustering exploration ({MODEL_VERSION})

Clusters were fitted from these features only: `{', '.join(features)}`. The RAPID response `{PRIMARY_Y_METRIC}` was excluded from fitting and summarized only afterward.

The run used {audit['eligible_rows']} eligible states from {audit['eligible_examples']} examples. A regular-structure candidate has at least {MIN_CLUSTER_SIZE} states and representation in at least {MIN_CLUSTER_EXAMPLES} examples per non-noise cluster. A sparse component is retained as an **outlier candidate** when there are at least three clusters and noise is at most {MAX_NOISE_FRACTION_FOR_REVIEW:.0%}; validate it separately rather than treating it as a reproducible class. **No algorithm is declared best automatically.** Silhouette (higher), Davies-Bouldin (lower), Calinski-Harabasz (higher), cluster count, noise fraction, resampling stability, and physical interpretation answer different questions and are not universally comparable across clustering families. A post-hoc response-mean range is descriptive only; it is not a clustering-quality criterion.

{summary.to_csv(index=False)}
"""
    (results_dir / "REPORT.md").write_text(report)
    print(f"Wrote {results_dir}")
    print("Wrote a clustering review table; inspect supported candidates rather than treating any single metric as a winner.")


if __name__ == "__main__":
    main()
