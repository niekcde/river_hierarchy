"""MAE/R2 comparison and a response-selected, not improvement-selected unit example."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'RAPID/outputs/uncapped_event_window_20261020'
SOURCE = BASE/'combination_grouped_validation_20261026'
EXPANSION = BASE/'combination_expansion_20261026'
SCOPES = ROOT/'RAPID/outputs/hydrograph_predictor_scopes_20261014'
DEST = SOURCE/'metric_and_unit_examples'
ORDER = ['baseline', 'plus80', 'plus240']
MODELS = ['geometry_9', 'depth_informed_10']
LABELS = ['Geometry (9 predictors)', 'Depth informed (10 predictors)']
COLORS = ['#207b9c', '#bb5832']
SOURCES = {}


def read(path):
    SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path, dtype={'example': str})


def main():
    DEST.mkdir(exist_ok=True)
    summary = read(SOURCE/'summary.csv')
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
    for j, (cohort, title) in enumerate([('original', 'Regular units / compounds'), ('segmented', 'Fully segmented cases'), ('combined240', 'All 240 combinations')]):
        for i, (metric, ylabel) in enumerate([('mae_minutes', 'MAE (minutes; lower is better)'), ('r2', 'R² (higher is better)')]):
            ax = axes[i, j]
            for model, label, color in zip(MODELS, LABELS, COLORS):
                z = summary[(summary.model == model) & (summary.cohort == cohort)].set_index('variant').loc[ORDER]
                mean = z[metric+'_mean'].to_numpy()
                ax.plot(range(3), mean, '-o', color=color, label=label, lw=2)
                ax.fill_between(range(3), z[metric+'_min'], z[metric+'_max'], color=color, alpha=.1)
                for x, value in enumerate(mean):
                    ax.annotate(f'{value:.2f}' if metric == 'r2' or cohort == 'original' else f'{value:.1f}',
                                (x, value), xytext=(0, 9 if (model == 'geometry_9') == (metric == 'mae_minutes') else -15), textcoords='offset points', ha='center', fontsize=9, color=color)
            ax.set_xticks(range(3), ['0', '80', '240'])
            ax.set_xlabel('Combinations added to training')
            ax.set_ylabel(ylabel)
            ax.margins(x=.15, y=.25)
            ax.spines[['top', 'right']].set_visible(False)
            if i == 0:
                ax.set_title(title)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2)
    fig.suptitle('Augmentation improves more than relative error\nMeans across five repeated group splits; shading = repetition range, not a confidence interval', fontsize=13)
    fig.savefig(DEST/'mae_r2.png', dpi=180)
    plt.close(fig)

    data = read(SOURCE/'input.parquet')
    graphs = read(SCOPES/'domain_graphs.csv')
    eligible = data[data.edit_type.eq('Simple') & data.centroid_shift_hours.abs().ge(1/60)].merge(graphs[graphs.scope.eq('U')], on=['example', 'state'], validate='one_to_one')
    eligible = eligible[eligible.parent_links.le(12)].copy()
    median = eligible.centroid_shift_hours.abs().median()
    eligible['distance_to_median'] = abs(eligible.centroid_shift_hours.abs()-median)
    chosen = eligible.sort_values(['distance_to_median', 'example', 'state']).iloc[0]
    ex, state = chosen.example, chosen.state
    parent = read(SCOPES/'graphs'/f'{chosen.parent_graph}.parquet')
    child = read(SCOPES/'graphs'/f'{chosen.child_graph}.parquet')
    # This response-based choice happens to be a two-branch unit; draw its exact topology.
    assert len(parent) == 2 and len(child) == 1
    assert parent.id_us_node.nunique() == parent.id_ds_node.nunique() == 1
    metrics = read(EXPANSION/'state_metrics.parquet')
    metrics = metrics[metrics.example.eq(ex) & metrics.window.eq('0.5%')].set_index('state')
    pm, cm = metrics.loc['S000_base'], metrics.loc[state]
    curves = [read(Path(m.outlet)) for m in [pm, cm]]
    predicted = read(SOURCE/'predictions.parquet')
    predicted = predicted[predicted.example.eq(ex) & predicted.state.eq(state)].copy()
    assert len(predicted) == 30
    actual = 60*float(chosen.centroid_shift_hours)
    np.testing.assert_allclose(actual, (cm.centroid_seconds-pm.centroid_seconds)/60)
    means = predicted.groupby(['model', 'variant']).predicted_hours.agg(['mean', 'min', 'max'])*60
    means.to_csv(DEST/'unit_predictions_minutes.csv')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained', height_ratios=[.8, 1.2])
    for ax, frame, title, color in [(axes[0, 0], parent, 'Before: two parallel branches', '#546a7b'), (axes[0, 1], child, 'After: one equivalent channel', '#bb5832')]:
        ax.scatter([0, 1], [0, 0], s=130, color='#263843', zorder=4)
        ax.text(0, -.16, 'Inlet', ha='center')
        ax.text(1, -.16, 'Outlet', ha='center')
        for k, r in enumerate(frame.itertuples()):
            rad = (.55 if k == 0 else -.55) if len(frame) == 2 else 0
            ax.add_patch(FancyArrowPatch((0, 0), (1, 0), connectionstyle=f'arc3,rad={rad}', arrowstyle='-|>', mutation_scale=15, linewidth=2.5, color=color, shrinkA=9, shrinkB=9))
            y = (-.39 if k == 0 else .39) if len(frame) == 2 else .15
            ax.text(.5, y, f'Link {r.id_link}: {r.link_length_m:.0f} m long · {r.width:.0f} m wide', ha='center', va='center', fontsize=10,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.95, pad=2))
        ax.set_xlim(-.2, 1.2)
        ax.set_ylim(-.55, .55)
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    ax = axes[1, 0]
    origin = float(pm.rise_seconds)
    for frame, color, label in zip(curves, ['#546a7b', '#bb5832'], ['Parent network outlet', 'Outlet after this unit collapse']):
        seconds = frame.time_utc.astype('int64').to_numpy()/1e9
        mask = (seconds >= pm.rise_seconds) & (seconds <= pm.fall_seconds)
        ax.plot((seconds[mask]-origin)/3600, frame.q_outlet_cms.to_numpy()[mask], color=color, label=label, lw=1.8, linestyle='-' if color == '#546a7b' else '--')
    ax.set_xlabel('Hours from shared response-window start')
    ax.set_ylabel('Outlet discharge (m³/s)')
    ax.set_title('Routed hydrographs: nearly overlapping', fontsize=12)
    ax.legend(fontsize=8, loc='upper left')
    ax.text(.03, .05, f'Centroid arrives {abs(actual):.2f} min earlier\nafter this unit is collapsed.', transform=ax.transAxes, fontsize=10,
            bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
    ax.spines[['top', 'right']].set_visible(False)
    ax = axes[1, 1]
    ax.axvline(actual, color='#263843', linestyle='--', label=f'Routed target: {actual:.2f} min')
    for mi, (model, color) in enumerate(zip(MODELS, COLORS)):
        for vi, variant in enumerate(ORDER):
            row = means.loc[(model, variant)]
            y = 5-(mi*3+vi)
            ax.errorbar(row['mean'], y, xerr=[[row['mean']-row['min']], [row['max']-row['mean']]], fmt='o', color=color, capsize=4)
            ax.annotate(f"{row['mean']:.2f}", (row['mean'], y), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=9, color=color)
    ax.set_yticks(range(6), ['Depth +240', 'Depth +80', 'Depth baseline', 'Geometry +240', 'Geometry +80', 'Geometry baseline'])
    ax.set_ylim(-.6, 5.8)
    ax.margins(x=.3)
    ax.set_xlabel('Centroid shift (minutes; negative = earlier)')
    ax.set_title('Held-out timing predictions', fontsize=12)
    ax.legend(fontsize=9, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'One individual unit: network {ex}, {state}\nTopology schematic (not a map); timing points = mean, whiskers = range of five held-out predictions', fontsize=13)
    fig.savefig(DEST/'unit_example.png', dpi=180)
    plt.close(fig)
    selection = dict(example=ex, state=state, edit_type=chosen.edit_type, routed_centroid_shift_minutes=actual,
        eligible_units=len(eligible), selection='Closest absolute centroid shift to the median among simple units with >=1 minute effect and <=12 parent links. No prediction errors used to select this example.',
        interpretation='A typical non-negligible individual-unit edit; augmentation barely changes its predictions. Large gains are concentrated in segmented and combined states.', sources=SOURCES)
    (DEST/'unit_example.json').write_text(json.dumps(selection, indent=2)+'\n')
    print(json.dumps({k:v for k,v in selection.items() if k!='sources'}, indent=2))
    print(means.to_string())


if __name__ == '__main__':
    main()
