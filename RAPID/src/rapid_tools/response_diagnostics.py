"""Signed timing plots, explicit score scales, and conditional response curves."""
from __future__ import annotations

import html
import uuid

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def _r2(y, predicted):
    y, predicted = np.asarray(y), np.asarray(predicted)
    if len(y) < 2 or not np.isfinite(predicted).all() or np.ptp(y) == 0:
        return float("nan")
    return float(r2_score(y, predicted))


def response_scores(observed, predicted, *, scale=1.0):
    """Score identical predictions; magnitude scores do not assess response sign.

    scale is in response units (one percentage point by default). Zero pairs
    are excluded only from ordinary log10 magnitude scores, never offset.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    y, p = np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    nonzero = (y != 0) & (p != 0)
    transform = lambda v: np.sign(v) * np.log1p(np.abs(v) / scale)
    return {
        "r2": _r2(y, p),
        "r2_linear": _r2(y, p),
        "r2_signed_log1p": _r2(transform(y), transform(p)),
        "signed_log_scale": scale,
        "r2_log10_magnitude": _r2(np.log10(np.abs(y[nonzero])), np.log10(np.abs(p[nonzero]))),
        "r2_linear_log_subset": _r2(y[nonzero], p[nonzero]),
        "n_scored": len(y),
        "n_nonfinite_excluded": int((~valid).sum()),
        "n_log_magnitude": int(nonzero.sum()),
        "n_zero_pairs_excluded": int((~nonzero).sum()),
        "sign_accuracy": float(np.mean(np.sign(y) == np.sign(p))) if len(y) else np.nan,
    }


def speed_change_pct(lag_change_pct):
    """Fixed-distance apparent peak-speed change; assumes positive base lag."""
    r = pd.to_numeric(lag_change_pct, errors="coerce") / 100.0
    return 100.0 * (1.0 / (1.0 + r.where(np.isfinite(r) & (r > -1))) - 1.0)


CATEGORY_NAMES = {
    "nested unit": "Nested",
    "part of compound unit": "Part of compound",
    "single simple": "Single unit",
    "compound indep": "Compound unit",
    "compound unit compound": "Compound unit",
}
CATEGORY_COLORS = {"Nested": "#2ca02c", "Part of compound": "#ff7f0e",
                   "Single unit": "#1f77b4", "Compound unit": "#d62728"}


RESISTANCE_AXIS_LABEL = "<i>L</i><sub>f</sub> · <i>E</i> · [−log<sub>10</sub>(Λ<sub>K</sub>)]"


def make_signed_magnitude_scatter(frame, *, x, y, log_x=False, log_y=True,
                                  response_name=None, title=None, x_label=None, y_label=None):
    """Signed linear or magnitude/log views with independent sign metadata.

    The resistance legend refers to -log10(Lambda_K), not Lambda_K itself.
    Speed uses the actual speed-change sign; faster states keep the circle
    used for earlier-lag states. Zeros are moved only on logarithmic axes.
    """
    response_name = response_name or ("Speed" if "speed" in y.lower() else "Lag")
    speed = response_name.lower() == "speed"
    df = frame.copy()
    df[x], df[y] = pd.to_numeric(df[x], errors="coerce"), pd.to_numeric(df[y], errors="coerce")
    df = df.loc[np.isfinite(df[x]) & np.isfinite(df[y])].copy()
    df["_category"] = df["plot_category"].map(CATEGORY_NAMES).fillna(df["plot_category"])
    sign = lambda v: "+" if v > 0 else ("−" if v < 0 else "0")
    resistance = df[x]
    if "Lambda_matched" in df:
        ratio = pd.to_numeric(df["Lambda_matched"], errors="coerce")
        resistance = -np.log10(ratio.where(np.isfinite(ratio) & ratio.gt(0)))
        resistance = resistance.fillna(df[x])
    df["_resistance_sign"] = resistance.map(sign)
    df["_response_sign"] = df[y].map(sign)
    floors = {}
    for col, logged in ((x, log_x), (y, log_y)):
        values = df[col].abs() if logged else df[col]
        positive = values[values > 0]
        floor = float(positive.min() / 10) if len(positive) else 1e-6
        floors[col] = floor
        df[f"_display_{col}"] = values.mask(values.eq(0), floor) if logged else values
    fig = go.Figure()
    hover_cols = [c for c in ["example_id", "state_id", "target_id", "mode", x, y, "Lambda_matched"] if c in df]
    hover = "<br>".join(f"{html.escape(c)}: %{{customdata[{i}]}}" for i, c in enumerate(hover_cols))
    for (category, resistance_sign, response_sign), part in df.groupby(
            ["_category", "_resistance_sign", "_response_sign"], sort=True):
        ordinary = resistance_sign == "+" and response_sign == ("+" if speed else "−")
        signs = f"x {resistance_sign}, {response_name.lower()} {response_sign}"
        fig.add_trace(go.Scatter(
            x=part[f"_display_{x}"], y=part[f"_display_{y}"], mode="markers",
            name=f"{category} · {signs}",
            meta={"category": category, "sign": signs,
                  "resistance_sign": resistance_sign, "response_sign": response_sign},
            marker=dict(symbol="circle" if ordinary or not (log_x or log_y) else "diamond", size=9,
                        color=CATEGORY_COLORS.get(category, "#777"), line=dict(width=0.7, color="black")),
            customdata=part[hover_cols].to_numpy(), hovertemplate=hover + "<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(template="plotly_white", title=title or "Resistance–response",
                      meta={"response_name": response_name}, showlegend=False,
                      height=570, margin=dict(b=70, r=25))
    fig.update_xaxes(type="log" if log_x else "linear", title=x_label or RESISTANCE_AXIS_LABEL)
    fig.update_yaxes(type="log" if log_y else "linear",
                     title=y_label or ("Speed change (%)" if speed else "Lag shift (%)"))
    for col, logged in ((x, log_x), (y, log_y)):
        if logged and df[col].eq(0).any():
            floor = floors[col]
            axis = fig.update_xaxes if col == x else fig.update_yaxes
            maximum = max(float(df[col].abs().max()), floor)
            powers = np.arange(np.ceil(np.log10(floor * 10)), np.floor(np.log10(maximum)) + 1)
            ticks = [floor, *np.power(10.0, powers)]
            axis(tickmode="array", tickvals=ticks, ticktext=["0 (display)", *[f"{t:g}" for t in ticks[1:]]])
    return fig


def signed_scatter_html(fig, *, include_plotlyjs=True):
    """Three stacked legends; category, resistance sign and response sign intersect.

    Keep the controls beside the chart like a Plotly legend, without requiring
    the multiple-native-legend feature (unavailable in older Plotly versions).
    """
    uid = "response_" + uuid.uuid4().hex
    categories = list(dict.fromkeys(t.meta["category"] for t in fig.data))
    response_name = (fig.layout.meta or {}).get("response_name", "Lag")
    panels = []
    for key, heading, labels in (
        ("category", "Experiment category", categories),
        ("resistance_sign", "Sign of −log<sub>10</sub>(Λ<sub>K</sub>)", ["+", "−", "0"]),
        ("response_sign", f"{html.escape(response_name)} sign", ["+", "−", "0"]),
    ):
        controls = []
        for label in labels:
            color = CATEGORY_COLORS.get(label, "#444")
            symbol = f'<span style="color:{color}">●</span> ' if key == "category" else ""
            controls.append(f'<label class="response-option"><input type="checkbox" checked data-filter="{key}" value="{html.escape(label, quote=True)}">{symbol}{html.escape(label)}</label>')
        layout = "display:flex;flex-direction:column;gap:8px" if key == "category" else "display:flex;gap:18px"
        panels.append(f'<section style="margin-bottom:24px"><div style="font-weight:600;margin-bottom:10px">{heading}</div><div style="{layout}">{"".join(controls)}</div></section>')
    script = """
    const plot = document.getElementById('{plot_id}');
    const controls = document.getElementById('CONTROLS');
    controls.querySelectorAll('input').forEach(box => box.addEventListener('change', () => {
      const selected = key => new Set(Array.from(controls.querySelectorAll('input[data-filter="' + key + '"]:checked')).map(b => b.value));
      const categories = selected('category');
      const resistance = selected('resistance_sign'), response = selected('response_sign');
      Plotly.restyle(plot, {visible: plot.data.map(t => categories.has(t.meta.category) && resistance.has(t.meta.resistance_sign) && response.has(t.meta.response_sign))});
    }));
    """.replace("CONTROLS", uid)
    chart = fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs, post_script=script)
    style = f"""<style>
    #{uid}_wrap {{display:grid;grid-template-columns:minmax(0,1fr) 210px;gap:12px;align-items:start}}
    #{uid} {{font:13px Arial,sans-serif;color:#2a3f5f;padding-top:58px}}
    #{uid} .response-option {{cursor:pointer;white-space:nowrap}}
    #{uid} input {{accent-color:#60788e;cursor:pointer;margin-right:6px}}
    @media(max-width:700px) {{#{uid}_wrap {{grid-template-columns:minmax(0,1fr)}} #{uid} {{padding:0 35px}}}}
    </style>"""
    return style + f'<div id="{uid}_wrap"><aside id="{uid}" style="order:2">{"".join(panels)}</aside><div style="min-width:0;order:1">{chart}</div></div>'


def show_signed_scatter(fig):
    from IPython.display import HTML, display
    display(HTML(signed_scatter_html(fig)))


def fit_response_relationships(frame, *, x, y, group="example_id"):
    """Compare curves within each original-sign quadrant using grouped CV.

    All models minimize squared log10-magnitude errors. Zeros are excluded.
    The upper-tail omission is a sensitivity experiment, not an estimate of an
    unobserved tail or a fit to a power-law probability distribution.
    """
    data = frame.copy()
    data[x], data[y] = pd.to_numeric(data[x], errors="coerce"), pd.to_numeric(data[y], errors="coerce")
    data = data.loc[np.isfinite(data[x]) & np.isfinite(data[y]) & data[x].ne(0) & data[y].ne(0) & data[group].notna()]
    scores, predictions, parameters = [], [], []
    for (sx, sy), part in data.groupby([np.sign(data[x]), np.sign(data[y])]):
        u, v = part[x].abs().to_numpy(), part[y].abs().to_numpy()
        groups = part[group].astype(str).to_numpy()
        log_u, log_v = np.log10(u), np.log10(v)
        quadrant = f"x {'+' if sx > 0 else '-'}, lag {'+' if sy > 0 else '-'}"
        designs = {"power_law": log_u[:, None], "exponential": u[:, None],
                   "quadratic_log": np.column_stack([log_u, log_u**2])}
        for name, design in designs.items():
            if len(u) <= design.shape[1] + 1 or np.linalg.matrix_rank(np.column_stack([np.ones(len(u)), design])) < design.shape[1] + 1:
                continue
            fitted = LinearRegression().fit(design, log_v)
            p_log = np.full(len(u), np.nan)
            for held_out in np.unique(groups):
                test = groups == held_out
                train = ~test
                if train.sum() <= design.shape[1] + 1 or np.linalg.matrix_rank(np.column_stack([np.ones(train.sum()), design[train]])) < design.shape[1] + 1:
                    continue
                model = LinearRegression().fit(design[train], log_v[train])
                p_log[test] = model.predict(design[test])
            with np.errstate(over="ignore", under="ignore"):
                pred = 10.0**p_log
            valid = np.isfinite(p_log) & np.isfinite(pred) & (pred > 0)
            complete = valid.all()
            scores.append({"quadrant": quadrant, "model": name, "n": len(u), "n_groups": len(np.unique(groups)),
                           "n_cv": int(valid.sum()), "r2_log10_cv": _r2(log_v, p_log) if complete else np.nan,
                           "r2_linear_cv": _r2(v, pred) if complete else np.nan,
                           "r2_log10_in_sample": _r2(log_v, fitted.predict(design))})
            parameters.append({"quadrant": quadrant, "model": name, "intercept_log10": fitted.intercept_,
                               "b": fitted.coef_[0], "c": fitted.coef_[1] if len(fitted.coef_) > 1 else 0,
                               "x_min": u.min(), "x_max": u.max()})
            for idx, a, b, pp in zip(part.index, u, v, pred):
                predictions.append({"row_index": idx, "quadrant": quadrant, "model": name,
                                    "x_magnitude": a, "observed_magnitude": b, "predicted_magnitude": pp})
        # Deliberately remove the top x decile to measure slope sensitivity.
        keep = u <= np.quantile(u, 0.9)
        if keep.sum() >= 3 and np.ptp(log_u[keep]) > 0:
            fitted = LinearRegression().fit(log_u[keep, None], log_v[keep])
            parameters.append({"quadrant": quadrant, "model": "power_law_without_upper_x_decile",
                               "intercept_log10": fitted.intercept_, "b": fitted.coef_[0], "c": 0,
                               "x_min": u[keep].min(), "x_max": u[keep].max()})
    return pd.DataFrame(scores), pd.DataFrame(parameters), pd.DataFrame(predictions)
