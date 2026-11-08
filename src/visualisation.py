"""Every figure in the project.

Matplotlib functions return a ``Figure`` for the notebooks; plotly functions
return a ``plotly.graph_objects.Figure`` for the Streamlit app. Nothing here calls
``plt.show()`` or writes files: use :func:`save_figure` for that.

Colours are Okabe-Ito (colour-blind safe) and fixed per decoder family, so greedy is the same
red in every chart.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from torch import Tensor

from src.analysis.distribution import probs_from_logits
from src.analysis.diversity import pareto_front
from src.config import get_settings

if TYPE_CHECKING:
    from src.model.logits import LanguageModel

FAMILY_COLOURS: dict[str, str] = {
    "greedy": "#D55E00",  # vermillion: the cautionary tale
    "penalty": "#CC79A7",  # reddish purple
    "beam": "#0072B2",  # blue
    "sampling": "#E69F00",  # orange
    "top-k": "#56B4E9",  # sky blue
    "nucleus": "#009E73",  # bluish green: the modern default
}
# Series colours for line charts: adjacent entries are easy to tell apart, black for emphasis.
SEQUENTIAL = ["#0072B2", "#E69F00", "#000000", "#CC79A7", "#56B4E9", "#009E73", "#D55E00"]
MARKERS = ["o", "s", "D", "^", "v", "P", "X"]
NEUTRAL = "#999999"
BAND = "#009E73"


def decoder_family(label: str) -> str:
    """Map a decoder label (``"greedy, rep=1.3"``, ``"T=1, top-p=0.8"``) to its family."""
    if label.startswith("beam"):
        return "beam"
    if label.startswith("greedy"):
        return "penalty" if "rep=" in label or "presence" in label else "greedy"
    if "top-p" in label:
        return "nucleus"
    if "top-k" in label:
        return "top-k"
    return "sampling"


def decoder_colour(label: str) -> str:
    return FAMILY_COLOURS[decoder_family(label)]


def _style(ax: Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)


def _figure(nrows: int = 1, ncols: int = 1, figsize: tuple[float, float] = (8, 4.5)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, layout="constrained", squeeze=False)
    for ax in axes.flat:
        _style(ax)
    return fig, axes


def save_figure(fig: Figure, name: str, directory: Path | None = None) -> Path:
    """Write ``fig`` to ``<figures dir>/<name>.png`` at 150 dpi and return the path."""
    out_dir = directory or get_settings().figures_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


# --- distributions --------------------------------------------------------------------------


def plot_top_tokens(df: pd.DataFrame, title: str = "Next-token distribution") -> Figure:
    """Horizontal bars of the most likely next tokens (a :func:`top_tokens` table)."""
    fig, axes = _figure(figsize=(7, max(3.0, 0.28 * len(df) + 1)))
    ax = axes[0, 0]
    ordered = df.iloc[::-1]
    ax.barh(ordered["token"], ordered["prob"], color=FAMILY_COLOURS["beam"])
    ax.set_xlabel("probability")
    ax.set_title(title, loc="left")
    ax.grid(axis="x", alpha=0.3)
    ax.grid(axis="y", visible=False)
    return fig


def plot_reshaped_distributions(
    model: LanguageModel,
    logits: Tensor,
    warped: Mapping[str, Tensor],
    n: int = 15,
    title: str = "One distribution, reshaped by the decoder",
) -> Figure:
    """Small multiples: the same top-``n`` tokens under each warp. Removed tokens are marked x.

    Args:
        model: supplies token labels.
        logits: ``(1, vocab)`` raw logits; their top ``n`` tokens fix the x-axis for every panel.
        warped: panel title -> ``(1, vocab)`` warped logits.
        n: tokens shown.
        title: figure title.
    """
    raw = probs_from_logits(logits.reshape(-1))
    ids = raw.topk(min(n, raw.numel())).indices.tolist()
    labels = model.token_labels(ids)
    ncols = min(3, len(warped))
    nrows = math.ceil(len(warped) / ncols)
    fig, axes = _figure(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows))
    for ax, (name, w) in zip(axes.flat, warped.items(), strict=False):
        row = w.reshape(-1)
        probs = probs_from_logits(row)[ids].tolist()
        removed = [not math.isfinite(v) for v in row[ids].tolist()]
        ax.bar(range(len(ids)), probs, color=FAMILY_COLOURS["nucleus"])
        cut = [x for x, r in enumerate(removed) if r]
        ax.plot(
            cut,
            [0.05] * len(cut),
            ls="none",
            marker="x",
            markersize=7,
            markeredgewidth=2,
            color=FAMILY_COLOURS["greedy"],
        )
        ax.set_xticks(range(len(ids)), labels, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_title(name, fontsize=11)
    for ax in list(axes.flat)[len(warped) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("probability")
    fig.suptitle(title, x=0.01, ha="left")
    fig.text(
        0.99,
        0.995,
        "× = removed by the decoder",
        ha="right",
        va="top",
        color=FAMILY_COLOURS["greedy"],
        fontsize=10,
    )
    return fig


def plot_entropy_vs_temperature(
    sweep: pd.DataFrame,
    vocab_size: int,
    band: tuple[float, float] | None = None,
    title: str = "Temperature flattens the distribution",
) -> Figure:
    """Entropy (bits) against temperature, with the uniform ceiling ``log2(vocab)``.

    The x-axis is symmetric-log so the region that matters (T below ~2) is not squashed by
    the long approach to uniform.

    Args:
        sweep: a :func:`temperature_sweep` table.
        vocab_size: for the uniform-distribution ceiling.
        band: optional ``(low, high)`` temperature range to shade as the usable band.
        title: figure title.
    """
    fig, axes = _figure()
    ax = axes[0, 0]
    ceiling = math.log2(vocab_size)
    if band is not None:
        ax.axvspan(*band, color=BAND, alpha=0.15, label="usable band")
    ax.axhline(ceiling, ls="--", color=NEUTRAL, label=f"uniform over {vocab_size} tokens")
    ax.plot(
        sweep["temperature"],
        sweep["entropy_bits"],
        marker="o",
        color=FAMILY_COLOURS["beam"],
        label="entropy of the next-token distribution",
    )
    _symlog_temperature_axis(ax, sweep["temperature"])
    ax.set_ylim(0, ceiling * 1.12)
    ax.set_xlabel("temperature T (symmetric log scale)")
    ax.set_ylabel("entropy (bits)")
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, loc="lower right")
    return fig


def _symlog_temperature_axis(ax: Axes, temperatures: pd.Series) -> None:
    ax.set_xscale("symlog", linthresh=0.25, linscale=0.6)
    candidates = [0, 0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 20, 50, 100]
    hi = float(temperatures.max())
    ticks = [t for t in candidates if t <= hi]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.set_xlim(-0.02, hi * 1.15)


def plot_metrics_vs(
    df: pd.DataFrame,
    x: str,
    ys: Mapping[str, str],
    *,
    xlabel: str,
    ylabel: str = "",
    title: str = "",
    band: tuple[float, float] | None = None,
    band_label: str = "usable band",
    regions: Mapping[str, float] | None = None,
    logx: bool = False,
    annotate: Mapping[str, float] | None = None,
    legend_loc: str = "best",
) -> Figure:
    """Several metric columns against one setting (temperature, beam width, penalty, ...).

    Args:
        df: one row per setting value.
        x: column on the x-axis.
        ys: column -> legend label. Each series gets its own colour and marker.
        xlabel / ylabel / title: text.
        band: optional x-range to shade.
        band_label: legend text for the band.
        regions: optional label -> x position, written along the top of the plot.
        logx: symmetric-log x-axis (for temperature sweeps).
        annotate: column -> vertical offset in points (negative = below) for value labels.
        legend_loc: matplotlib legend location.
    """
    fig, axes = _figure()
    ax = axes[0, 0]
    if band is not None:
        ax.axvspan(*band, color=BAND, alpha=0.15, label=band_label)
    for i, (col, label) in enumerate(ys.items()):
        ax.plot(
            df[x],
            df[col],
            marker=MARKERS[i % len(MARKERS)],
            label=label,
            color=SEQUENTIAL[i % len(SEQUENTIAL)],
            lw=2.2 if i == 2 else 1.6,
        )
    if logx:
        _symlog_temperature_axis(ax, df[x])
    for col, dy in (annotate or {}).items():
        for xv, yv in zip(df[x], df[col], strict=True):
            ax.annotate(
                f"{yv:.2f}",
                (xv, yv),
                textcoords="offset points",
                xytext=(0, dy),
                ha="center",
                va="center",
                fontsize=9,
            )
    for text, xpos in (regions or {}).items():
        ax.text(
            xpos,
            1.01,
            text,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=10,
            color="#444444",
            style="italic",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=22 if regions else 6)
    ax.legend(frameon=False, loc=legend_loc)
    return fig


def plot_kept_set_sizes(
    df: pd.DataFrame, title: str = "Fixed k ignores confidence; nucleus adapts"
) -> Figure:
    """Tokens kept by top-k and top-p on each distribution (a :func:`kept_set_sizes` table).

    Series are drawn with nested markers of decreasing size, so lines that coincide exactly
    (top-k keeps the same count on every distribution) stay visible as concentric markers.
    """
    fig, axes = _figure(1, 2, figsize=(11, 4.4))
    for ax, method in zip(axes[0], ["top-k", "top-p"], strict=True):
        sub = df[df["method"] == method]
        groups = list(sub.groupby("distribution", sort=False))
        for i, (name, group) in enumerate(groups):
            ax.plot(
                group["setting"],
                group["kept"],
                marker=MARKERS[i % len(MARKERS)],
                markersize=11 - 3 * i,
                color=SEQUENTIAL[i % len(SEQUENTIAL)],
                label=str(name),
                lw=1.5,
            )
        counts = [tuple(g["kept"]) for _, g in groups]
        if len(set(counts)) == 1:
            ax.text(
                0.97,
                0.05,
                f"identical on all {len(groups)} distributions",
                transform=ax.transAxes,
                ha="right",
                fontsize=10,
                style="italic",
                color="#444444",
            )
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if method == "top-k":
            ax.set_xticks(sorted(sub["setting"].unique()))  # tick exactly the k values tried
        ax.set_xlabel("k" if method == "top-k" else "p")
        ax.set_ylabel("tokens kept")
        ax.set_title(method, loc="left")
        ax.legend(frameon=False, loc="upper left")
    fig.suptitle(title, x=0.01, ha="left")
    return fig


# --- decoder comparisons --------------------------------------------------------------------


def plot_decoder_bars(
    summary: pd.DataFrame,
    metric: str,
    *,
    ylabel: str,
    title: str = "",
    ylim: tuple[float, float] | None = None,
) -> Figure:
    """One bar per decoder for ``metric``, coloured by decoder family."""
    fig, axes = _figure(figsize=(9, 4))
    ax = axes[0, 0]
    labels = [str(i) for i in summary.index]
    bars = ax.bar(labels, summary[metric], color=[decoder_colour(lab) for lab in labels])
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="x", rotation=25)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    return fig


def plot_grouped_bars(
    summary: pd.DataFrame,
    columns: Mapping[str, str],
    *,
    title: str = "",
    ylabel: str = "",
    ylim: tuple[float, float] = (0, 1.3),
) -> Figure:
    """Grouped bars: one group per decoder (row), one bar per metric column, values labelled.

    Args:
        summary: one row per decoder.
        columns: column -> legend label.
        title / ylabel: text.
        ylim: y-axis range (headroom for the value labels).
    """
    fig, axes = _figure(figsize=(10, 4.6))
    ax = axes[0, 0]
    labels = [str(i) for i in summary.index]
    n = len(columns)
    width = 0.8 / n
    for j, (col, label) in enumerate(columns.items()):
        xs = [i + (j - (n - 1) / 2) * width for i in range(len(labels))]
        bars = ax.bar(xs, summary[col], width, label=label, color=SEQUENTIAL[j])
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    ax.set_ylim(*ylim)
    ax.set_yticks([t / 5 for t in range(6)])  # 0..1; the headroom above holds the legend
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, loc="upper left", ncols=n)
    return fig


def plot_task_tradeoff(
    summary: pd.DataFrame,
    title: str = "The decoder that wins one task loses the other",
) -> Figure:
    """Single-answer accuracy next to the open-ended score, per decoder."""
    return plot_grouped_bars(
        summary,
        {
            "single_accuracy": "single-answer accuracy",
            "open_score": "open-ended score (fluent x varied)",
        },
        title=title,
        ylabel="score (0 to 1)",
    )


def _clusters(xs: Sequence[float], ys: Sequence[float], radius: float) -> list[list[int]]:
    """Group points closer than ``radius`` in axis-normalised units (single linkage)."""
    x_span = max(max(xs) - min(xs), 1e-9)
    y_span = max(max(ys) - min(ys), 1e-9)
    norm = [((a - min(xs)) / x_span, (b - min(ys)) / y_span) for a, b in zip(xs, ys, strict=True)]
    label = list(range(len(xs)))
    for i in range(len(xs)):
        for j in range(i):
            near = abs(norm[i][0] - norm[j][0]) < radius and abs(norm[i][1] - norm[j][1]) < radius
            if near and label[i] != label[j]:
                old, new = label[i], label[j]
                label = [new if c == old else c for c in label]
    groups: dict[int, list[int]] = {}
    for i, c in enumerate(label):
        groups.setdefault(c, []).append(i)
    return list(groups.values())


def _numbered_labels(ax: Axes, xs: Sequence[float], ys: Sequence[float]) -> None:
    """Number each point; crowded clusters get a column of numbers with leader lines.

    A lone point is numbered just above-right. A cluster's numbers are stacked in a column
    beside it, on the side facing the middle of the plot (where there is room), ordered top to
    bottom like the points, each joined to its own marker by a thin line.
    """
    x_span = max(max(xs) - min(xs), 1e-9)
    y_span = max(max(ys) - min(ys), 1e-9)
    x_mid = (max(xs) + min(xs)) / 2
    top = max(ys)
    for members in _clusters(xs, ys, radius=0.06):
        if len(members) == 1:
            i = members[0]
            ax.annotate(
                str(i + 1),
                (xs[i], ys[i]),
                textcoords="offset points",
                xytext=(7, 6),
                fontsize=11,
                fontweight="bold",
            )
            continue
        members = sorted(members, key=lambda i: -ys[i])
        cx = sum(xs[i] for i in members) / len(members)
        cy = sum(ys[i] for i in members) / len(members)
        side = 1 if cx < x_mid else -1
        edge = max(xs[i] for i in members) if side > 0 else min(xs[i] for i in members)
        label_x = edge + side * 0.07 * x_span
        step = 0.075 * y_span
        for rank, i in enumerate(members):
            label_y = cy + ((len(members) - 1) / 2 - rank) * step
            top = max(top, label_y)
            ax.annotate(
                str(i + 1),
                (xs[i], ys[i]),
                xytext=(label_x, label_y),
                textcoords="data",
                ha="left" if side > 0 else "right",
                va="center",
                fontsize=11,
                fontweight="bold",
                arrowprops={
                    "arrowstyle": "-",
                    "color": NEUTRAL,
                    "lw": 0.8,
                    "shrinkA": 3,
                    "shrinkB": 6,
                },
            )
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, max(hi, top + 0.06 * y_span))
    # The extra headroom is for labels only: tick up to the first value at or above the data.
    ticks = sorted(t for t in ax.get_yticks() if t >= lo)
    ceiling = next((t for t in ticks if t >= max(ys) - 1e-9), ticks[-1])
    ax.set_yticks([t for t in ticks if t <= ceiling])


def plot_quality_diversity(
    summary: pd.DataFrame,
    x: str = "open_diversity",
    y: str = "open_valid_words",
    *,
    xlabel: str = "diversity (distinct-2 pooled across samples)",
    ylabel: str = "fluency (share of real words)",
    title: str = "Quality versus diversity",
    label_offsets: Mapping[str, tuple[float, float]] | None = None,
    front: bool = False,
    numbered: bool = False,
) -> Figure:
    """Scatter of decoders on two metrics, coloured by family.

    Args:
        summary: one row per decoder.
        x / y: metric columns.
        xlabel / ylabel / title: text.
        label_offsets: per-decoder label offset in points, where inline labels collide.
        front: connect the Pareto-optimal decoders (both metrics maximised) with a dashed line.
        numbered: for crowded charts, mark points with numbers and list the decoders in a
            legend beside the plot instead of labelling inline. Numbers of near-coincident
            points are staggered automatically.
    """
    fig, axes = _figure(figsize=(11, 5.5) if numbered else (8, 5.5))
    ax = axes[0, 0]
    ax.grid(axis="x", alpha=0.3)
    offsets = label_offsets or {}
    names = [str(label) for label in summary.index]
    xs = summary[x].astype(float).tolist()
    ys = summary[y].astype(float).tolist()
    for i, name in enumerate(names):
        legend_label = f"{i + 1}. {name}" if numbered else None
        ax.scatter(xs[i], ys[i], s=90, color=decoder_colour(name), zorder=3, label=legend_label)
        if not numbered:
            ax.annotate(
                name,
                (xs[i], ys[i]),
                textcoords="offset points",
                xytext=offsets.get(name, (7, 5)),
                fontsize=10,
            )
    if numbered:
        _numbered_labels(ax, xs, ys)
    if front:
        best = pareto_front(summary, x=x, y=y)
        ax.plot(best[x], best[y], ls="--", color=NEUTRAL, zorder=2, label="Pareto front")
    if numbered:
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=10)
    elif front:
        ax.legend(frameon=False, loc="lower left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    return fig


def plot_token_trace(
    df: pd.DataFrame,
    loop_start: int | None = None,
    title: str = "How sure the model was at every step",
    loop_label: str = "loop",
) -> Figure:
    """Probability of each chosen token along a generation (a :func:`token_probabilities` table).

    Args:
        df: per-step table.
        loop_start: generated position where a loop begins; shaded from there on.
        title: figure title.
        loop_label: legend text for the shaded region.
    """
    fig, axes = _figure(figsize=(10, 3.8))
    ax = axes[0, 0]
    ax.plot(df["position"], df["p_chosen"], color=FAMILY_COLOURS["greedy"], lw=1.2)
    if loop_start is not None:
        ax.axvspan(loop_start, df["position"].max(), color=NEUTRAL, alpha=0.15, label=loop_label)
        ax.legend(frameon=False, loc="lower right")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("generated character")
    ax.set_ylabel("p(chosen token)")
    ax.set_title(title, loc="left")
    return fig


def plot_training_loss(losses: Sequence[float], title: str = "Char model training loss") -> Figure:
    """Cross-entropy per step (nats), smoothed over a short window."""
    fig, axes = _figure(figsize=(7, 3.5))
    ax = axes[0, 0]
    series = pd.Series(losses)
    ax.plot(series.index, series, color=NEUTRAL, alpha=0.4, lw=0.8, label="per step")
    ax.plot(series.index, series.rolling(50, min_periods=1).mean(), color="#0072B2", label="mean")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy (nats)")
    ax.set_title(title, loc="left")
    ax.legend(frameon=False)
    return fig


# --- plotly, for the app --------------------------------------------------------------------


def plotly_top_tokens(
    df: pd.DataFrame, title: str = "", removed: Sequence[bool] = ()
) -> go.Figure:
    """Bar chart of a :func:`top_tokens` table; tokens flagged in ``removed`` are greyed out."""
    flags = list(removed) or [False] * len(df)
    colours = [NEUTRAL if r else FAMILY_COLOURS["nucleus"] for r in flags]
    fig = go.Figure(go.Bar(x=df["token"], y=df["prob"], marker_color=colours))
    fig.update_layout(
        title=title,
        xaxis_title="token",
        yaxis_title="probability",
        yaxis_range=[0, 1],
        margin={"l": 40, "r": 10, "t": 40, "b": 40},
        height=320,
    )
    return fig


def plotly_lines(
    df: pd.DataFrame,
    x: str,
    ys: Mapping[str, str],
    *,
    title: str = "",
    xlabel: str = "",
    vline: float | None = None,
) -> go.Figure:
    """Line chart of several columns; optional vertical marker (e.g. the current setting)."""
    fig = go.Figure()
    for colour, (col, label) in zip(SEQUENTIAL, ys.items(), strict=False):
        fig.add_trace(
            go.Scatter(x=df[x], y=df[col], mode="lines+markers", name=label, line_color=colour)
        )
    if vline is not None:
        fig.add_vline(x=vline, line_dash="dash", line_color=NEUTRAL)
    fig.update_layout(
        title=title, xaxis_title=xlabel, margin={"l": 40, "r": 10, "t": 40, "b": 40}, height=340
    )
    return fig


def plotly_decoder_bars(summary: pd.DataFrame, metric: str, title: str = "") -> go.Figure:
    """One bar per decoder for ``metric``, coloured by family."""
    labels = [str(i) for i in summary.index]
    fig = go.Figure(
        go.Bar(x=labels, y=summary[metric], marker_color=[decoder_colour(lab) for lab in labels])
    )
    fig.update_layout(title=title, margin={"l": 40, "r": 10, "t": 40, "b": 80}, height=340)
    return fig
