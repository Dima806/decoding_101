"""Every plotting function renders without error and returns the right figure type."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import pytest
import torch
from matplotlib.figure import Figure

from src import visualisation as viz
from src.analysis import kept_set_sizes, temperature_sweep, token_probabilities, top_tokens
from src.decoders import Decoder


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def summary() -> pd.DataFrame:
    labels = ["greedy", "greedy, rep=1.3", "beam (B=4)", "T=1", "T=1, top-k=5", "T=1, top-p=0.8"]
    return pd.DataFrame(
        {
            "open_distinct_2": [0.3, 0.7, 0.6, 0.9, 0.85, 0.88],
            "open_diversity": [0.06, 0.15, 0.12, 0.8, 0.7, 0.75],
            "open_valid_words": [1.0, 1.0, 1.0, 0.8, 0.95, 0.97],
            "open_score": [0.3, 0.7, 0.6, 0.72, 0.8, 0.85],
            "single_accuracy": [1.0, 1.0, 1.0, 0.9, 0.95, 1.0],
        },
        index=pd.Index(labels, name="decoder"),
    )


def test_decoder_family_and_colour() -> None:
    assert viz.decoder_family("greedy") == "greedy"
    assert viz.decoder_family("greedy, rep=1.3") == "penalty"
    assert viz.decoder_family("beam (B=4)") == "beam"
    assert viz.decoder_family("T=0.7, top-p=0.9") == "nucleus"
    assert viz.decoder_family("T=1, top-k=5") == "top-k"
    assert viz.decoder_family("T=2") == "sampling"
    assert viz.decoder_colour("greedy").startswith("#")


def test_distribution_plots(char_lm, tmp_path: Path) -> None:
    logits = char_lm.next_token_logits(char_lm.encode("the sea"))
    df = top_tokens(char_lm, logits, n=10)
    figs = [
        viz.plot_top_tokens(df),
        viz.plot_reshaped_distributions(
            char_lm,
            logits,
            {"raw": logits, "top-k 3": Decoder(top_k=3).warp(logits, torch.zeros(1, 1).long())},
        ),
        viz.plot_entropy_vs_temperature(
            temperature_sweep(logits, [0.0, 1.0, 3.0]), char_lm.vocab_size, band=(0.5, 1.0)
        ),
        viz.plot_kept_set_sizes(kept_set_sizes({"a": logits}, ks=[1, 5], ps=[0.5, 0.9])),
        viz.plot_training_loss(char_lm.train_loss),
        viz.plot_token_trace(token_probabilities(char_lm, char_lm.encode("the sea"), 2), 1),
    ]
    assert all(isinstance(f, Figure) for f in figs)
    path = viz.save_figure(figs[0], "smoke", directory=tmp_path)
    assert path == tmp_path / "smoke.png"
    assert path.stat().st_size > 0


def test_comparison_plots(summary: pd.DataFrame) -> None:
    sweep = pd.DataFrame({"t": [0.0, 1.0, 2.0], "a": [0.1, 0.5, 0.9], "b": [1.0, 0.9, 0.2]})
    figs = [
        viz.plot_decoder_bars(summary, "open_distinct_2", ylabel="distinct-2", ylim=(0, 1)),
        viz.plot_task_tradeoff(summary),
        viz.plot_grouped_bars(
            summary, {"open_distinct_2": "varied", "open_valid_words": "fluent"}
        ),
        viz.plot_quality_diversity(summary),
        viz.plot_quality_diversity(summary, front=True, label_offsets={"greedy": (-5, -15)}),
        viz.plot_quality_diversity(summary, front=True, numbered=True),
        viz.plot_metrics_vs(sweep, "t", {"a": "A", "b": "B"}, xlabel="T", band=(0.5, 1.5)),
        viz.plot_metrics_vs(sweep, "t", {"a": "A"}, xlabel="T", logx=True),
    ]
    assert all(isinstance(f, Figure) for f in figs)


def test_plotly_figures(summary: pd.DataFrame, char_lm) -> None:
    df = top_tokens(char_lm, char_lm.next_token_logits(char_lm.encode("the")), n=5)
    sweep = pd.DataFrame({"t": [0.0, 1.0], "a": [0.1, 0.5]})
    figs = [
        viz.plotly_top_tokens(df, "raw", removed=[False, True, False, False, True]),
        viz.plotly_lines(sweep, "t", {"a": "A"}, vline=0.5),
        viz.plotly_decoder_bars(summary, "open_score"),
    ]
    assert all(isinstance(f, go.Figure) for f in figs)


def test_clusters_group_only_near_points() -> None:
    groups = viz._clusters([0.720, 0.725, 0.1, 0.99], [0.908, 0.928, 0.99, 0.3], radius=0.06)
    assert sorted(sorted(g) for g in groups) == [[0, 1], [2], [3]]
