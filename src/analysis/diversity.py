"""Quality versus diversity: the trade-off every decoder sits somewhere on.

* Quality is perplexity-style: the mean log-probability the *unwarped* model assigns to the
  generated tokens. High means "the model finds this likely". Careful: loops are very likely,
  so likelihood alone rewards degenerate text (Holtzman et al., 2020). Read it next to diversity.
* Diversity is distinct-n pooled across several samples for the same prompt: do different runs
  say different things?
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import pandas as pd
import torch
from torch import Tensor

from src.analysis.repetition import distinct_n, words

if TYPE_CHECKING:
    from src.model.logits import LanguageModel


def sequence_logprob(
    model: LanguageModel,
    ids: Tensor,
    prompt_len: int,
    reduction: Literal["mean", "sum"] = "mean",
) -> float:
    """Log-probability (nats) of ``ids[:, prompt_len:]`` under the raw model.

    Args:
        model: the model whose fixed distribution is used to score (no temperature, no filter).
        ids: ``(1, seq)`` prompt plus generated tokens.
        prompt_len: number of leading prompt tokens, which are not scored.
        reduction: ``"mean"`` per token (perplexity-style) or ``"sum"`` (what beam maximises).

    Returns:
        The log-probability; ``nan`` if nothing was generated.
    """
    total, count = 0.0, 0
    for pos in range(max(prompt_len, 1), ids.size(1)):
        logprobs = torch.log_softmax(model.next_token_logits(ids[:, :pos]), dim=-1)
        total += logprobs[0, ids[0, pos]].item()
        count += 1
    if not count:
        return math.nan
    return total / count if reduction == "mean" else total


def perplexity(mean_logprob: float) -> float:
    """``exp(-mean log p)``: the model's average branching factor on the text."""
    return math.exp(-mean_logprob)


def pooled_distinct_n(texts: Sequence[str], n: int = 2) -> float:
    """Distinct-n over the n-grams of all ``texts`` together: diversity *across* samples."""
    pooled = [gram for text in texts for gram in _ngrams_of(text, n)]
    return len(set(pooled)) / len(pooled) if pooled else 1.0


def _ngrams_of(text: str, n: int) -> list[tuple[str, ...]]:
    units = words(text)
    return [tuple(units[i : i + n]) for i in range(len(units) - n + 1)]


def within_distinct_n(texts: Sequence[str], n: int = 2) -> float:
    """Mean distinct-n *within* each text: is each sample free of loops?"""
    return sum(distinct_n(words(t), n) for t in texts) / len(texts) if texts else 1.0


def pareto_front(df: pd.DataFrame, x: str = "diversity", y: str = "quality") -> pd.DataFrame:
    """Rows not dominated on (``x``, ``y``), both maximised, sorted by ``x``."""
    ordered = df.sort_values([x, y], ascending=[False, False])
    best_y = -math.inf
    keep = []
    for idx, value in zip(ordered.index, ordered[y], strict=True):
        if value > best_y:
            keep.append(idx)
            best_y = value
    return df.loc[keep].sort_values(x)
