"""Describe a next-token distribution: how peaked it is, where its mass sits, what survives.

Everything takes logits (``-inf`` = removed) so it applies equally to the model's raw output and
to any warped version of it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd
import torch
from torch import Tensor

from src.decoders.temperature import apply_temperature
from src.decoders.top_k import apply_top_k
from src.decoders.top_p import apply_top_p

if TYPE_CHECKING:
    from src.model.logits import LanguageModel


def probs_from_logits(logits: Tensor) -> Tensor:
    """Softmax along the vocabulary. ``-inf`` logits map to exactly 0."""
    return torch.softmax(logits, dim=-1)


def entropy(logits: Tensor, base: float = math.e) -> Tensor:
    """Shannon entropy ``H = -sum p log p`` per row, in nats by default.

    Zero-probability (masked) tokens contribute 0 rather than ``nan``.

    Args:
        logits: ``(..., vocab)``.
        base: logarithm base; ``2`` gives bits.

    Returns:
        ``(...)`` entropies.
    """
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    terms = torch.where(p > 0, p * logp, torch.zeros_like(p))
    return -terms.sum(dim=-1) / math.log(base) + 0.0  # + 0.0 turns -0.0 into 0.0


def effective_choices(logits: Tensor) -> Tensor:
    """``exp(H)``: the number of equally likely tokens with the same entropy."""
    return entropy(logits).exp()


def top_mass(logits: Tensor, k: int) -> Tensor:
    """Probability mass on the ``k`` most likely tokens, per row."""
    probs = probs_from_logits(logits)
    return probs.topk(min(k, probs.size(-1)), dim=-1).values.sum(dim=-1)


def support_size(logits: Tensor) -> Tensor:
    """Number of tokens still allowed (finite logit), per row."""
    return torch.isfinite(logits).sum(dim=-1)


def renormalize(probs: Tensor, keep: Tensor) -> Tensor:
    """The teaching view of a filter: zero the removed tokens, divide by what is left.

    Mathematically identical to masking logits with ``-inf`` and taking one softmax, which is
    what the decoders do.
    """
    kept = probs * keep.to(probs.dtype)
    return kept / kept.sum(dim=-1, keepdim=True)


def temperature_sweep(logits: Tensor, temperatures: Sequence[float]) -> pd.DataFrame:
    """How one fixed distribution reshapes as ``T`` changes.

    Args:
        logits: ``(1, vocab)`` or ``(vocab,)`` raw logits.
        temperatures: values of ``T >= 0``.

    Returns:
        One row per temperature: entropy (bits), effective choices, top-1 probability.
    """
    row = logits.reshape(1, -1)
    records = []
    for t in temperatures:
        warped = apply_temperature(row, t)
        records.append(
            {
                "temperature": t,
                "entropy_bits": entropy(warped, base=2).item(),
                "effective_choices": effective_choices(warped).item(),
                "top1_prob": probs_from_logits(warped).max().item(),
            }
        )
    return pd.DataFrame.from_records(records)


def top_tokens(model: LanguageModel, logits: Tensor, n: int = 20) -> pd.DataFrame:
    """The ``n`` most likely next tokens as a table: rank, id, label, logit, probability.

    Args:
        model: supplies display labels.
        logits: ``(1, vocab)`` or ``(vocab,)``.
        n: rows to return (capped at the vocabulary size).
    """
    row = logits.reshape(-1)
    probs = probs_from_logits(row)
    values, ids = probs.topk(min(n, row.numel()))
    id_list = [int(i) for i in ids.tolist()]
    return pd.DataFrame(
        {
            "rank": range(1, len(id_list) + 1),
            "token_id": id_list,
            "token": model.token_labels(id_list),
            "logit": row[ids].tolist(),
            "prob": values.tolist(),
        }
    )


def kept_set_sizes(
    logits_by_name: dict[str, Tensor], ks: Sequence[int], ps: Sequence[float]
) -> pd.DataFrame:
    """How many tokens top-k and top-p keep on each distribution, plus the mass they keep.

    The point: a fixed ``k`` keeps the same number of tokens whether the model is sure or not,
    while nucleus keeps few when the distribution is peaked and many when it is flat.

    Returns:
        Long table with columns ``distribution, method, setting, kept, mass_kept``.
    """
    records = []
    for name, logits in logits_by_name.items():
        row = logits.reshape(1, -1)
        probs = probs_from_logits(row)
        settings: list[tuple[str, float, Tensor]] = [
            ("top-k", k, apply_top_k(row, k)) for k in ks
        ] + [("top-p", p, apply_top_p(row, p)) for p in ps]
        for method, setting, warped in settings:
            keep = torch.isfinite(warped)
            records.append(
                {
                    "distribution": name,
                    "method": method,
                    "setting": setting,
                    "kept": int(keep.sum()),
                    "mass_kept": float(probs[keep].sum()),
                }
            )
    return pd.DataFrame.from_records(records)


def token_probabilities(model: LanguageModel, ids: Tensor, prompt_len: int) -> pd.DataFrame:
    """Step-by-step view of a generated sequence under the raw model.

    Args:
        model: the fixed model.
        ids: ``(1, seq)`` prompt plus generated tokens.
        prompt_len: number of leading prompt tokens (not reported).

    Returns:
        One row per generated token: position, label, probability of the chosen token, top-1
        probability, and the entropy (bits) of the distribution it was drawn from.
    """
    records = []
    for pos in range(max(prompt_len, 1), ids.size(1)):
        logits = model.next_token_logits(ids[:, :pos])
        probs = probs_from_logits(logits)[0]
        token = int(ids[0, pos])
        records.append(
            {
                "position": pos - prompt_len,
                "token": model.token_labels([token])[0],
                "p_chosen": probs[token].item(),
                "p_top1": probs.max().item(),
                "entropy_bits": entropy(logits, base=2).item(),
            }
        )
    return pd.DataFrame.from_records(records)
