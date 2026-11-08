"""Top-p (nucleus): keep the smallest set of tokens whose probability mass reaches p.

Sort probabilities descending, walk down the list, stop as soon as the running total reaches p.
The cut-off adapts: a confident distribution keeps one or two tokens, an unsure one keeps many.
(Holtzman et al., 2020, "The Curious Case of Neural Text Degeneration".)
"""

from __future__ import annotations

import torch
from torch import Tensor

NEG_INF = float("-inf")


def apply_top_p(
    logits: Tensor,
    p: float,
    min_tokens_to_keep: int = 1,
    filter_value: float = NEG_INF,
) -> Tensor:
    """Mask everything outside the nucleus.

    A token is kept iff the mass of the tokens *strictly more probable* than it is below ``p``.
    Equivalently: keep the shortest prefix of the sorted list whose cumulative probability is
    ``>= p``, *including* the token that crosses ``p``. Same rule as
    ``transformers.TopPLogitsWarper``.

    Args:
        logits: ``(batch, vocab)`` scores. Not modified.
        p: nucleus mass in ``(0, 1]``. ``p == 1`` keeps everything.
        min_tokens_to_keep: always keep at least this many tokens.
        filter_value: value written into removed positions.

    Returns:
        ``(batch, vocab)`` logits with removed tokens set to ``filter_value``.

    Raises:
        ValueError: if ``p`` is outside ``(0, 1]``.
    """
    if not 0.0 < p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {p}")
    if p == 1.0:
        return logits.clone()
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True, stable=True)
    sorted_probs = sorted_logits.softmax(dim=-1)
    mass_before = sorted_probs.cumsum(dim=-1) - sorted_probs
    remove_sorted = mass_before >= p
    remove_sorted[..., :min_tokens_to_keep] = False
    remove = remove_sorted.scatter(-1, sorted_idx, remove_sorted)
    return logits.masked_fill(remove, filter_value)
