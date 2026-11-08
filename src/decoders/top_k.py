"""Top-k: keep the k most probable tokens, mask the rest, sample from what is left.

Cuts the absurd long tail with a *fixed* budget of k tokens. That budget is too generous when
the model is confident (one sensible token, k-1 bad ones) and too stingy when it is unsure.
"""

from __future__ import annotations

import torch
from torch import Tensor

NEG_INF = float("-inf")


def apply_top_k(logits: Tensor, k: int, filter_value: float = NEG_INF) -> Tensor:
    """Mask every token whose logit is below the k-th largest.

    Ties at the threshold are all kept, so more than ``k`` tokens can survive. This matches
    ``transformers.TopKLogitsWarper``.

    Args:
        logits: ``(batch, vocab)`` scores. Not modified.
        k: number of tokens to keep. ``k <= 0`` disables the filter.
        filter_value: value written into removed positions.

    Returns:
        ``(batch, vocab)`` logits with removed tokens set to ``filter_value``.
    """
    if k <= 0:
        return logits.clone()
    k = min(k, logits.size(-1))
    threshold = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < threshold, filter_value)
