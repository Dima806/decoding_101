"""Greedy decoding: always take the single most probable token.

The simplest decoder and the cautionary one. Once the model is slightly more confident about
continuing a phrase it has already produced, argmax picks it again, and again: a loop.
"""

from __future__ import annotations

from torch import Tensor


def greedy_pick(logits: Tensor) -> Tensor:
    """Return the argmax token for each row.

    Args:
        logits: ``(batch, vocab)`` scores (raw or already warped).

    Returns:
        ``(batch, 1)`` int64 token ids. Ties go to the lowest token id, like ``torch.argmax``.
    """
    return logits.argmax(dim=-1, keepdim=True)
