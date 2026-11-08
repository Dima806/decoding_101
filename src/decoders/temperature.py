"""Temperature: divide the logits by T before the softmax. That is the whole knob.

    p_i(T) = exp(z_i / T) / sum_j exp(z_j / T)

T < 1 sharpens the distribution toward the top token (T -> 0 is greedy, and loops).
T > 1 flattens it toward uniform (T -> inf samples the whole vocabulary, and rambles).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

NEG_INF = float("-inf")


def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    """Scale logits by ``1 / temperature``.

    Args:
        logits: ``(batch, vocab)`` scores. Not modified.
        temperature: ``T >= 0``. ``T == 0`` is the greedy limit and is special-cased *before*
            any division: every token except the argmax is masked to ``-inf``.

    Returns:
        ``(batch, vocab)`` warped logits.

    Raises:
        ValueError: if ``temperature`` is negative or not finite.
    """
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError(f"temperature must be a finite number >= 0, got {temperature}")
    if temperature == 0:
        keep = torch.zeros_like(logits, dtype=torch.bool)
        keep.scatter_(-1, logits.argmax(dim=-1, keepdim=True), True)
        return logits.masked_fill(~keep, NEG_INF)
    return logits / temperature
