"""Penalties: discourage tokens that already appear in the context. The practical loop patch.

Two families, both applied to logits before temperature:

* Repetition penalty (Keskar et al., 2019, CTRL; what ``transformers`` implements). Multiplicative
  and sign-dependent, applied once per distinct token already in the context::

      z_t <- z_t * rho   if z_t < 0
      z_t <- z_t / rho   otherwise

  Dividing a *negative* logit would raise it, i.e. reward repetition, hence the branch.

* Presence / frequency penalties (OpenAI-style, additive)::

      z_t <- z_t - presence * 1[c_t > 0] - frequency * c_t

  where ``c_t`` counts token ``t`` in the context.
"""

from __future__ import annotations

import torch
from torch import Tensor


def apply_repetition_penalty(logits: Tensor, input_ids: Tensor, penalty: float) -> Tensor:
    """CTRL repetition penalty.

    Args:
        logits: ``(batch, vocab)`` scores. Not modified.
        input_ids: ``(batch, seq)`` context so far (prompt included).
        penalty: ``rho > 0``. ``1.0`` is off, ``> 1`` discourages repeats, ``< 1`` rewards them.

    Returns:
        ``(batch, vocab)`` penalised logits.
    """
    if penalty <= 0:
        raise ValueError(f"repetition penalty must be > 0, got {penalty}")
    if penalty == 1.0 or input_ids.numel() == 0:
        return logits.clone()
    seen = torch.gather(logits, -1, input_ids)
    seen = torch.where(seen < 0, seen * penalty, seen / penalty)
    return logits.scatter(-1, input_ids, seen)


def token_counts(input_ids: Tensor, vocab_size: int) -> Tensor:
    """``(batch, vocab)`` float tensor: how many times each token occurs in each row."""
    counts = torch.zeros(input_ids.size(0), vocab_size, dtype=torch.float32)
    return counts.scatter_add_(-1, input_ids, torch.ones_like(input_ids, dtype=torch.float32))


def apply_presence_frequency_penalty(
    logits: Tensor,
    input_ids: Tensor,
    presence: float = 0.0,
    frequency: float = 0.0,
) -> Tensor:
    """OpenAI-style additive penalties.

    Args:
        logits: ``(batch, vocab)`` scores. Not modified.
        input_ids: ``(batch, seq)`` context so far.
        presence: flat penalty for any token seen at least once.
        frequency: penalty per occurrence.

    Returns:
        ``(batch, vocab)`` penalised logits.
    """
    if presence == 0.0 and frequency == 0.0:
        return logits.clone()
    counts = token_counts(input_ids, logits.size(-1))
    return logits - presence * (counts > 0).to(logits.dtype) - frequency * counts
