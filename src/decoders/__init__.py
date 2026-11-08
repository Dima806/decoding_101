"""Decoders: the few lines that turn a next-token distribution into text.

The model's job ends at the logits. Everything here is the sampler, and it runs in a fixed
order, the same one ``transformers.generate`` uses::

    penalties -> temperature -> top-k -> top-p -> softmax -> pick (argmax or sample)

Warpers are pure ``logits -> logits`` functions that mask removed tokens with ``-inf``.
:class:`Decoder` chains them; :func:`generate` is the one generation loop in the project.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch
from torch import Tensor

from src.decoders.beam import BeamResult, BeamSearch, beam_search
from src.decoders.greedy import greedy_pick
from src.decoders.penalties import (
    apply_presence_frequency_penalty,
    apply_repetition_penalty,
    token_counts,
)
from src.decoders.temperature import apply_temperature
from src.decoders.top_k import apply_top_k
from src.decoders.top_p import apply_top_p

if TYPE_CHECKING:
    from src.model.logits import LanguageModel

__all__ = [
    "BeamResult",
    "BeamSearch",
    "Decoder",
    "DecodingStrategy",
    "apply_presence_frequency_penalty",
    "apply_repetition_penalty",
    "apply_temperature",
    "apply_top_k",
    "apply_top_p",
    "beam_search",
    "generate",
    "greedy_pick",
    "make_generator",
    "sample_from_logits",
    "token_counts",
]


def make_generator(seed: int) -> torch.Generator:
    """A CPU ``torch.Generator`` seeded with ``seed``. Thread it through every stochastic call."""
    return torch.Generator().manual_seed(seed)


def sample_from_logits(logits: Tensor, generator: torch.Generator) -> Tensor:
    """Softmax, then one multinomial draw per row.

    Args:
        logits: ``(batch, vocab)`` warped logits; ``-inf`` entries get probability exactly 0.
        generator: seeded RNG. One draw per row per call, so the stream is predictable.

    Returns:
        ``(batch, 1)`` int64 token ids.
    """
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


@dataclass(frozen=True)
class Decoder:
    """One-token-at-a-time decoder: penalties -> temperature -> top-k -> top-p -> pick.

    ``temperature == 0`` picks the argmax (greedy); anything else samples.

    Attributes:
        temperature: logit divisor ``T >= 0``.
        top_k: keep the k most likely tokens; ``0`` disables.
        top_p: nucleus mass in ``(0, 1]``; ``1.0`` disables.
        repetition_penalty: CTRL penalty ``rho > 0``; ``1.0`` disables.
        presence_penalty: additive penalty for any seen token; ``0.0`` disables.
        frequency_penalty: additive penalty per occurrence; ``0.0`` disables.
        name: optional display label overriding :attr:`label`.
    """

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    name: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError(f"temperature must be a finite number >= 0, got {self.temperature}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.repetition_penalty <= 0:
            raise ValueError(f"repetition_penalty must be > 0, got {self.repetition_penalty}")

    # --- constructors for the named decoders ------------------------------------------------

    @classmethod
    def greedy(cls, repetition_penalty: float = 1.0) -> Decoder:
        """Always the argmax, optionally with a repetition penalty."""
        return cls(temperature=0.0, repetition_penalty=repetition_penalty)

    @classmethod
    def sampling(cls, temperature: float = 1.0) -> Decoder:
        """Sample from the whole (temperature-scaled) distribution."""
        return cls(temperature=temperature)

    @classmethod
    def top_k_sampling(cls, k: int, temperature: float = 1.0) -> Decoder:
        """Sample from the k most likely tokens."""
        return cls(temperature=temperature, top_k=k)

    @classmethod
    def nucleus(cls, p: float, temperature: float = 1.0) -> Decoder:
        """Sample from the smallest set of tokens covering mass ``p``."""
        return cls(temperature=temperature, top_p=p)

    # --- strategy interface -----------------------------------------------------------------

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0

    @property
    def is_deterministic(self) -> bool:
        return self.is_greedy

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        parts = ["greedy"] if self.is_greedy else [f"T={self.temperature:g}"]
        if self.top_k > 0:
            parts.append(f"top-k={self.top_k}")
        if self.top_p < 1.0:
            parts.append(f"top-p={self.top_p:g}")
        if self.repetition_penalty != 1.0:
            parts.append(f"rep={self.repetition_penalty:g}")
        if self.presence_penalty:
            parts.append(f"presence={self.presence_penalty:g}")
        if self.frequency_penalty:
            parts.append(f"freq={self.frequency_penalty:g}")
        return ", ".join(parts)

    def warp(self, logits: Tensor, input_ids: Tensor) -> Tensor:
        """Apply the full warper pipeline. ``(batch, vocab) -> (batch, vocab)`` logits."""
        logits = apply_repetition_penalty(logits, input_ids, self.repetition_penalty)
        logits = apply_presence_frequency_penalty(
            logits, input_ids, self.presence_penalty, self.frequency_penalty
        )
        logits = apply_temperature(logits, self.temperature)
        logits = apply_top_k(logits, self.top_k)
        return apply_top_p(logits, self.top_p)

    def __call__(
        self, logits: Tensor, input_ids: Tensor, generator: torch.Generator | None = None
    ) -> Tensor:
        """Pick the next token for each row: ``(batch, vocab) -> (batch, 1)``."""
        warped = self.warp(logits, input_ids)
        if self.is_greedy:
            return greedy_pick(warped)
        if generator is None:
            raise ValueError("sampling decoders need a seeded torch.Generator (make_generator)")
        return sample_from_logits(warped, generator)

    def generate(
        self,
        model: LanguageModel,
        input_ids: Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Run :func:`generate` with this decoder."""
        return generate(
            model,
            input_ids,
            decoder=self,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            generator=generator,
        )


class DecodingStrategy(Protocol):
    """Anything that can turn a prompt into a sequence: :class:`Decoder` or :class:`BeamSearch`."""

    @property
    def label(self) -> str: ...

    @property
    def is_deterministic(self) -> bool: ...

    def generate(
        self,
        model: LanguageModel,
        input_ids: Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor: ...


def generate(
    model: LanguageModel,
    input_ids: Tensor,
    *,
    decoder: Decoder,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """The generation loop: ask the model for logits, let the decoder pick, append, repeat.

    The model is called on the full sequence every step (no KV cache), which keeps the loop
    readable and makes its logits identical to a plain forward pass.

    Args:
        model: anything with ``next_token_logits((batch, seq)) -> (batch, vocab)``.
        input_ids: ``(batch, seq)`` prompt ids.
        decoder: picks one token per row from the logits.
        max_new_tokens: generation budget.
        eos_token_id: rows that emit it are finished and padded with it from then on.
        generator: seeded RNG, required for sampling decoders.

    Returns:
        ``(batch, seq + n_new)`` ids, prompt included.
    """
    ids = input_ids
    finished = torch.zeros(ids.size(0), dtype=torch.bool)
    for _ in range(max_new_tokens):
        next_ids = decoder(model.next_token_logits(ids), ids, generator)
        if eos_token_id is not None:
            next_ids = torch.where(finished[:, None], eos_token_id, next_ids)
            finished |= next_ids.squeeze(-1) == eos_token_id
        ids = torch.cat([ids, next_ids], dim=-1)
        if bool(finished.all()):
            break
    return ids
