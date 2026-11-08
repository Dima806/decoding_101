"""Beam search: keep the B most probable *sequences*, not the most probable next token.

Each step expands every live beam by every token, scores each candidate by its accumulated
log-probability, and keeps the best B. Finished hypotheses are ranked by a length-normalised
score::

    score = sum(log p) / (generated_length ** length_penalty)

where ``generated_length`` counts new tokens only (prompt excluded, EOS included), as in
``transformers`` >= 4.36. Beam search optimises for probability, which is what you want for a
single correct answer and exactly what makes open-ended text safe, generic, and repetitive.

The bookkeeping mirrors the vectorised ``transformers`` implementation (2*B candidates per step,
only the top B may finish, the default ``early_stopping=False`` stop heuristic) so the
from-scratch result can be asserted token-for-token against ``model.generate(num_beams=B)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from src.model.logits import LanguageModel

DEAD = -1.0e9  # score assigned to candidates that already stopped, as in transformers


@dataclass(frozen=True, eq=False)
class BeamResult:
    """Outcome of :func:`beam_search`.

    Attributes:
        sequence: ``(1, seq)`` best hypothesis, prompt included.
        score: its length-normalised log-probability.
        hypotheses: every finished hypothesis kept, best first, as ``(1-D ids, score)``.
    """

    sequence: Tensor
    score: float
    hypotheses: list[tuple[Tensor, float]]


def beam_search(
    model: LanguageModel,
    input_ids: Tensor,
    *,
    num_beams: int,
    max_new_tokens: int,
    length_penalty: float = 1.0,
    eos_token_id: int | None = None,
) -> BeamResult:
    """Decode one prompt with beam search.

    Args:
        model: anything with ``next_token_logits((beams, seq)) -> (beams, vocab)``.
        input_ids: ``(1, seq)`` prompt ids.
        num_beams: beam width ``B``. ``B == 1`` is greedy.
        max_new_tokens: generation budget.
        length_penalty: exponent on the generated length in the final score.
        eos_token_id: token that ends a hypothesis early, or ``None``.

    Returns:
        A :class:`BeamResult`.
    """
    if input_ids.dim() != 2 or input_ids.size(0) != 1:
        raise ValueError("beam_search decodes one prompt at a time: input_ids must be (1, seq)")
    if num_beams < 1 or max_new_tokens < 1:
        raise ValueError("num_beams and max_new_tokens must both be >= 1")

    # transformers keeps max(2, 1 + n_eos) * B candidates; with one EOS id that is 2B.
    n_candidates = 2 * num_beams
    running = input_ids
    running_scores = torch.zeros(1)
    finished: list[tuple[Tensor, float]] = []

    for step in range(max_new_tokens):
        generated = step + 1
        logprobs = torch.log_softmax(model.next_token_logits(running), dim=-1)
        vocab = logprobs.size(-1)
        totals = (running_scores[:, None] + logprobs).flatten()
        cand_scores, cand_idx = torch.topk(totals, min(n_candidates, totals.numel()))
        beam_idx = torch.div(cand_idx, vocab, rounding_mode="floor")
        tokens = cand_idx % vocab
        cand_seqs = torch.cat([running[beam_idx], tokens[:, None]], dim=-1)

        stops = torch.full_like(tokens, step == max_new_tokens - 1, dtype=torch.bool)
        if eos_token_id is not None:
            stops |= tokens == eos_token_id

        # Only the top B candidates may finish; the other B exist so B beams stay alive.
        for rank in range(min(num_beams, tokens.numel())):
            if stops[rank]:
                normalised = cand_scores[rank].item() / generated**length_penalty
                finished.append((cand_seqs[rank], normalised))
        finished = sorted(finished, key=lambda hyp: hyp[1], reverse=True)[:num_beams]

        live_scores = cand_scores + stops.to(cand_scores.dtype) * DEAD
        keep = torch.topk(live_scores, min(num_beams, live_scores.numel())).indices
        running, running_scores = cand_seqs[keep], live_scores[keep]

        if bool(stops.all()):
            break
        # Stop once no live beam can beat the worst of B finished hypotheses.
        if len(finished) == num_beams:
            best_possible = running_scores[0].item() / generated**length_penalty
            if best_possible <= finished[-1][1]:
                break

    best_ids, best_score = finished[0]
    return BeamResult(sequence=best_ids[None, :], score=best_score, hypotheses=finished)


@dataclass(frozen=True)
class BeamSearch:
    """Beam search as a decoding strategy, interchangeable with :class:`src.decoders.Decoder`."""

    num_beams: int = 4
    length_penalty: float = 1.0
    name: str | None = None

    @property
    def label(self) -> str:
        return self.name or f"beam (B={self.num_beams})"

    @property
    def is_deterministic(self) -> bool:
        return True

    def generate(
        self,
        model: LanguageModel,
        input_ids: Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """``(1, seq)`` best sequence. ``generator`` is accepted for interface parity only."""
        return beam_search(
            model,
            input_ids,
            num_beams=self.num_beams,
            max_new_tokens=max_new_tokens,
            length_penalty=self.length_penalty,
            eos_token_id=eos_token_id,
        ).sequence
