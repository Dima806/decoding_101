"""Hold the from-scratch decoders to ``transformers``' own implementation.

Two levels of check:

* warpers - our ``logits -> logits`` functions against ``TemperatureLogitsWarper``,
  ``TopKLogitsWarper``, ``TopPLogitsWarper`` and ``RepetitionPenaltyLogitsProcessor``. Exact.
* generation - our greedy and beam loops against ``model.generate(do_sample=False)``,
  token for token.

Sampled generation is deliberately *not* compared token by token: ``generate`` draws from the
global torch RNG while this project threads an explicit ``torch.Generator``, so the random
streams differ by construction. The warped distributions they sample from are compared instead.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import pandas as pd
import torch
from torch import Tensor

from src.decoders import BeamSearch, Decoder, DecodingStrategy
from src.model.logits import HFLanguageModel

DEFAULT_WARPER_CASES: tuple[dict[str, Any], ...] = (
    {"temperature": 0.7},
    {"temperature": 1.8},
    {"top_k": 5},
    {"top_k": 50},
    {"top_p": 0.5},
    {"top_p": 0.9},
    {"repetition_penalty": 1.3},
    {"temperature": 0.8, "top_k": 40, "top_p": 0.95, "repetition_penalty": 1.2},
)


def reference_warp(
    logits: Tensor,
    input_ids: Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> Tensor:
    """Warp ``logits`` with ``transformers``' processors, in ``generate``'s order.

    ``temperature`` must be > 0 here: ``transformers`` rejects 0, whereas our
    :func:`src.decoders.apply_temperature` treats it as the greedy limit.
    """
    from transformers import (
        LogitsProcessorList,
        RepetitionPenaltyLogitsProcessor,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )

    processors = LogitsProcessorList()
    if repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty)))
    if temperature != 1.0:
        processors.append(TemperatureLogitsWarper(float(temperature)))
    if top_k > 0:
        processors.append(TopKLogitsWarper(top_k=top_k))
    if top_p < 1.0:
        processors.append(TopPLogitsWarper(top_p=float(top_p)))
    # transformers annotates with the legacy LongTensor / FloatTensor classes.
    return processors(
        cast("torch.LongTensor", input_ids), cast("torch.FloatTensor", logits.clone())
    )


def compare_warpers(
    logits: Tensor,
    input_ids: Tensor,
    cases: Sequence[Mapping[str, Any]] = DEFAULT_WARPER_CASES,
) -> pd.DataFrame:
    """Run every case through our pipeline and ``transformers``'; report agreement.

    Returns:
        One row per case: settings, tokens kept by each side, max absolute difference over kept
        logits, and ``identical`` (bit-for-bit equal, ``-inf`` included).
    """
    records = []
    for case in cases:
        ours = Decoder(**case).warp(logits, input_ids)
        ref = reference_warp(logits, input_ids, **case)
        same_mask = torch.equal(torch.isfinite(ours), torch.isfinite(ref))
        finite = torch.isfinite(ours) & torch.isfinite(ref)
        diff = (ours[finite] - ref[finite]).abs().max().item() if finite.any() else 0.0
        records.append(
            {
                "case": ", ".join(f"{k}={v}" for k, v in case.items()),
                "kept_ours": int(torch.isfinite(ours).sum()),
                "kept_transformers": int(torch.isfinite(ref).sum()),
                "same_tokens_kept": same_mask,
                "max_abs_diff": diff,
                "identical": torch.equal(ours, ref),
            }
        )
    return pd.DataFrame.from_records(records)


def reference_generate(
    model: HFLanguageModel,
    input_ids: Tensor,
    *,
    max_new_tokens: int,
    num_beams: int = 1,
    repetition_penalty: float = 1.0,
    length_penalty: float = 1.0,
) -> Tensor:
    """Deterministic ``model.generate`` with the cache off, so its forward passes match ours."""
    eos = model.eos_token_id
    kwargs: dict[str, Any] = {
        "attention_mask": torch.ones_like(input_ids),
        "do_sample": False,
        "num_beams": num_beams,
        "max_new_tokens": max_new_tokens,
        "use_cache": False,
        "pad_token_id": eos if eos is not None else 0,
        "eos_token_id": eos,
    }
    if repetition_penalty != 1.0:
        kwargs["repetition_penalty"] = repetition_penalty
    if num_beams > 1:
        kwargs["length_penalty"] = length_penalty
        kwargs["early_stopping"] = False
    with torch.no_grad():
        out = model.model.generate(input_ids, **kwargs)
    return torch.as_tensor(out)


def compare_generation(
    model: HFLanguageModel,
    input_ids: Tensor,
    *,
    max_new_tokens: int = 20,
    beam_widths: Sequence[int] = (2, 4),
    repetition_penalty: float = 1.3,
    length_penalty: float = 1.0,
) -> pd.DataFrame:
    """Greedy, penalised greedy, and beam search: ours vs ``generate``, token for token."""
    cases: list[tuple[DecodingStrategy, dict[str, Any]]] = [
        (Decoder.greedy(), {}),
        (Decoder.greedy(repetition_penalty), {"repetition_penalty": repetition_penalty}),
    ]
    cases += [
        (BeamSearch(b, length_penalty), {"num_beams": b, "length_penalty": length_penalty})
        for b in beam_widths
    ]
    records = []
    for strategy, ref_kwargs in cases:
        ours = strategy.generate(model, input_ids, max_new_tokens, eos_token_id=model.eos_token_id)
        ref = reference_generate(model, input_ids, max_new_tokens=max_new_tokens, **ref_kwargs)
        records.append(
            {
                "decoder": strategy.label,
                "tokens_ours": ours.size(1) - input_ids.size(1),
                "tokens_transformers": ref.size(1) - input_ids.size(1),
                "identical": torch.equal(ours, ref),
            }
        )
    return pd.DataFrame.from_records(records)
