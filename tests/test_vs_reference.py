"""The from-scratch decoders match transformers: warpers exactly, greedy and beam token for token."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from src.decoders import BeamSearch, Decoder
from src.evaluation import (
    DEFAULT_WARPER_CASES,
    compare_generation,
    compare_warpers,
    reference_generate,
    reference_warp,
)
from src.model import HFLanguageModel


@pytest.mark.parametrize("seed", range(5))
def test_warpers_identical_to_transformers_on_random_logits(seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(4, 1000, generator=g) * 3
    ids = torch.randint(0, 1000, (4, 30), generator=g)
    report = compare_warpers(logits, ids)
    assert len(report) == len(DEFAULT_WARPER_CASES)
    assert report["identical"].all(), report[~report["identical"]]


def test_warpers_identical_to_transformers_on_real_distilgpt2_logits(
    recorded_logits: list[tuple[Tensor, Tensor]],
) -> None:
    for ids, logits in recorded_logits:
        report = compare_warpers(logits, ids)
        assert report["identical"].all(), report[~report["identical"]]
        assert (report["kept_ours"] == report["kept_transformers"]).all()


def test_reference_warp_applies_nothing_by_default(random_logits: Tensor) -> None:
    ids = torch.zeros(4, 1, dtype=torch.int64)
    assert torch.equal(reference_warp(random_logits, ids), random_logits)


@pytest.mark.parametrize("seed", range(3))
def test_greedy_and_beam_identical_to_generate(seed: int, make_tiny_gpt2) -> None:
    model = make_tiny_gpt2(seed)
    ids = torch.randint(2, 96, (1, 6), generator=torch.Generator().manual_seed(seed))
    report = compare_generation(model, ids, max_new_tokens=20, beam_widths=(2, 3, 5))
    assert report["identical"].all(), report


@pytest.mark.parametrize("seed", range(4))
def test_beam_identical_to_generate_when_eos_ends_hypotheses(seed: int, make_tiny_gpt2) -> None:
    """Pick EOS among tokens the model likes, so hypotheses finish early and compete."""
    model = make_tiny_gpt2(seed, vocab_size=48)
    ids = torch.randint(2, 48, (1, 5), generator=torch.Generator().manual_seed(seed))
    free_run = Decoder.greedy().generate(model, ids, 8)[0, 5:].tolist()
    eos = free_run[3]
    cfg = model.model.generation_config
    cfg.eos_token_id = cfg.pad_token_id = eos
    for beams, length_penalty in [(2, 1.0), (4, 1.0), (3, 0.0), (3, 2.0)]:
        ours = BeamSearch(beams, length_penalty).generate(model, ids, 15, eos_token_id=eos)
        ref = reference_generate(
            model, ids, max_new_tokens=15, num_beams=beams, length_penalty=length_penalty
        )
        assert torch.equal(ours, ref), (beams, length_penalty, ours.tolist(), ref.tolist())


@pytest.mark.network
def test_pretrained_tiny_model_matches_generate() -> None:
    """End to end through from_pretrained and a real tokenizer (needs the HF Hub)."""
    try:
        model = HFLanguageModel.from_pretrained("sshleifer/tiny-gpt2")
    except OSError as exc:  # pragma: no cover - depends on the network
        pytest.skip(f"Hub unavailable: {exc}")
    ids = model.encode("Hello world, this is")
    assert model.decode(ids) == "Hello world, this is"
    report = compare_generation(model, ids, max_new_tokens=8, beam_widths=(2,))
    assert report["identical"].all(), report
