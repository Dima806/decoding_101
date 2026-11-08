"""Temperature is one line: T -> 0 is greedy, T -> inf is uniform, entropy rises with T."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from src.analysis import entropy, probs_from_logits, temperature_sweep
from src.config import get_settings
from src.decoders import Decoder, apply_temperature, greedy_pick, make_generator
from src.model import next_token_logits


def test_temperature_is_division(random_logits: Tensor) -> None:
    assert torch.equal(apply_temperature(random_logits, 0.5), random_logits / 0.5)
    assert torch.equal(apply_temperature(random_logits, 1.0), random_logits)


def test_tiny_temperature_is_greedy(random_logits: Tensor) -> None:
    probs = probs_from_logits(apply_temperature(random_logits, 1e-6))
    one_hot = torch.nn.functional.one_hot(greedy_pick(random_logits).squeeze(-1), 200).float()
    torch.testing.assert_close(probs, one_hot)


def test_huge_temperature_is_uniform(random_logits: Tensor) -> None:
    probs = probs_from_logits(apply_temperature(random_logits, 1e6))
    torch.testing.assert_close(probs, torch.full_like(probs, 1 / 200), atol=1e-6, rtol=0)


def test_zero_temperature_keeps_only_the_argmax(random_logits: Tensor) -> None:
    warped = apply_temperature(random_logits, 0.0)
    assert torch.isfinite(warped).sum(dim=-1).tolist() == [1, 1, 1, 1]
    assert torch.equal(greedy_pick(warped), greedy_pick(random_logits))


@pytest.mark.parametrize("bad", [-0.1, math.inf, math.nan])
def test_invalid_temperature_raises(bad: float, random_logits: Tensor) -> None:
    with pytest.raises(ValueError):
        apply_temperature(random_logits, bad)


def test_entropy_rises_with_temperature_on_random_logits(random_logits: Tensor) -> None:
    temps = get_settings().decoding.temperatures
    for row in random_logits:
        h = temperature_sweep(row, temps)["entropy_bits"].tolist()
        assert all(b >= a - 1e-6 for a, b in zip(h, h[1:], strict=False))
        assert h[0] == pytest.approx(0.0, abs=1e-6)
        assert h[-1] < math.log2(200)


def test_entropy_rises_with_temperature_on_the_char_model(char_lm) -> None:
    logits = next_token_logits(char_lm, "the keeper ")
    temps = [0.1, 0.5, 1.0, 2.0, 5.0, 50.0]
    h = [entropy(apply_temperature(logits, t), base=2).item() for t in temps]
    assert h == sorted(h)
    assert h[-1] == pytest.approx(math.log2(char_lm.vocab_size), abs=0.05)


def test_near_zero_temperature_generates_the_greedy_text(char_lm) -> None:
    ids = char_lm.encode("the fox")
    greedy = Decoder.greedy().generate(char_lm, ids, 120)
    cold = Decoder.sampling(1e-4).generate(char_lm, ids, 120, generator=make_generator(3))
    assert torch.equal(cold, greedy)
