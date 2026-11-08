"""Top-k / top-p keep the right tokens and renormalise; penalties and sampling behave."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from src.analysis import probs_from_logits, renormalize
from src.decoders import (
    BeamSearch,
    Decoder,
    apply_presence_frequency_penalty,
    apply_repetition_penalty,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    beam_search,
    generate,
    greedy_pick,
    make_generator,
    sample_from_logits,
    token_counts,
)


def kept(logits: Tensor) -> set[int]:
    return set(torch.isfinite(logits[0]).nonzero().flatten().tolist())


def logits_from_probs(probs: list[float]) -> Tensor:
    return torch.log(torch.tensor([probs]))


# --- greedy ---------------------------------------------------------------------------------


def test_greedy_pick_is_row_argmax(random_logits: Tensor) -> None:
    picks = greedy_pick(random_logits)
    assert picks.shape == (4, 1)
    assert torch.equal(picks.squeeze(-1), random_logits.argmax(dim=-1))


# --- top-k ----------------------------------------------------------------------------------


def test_top_k_keeps_exactly_the_k_largest() -> None:
    logits = torch.tensor([[0.1, 3.0, -1.0, 2.0, 0.5]])
    assert kept(apply_top_k(logits, 2)) == {1, 3}
    assert kept(apply_top_k(logits, 1)) == {1}


def test_top_k_keeps_every_tie_at_the_threshold() -> None:
    logits = torch.tensor([[1.0, 2.0, 2.0, 2.0, 0.0]])
    assert kept(apply_top_k(logits, 2)) == {1, 2, 3}


def test_top_k_disabled_or_larger_than_vocab_keeps_everything(random_logits: Tensor) -> None:
    assert torch.equal(apply_top_k(random_logits, 0), random_logits)
    assert torch.equal(apply_top_k(random_logits, 10_000), random_logits)


# --- top-p ----------------------------------------------------------------------------------


PROBS = [0.5, 0.3, 0.15, 0.05]


@pytest.mark.parametrize(
    ("p", "expected"),
    [
        (0.45, {0}),  # the top token alone already covers p
        (0.75, {0, 1}),  # 0.5 < p, 0.5 + 0.3 >= p: the crossing token is included
        (0.85, {0, 1, 2}),
        (0.99, {0, 1, 2, 3}),
    ],
)
def test_top_p_keeps_the_smallest_prefix_reaching_p(p: float, expected: set[int]) -> None:
    assert kept(apply_top_p(logits_from_probs(PROBS), p)) == expected


def test_top_p_respects_min_tokens_to_keep() -> None:
    logits = logits_from_probs(PROBS)
    assert kept(apply_top_p(logits, 0.01)) == {0}
    assert kept(apply_top_p(logits, 0.01, min_tokens_to_keep=3)) == {0, 1, 2}


def test_top_p_one_keeps_everything_and_bad_p_raises(random_logits: Tensor) -> None:
    assert torch.equal(apply_top_p(random_logits, 1.0), random_logits)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            apply_top_p(random_logits, bad)


def test_top_p_adapts_to_peakedness() -> None:
    peaked = logits_from_probs([0.95, 0.02, 0.01, 0.01, 0.01])
    flat = logits_from_probs([0.2] * 5)
    assert len(kept(apply_top_p(peaked, 0.9))) == 1
    assert len(kept(apply_top_p(flat, 0.9))) == 5


# --- shared warper properties ---------------------------------------------------------------

WARPERS = {
    "temperature": lambda z: apply_temperature(z, 0.7),
    "temperature0": lambda z: apply_temperature(z, 0.0),
    "top_k": lambda z: apply_top_k(z, 7),
    "top_p": lambda z: apply_top_p(z, 0.8),
    "pipeline": lambda z: Decoder(temperature=1.3, top_k=50, top_p=0.9).warp(
        z, torch.zeros(z.size(0), 1, dtype=torch.int64)
    ),
}


@pytest.mark.parametrize("name", WARPERS)
def test_warped_distribution_renormalises_to_one(name: str, random_logits: Tensor) -> None:
    probs = probs_from_logits(WARPERS[name](random_logits))
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(4))


@pytest.mark.parametrize("name", ["top_k", "top_p"])
def test_masking_equals_zero_the_tail_and_renormalise(name: str, random_logits: Tensor) -> None:
    warped = WARPERS[name](random_logits)
    teaching_view = renormalize(probs_from_logits(random_logits), torch.isfinite(warped))
    torch.testing.assert_close(probs_from_logits(warped), teaching_view, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("name", WARPERS)
def test_warpers_do_not_mutate_their_input(name: str, random_logits: Tensor) -> None:
    before = random_logits.clone()
    WARPERS[name](random_logits)
    assert torch.equal(random_logits, before)


# --- penalties ------------------------------------------------------------------------------


def test_repetition_penalty_lowers_seen_tokens_whatever_their_sign() -> None:
    logits = torch.tensor([[2.0, -2.0, 1.0, -1.0]])
    seen = torch.tensor([[0, 1]])
    out = apply_repetition_penalty(logits, seen, 2.0)
    assert out[0, 0] == 1.0  # positive: divided
    assert out[0, 1] == -4.0  # negative: multiplied (dividing would *raise* it)
    assert torch.equal(out[0, 2:], logits[0, 2:])  # unseen tokens untouched


def test_repetition_penalty_applies_once_per_token() -> None:
    logits = torch.tensor([[2.0, 1.0]])
    once = apply_repetition_penalty(logits, torch.tensor([[0]]), 2.0)
    thrice = apply_repetition_penalty(logits, torch.tensor([[0, 0, 0]]), 2.0)
    assert torch.equal(once, thrice)


def test_repetition_penalty_validation(random_logits: Tensor) -> None:
    ids = torch.zeros(4, 3, dtype=torch.int64)
    assert torch.equal(apply_repetition_penalty(random_logits, ids, 1.0), random_logits)
    with pytest.raises(ValueError):
        apply_repetition_penalty(random_logits, ids, 0.0)


def test_presence_and_frequency_penalties_are_additive() -> None:
    logits = torch.zeros(1, 4)
    ids = torch.tensor([[1, 2, 2, 2]])
    assert torch.equal(token_counts(ids, 4), torch.tensor([[0.0, 1.0, 3.0, 0.0]]))
    out = apply_presence_frequency_penalty(logits, ids, presence=0.5, frequency=0.25)
    torch.testing.assert_close(out, torch.tensor([[0.0, -0.75, -1.25, 0.0]]))
    assert torch.equal(apply_presence_frequency_penalty(logits, ids), logits)


# --- the Decoder pipeline -------------------------------------------------------------------


def test_decoder_validates_settings() -> None:
    for kwargs in ({"temperature": -1.0}, {"top_p": 0.0}, {"repetition_penalty": 0.0}):
        with pytest.raises(ValueError):
            Decoder(**kwargs)


def test_decoder_labels_name_the_settings() -> None:
    assert Decoder.greedy().label == "greedy"
    assert Decoder.greedy(1.3).label == "greedy, rep=1.3"
    assert Decoder.nucleus(0.9, 0.7).label == "T=0.7, top-p=0.9"
    assert Decoder.top_k_sampling(5).label == "T=1, top-k=5"
    assert Decoder(presence_penalty=0.5, frequency_penalty=0.1).label == (
        "T=1, presence=0.5, freq=0.1"
    )
    assert Decoder(name="custom").label == "custom"
    assert BeamSearch(4).label == "beam (B=4)"


def test_pipeline_runs_penalty_then_temperature_then_top_k_then_top_p() -> None:
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0]])
    ids = torch.tensor([[0]])
    decoder = Decoder(temperature=0.5, top_k=3, top_p=0.9, repetition_penalty=2.0)
    expected = apply_repetition_penalty(logits, ids, 2.0)
    expected = apply_top_p(apply_top_k(apply_temperature(expected, 0.5), 3), 0.9)
    assert torch.equal(decoder.warp(logits, ids), expected)


def test_sampling_needs_a_generator_and_is_reproducible(random_logits: Tensor) -> None:
    ids = torch.zeros(4, 1, dtype=torch.int64)
    decoder = Decoder.sampling(1.0)
    with pytest.raises(ValueError):
        decoder(random_logits, ids)
    first = decoder(random_logits, ids, make_generator(7))
    second = decoder(random_logits, ids, make_generator(7))
    assert torch.equal(first, second)


def test_sampling_never_draws_a_removed_token(random_logits: Tensor) -> None:
    warped = apply_top_k(random_logits[:1], 3)
    allowed = kept(warped)
    g = make_generator(0)
    draws = {int(sample_from_logits(warped, g)) for _ in range(500)}
    assert draws <= allowed


def test_sample_frequencies_match_the_warped_distribution() -> None:
    logits = logits_from_probs([0.5, 0.25, 0.15, 0.07, 0.03])
    warped = apply_top_p(logits, 0.85)  # keeps the first three, renormalised
    target = probs_from_logits(warped)[0]
    g = make_generator(1)
    n = 20_000
    draws = torch.multinomial(target, n, replacement=True, generator=g)
    freq = torch.bincount(draws, minlength=5).float() / n
    assert freq[3:].sum() == 0
    assert 0.5 * (freq - target).abs().sum() < 0.02  # total-variation distance


# --- the generation loop and beam search ----------------------------------------------------


class _Scripted:
    """Model whose argmax is always ``(last token + 1) % vocab``."""

    vocab_size = 5
    eos_token_id = 4
    name = "scripted"

    def next_token_logits(self, input_ids: Tensor) -> Tensor:
        nxt = (input_ids[:, -1] + 1) % self.vocab_size
        return torch.nn.functional.one_hot(nxt, self.vocab_size).float() * 10


def test_generate_appends_and_stops_at_eos() -> None:
    model = _Scripted()
    ids = torch.tensor([[0]])
    out = generate(model, ids, decoder=Decoder.greedy(), max_new_tokens=10)
    assert out.tolist() == [[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0]]
    stopped = generate(model, ids, decoder=Decoder.greedy(), max_new_tokens=10, eos_token_id=4)
    assert stopped.tolist() == [[0, 1, 2, 3, 4]]


def test_generate_pads_finished_rows_with_eos() -> None:
    model = _Scripted()
    ids = torch.tensor([[2], [0]])
    out = generate(model, ids, decoder=Decoder.greedy(), max_new_tokens=4, eos_token_id=4)
    assert out.tolist() == [[2, 3, 4, 4, 4], [0, 1, 2, 3, 4]]


def test_beam_finds_the_more_probable_sequence_greedy_misses(table_lm) -> None:
    ids = torch.tensor([[0]])
    greedy = Decoder.greedy().generate(table_lm, ids, 2)
    beam = beam_search(table_lm, ids, num_beams=2, max_new_tokens=2)
    assert greedy.tolist() == [[0, 1, 0]]  # took 0.6, then a coin flip among six
    assert beam.sequence.tolist() == [[0, 2, 3]]  # took 0.4, then 0.97
    assert math.isclose(beam.score, math.log(0.4 * 0.97) / 2, rel_tol=1e-5)
    assert len(beam.hypotheses) == 2


def test_beam_width_one_is_greedy(char_lm) -> None:
    ids = char_lm.encode("the sea was")
    greedy = Decoder.greedy().generate(char_lm, ids, 60)
    assert torch.equal(BeamSearch(1).generate(char_lm, ids, 60), greedy)


def test_beam_search_validates_input(table_lm) -> None:
    with pytest.raises(ValueError):
        beam_search(table_lm, torch.tensor([[0], [0]]), num_beams=2, max_new_tokens=2)
    with pytest.raises(ValueError):
        beam_search(table_lm, torch.tensor([[0]]), num_beams=0, max_new_tokens=2)
