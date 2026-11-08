"""Greedy loops; nucleus does not; a repetition penalty helps. Measured, not eyeballed."""

from __future__ import annotations

import statistics

from src.analysis import (
    distinct_n,
    find_cycle,
    longest_repeated_span,
    ngrams,
    repetition_metrics,
    repetition_rate,
    words,
)
from src.config import get_settings
from src.datasets import char_prompts
from src.decoders import BeamSearch, Decoder, DecodingStrategy, make_generator

N_NEW = get_settings().generation.max_new_tokens


def test_words_lowercases_and_drops_punctuation() -> None:
    assert words("The sea, the SEA. It's calm!") == ["the", "sea", "the", "sea", "it's", "calm"]


def test_ngrams_and_distinct_n() -> None:
    seq = ["a", "b", "a", "b", "a"]
    assert ngrams(seq, 2) == [("a", "b"), ("b", "a"), ("a", "b"), ("b", "a")]
    assert distinct_n(seq, 1) == 2 / 5
    assert distinct_n(seq, 2) == 2 / 4
    assert repetition_rate(seq, 2) == 0.5
    assert distinct_n(["x"], 3) == 1.0  # too short to repeat


def test_longest_repeated_span() -> None:
    assert longest_repeated_span("abcdef") == 0
    assert longest_repeated_span("abcabc") == 3
    assert longest_repeated_span("aaaa") == 3  # overlaps count
    assert longest_repeated_span(["the", "sea", "and", "the", "sea", "and"]) == 3
    assert longest_repeated_span([]) == 0


def test_repetition_metrics_keys() -> None:
    m = repetition_metrics("the sea and the sea and the sea")
    assert set(m) == {"n_words", "distinct_1", "distinct_2", "distinct_3", "longest_repeat"}
    assert m["longest_repeat"] == 5


def mean_open_metrics(char_lm, strategy: DecodingStrategy, n_samples: int = 3) -> dict[str, float]:
    distinct, longest = [], []
    for prompt in char_prompts("open"):
        ids = char_lm.encode(prompt.text)
        for seed in range(1 if strategy.is_deterministic else n_samples):
            out = strategy.generate(char_lm, ids, N_NEW, generator=make_generator(seed))
            m = repetition_metrics(char_lm.decode(out[0, ids.size(1) :]))
            distinct.append(m["distinct_2"])
            longest.append(m["longest_repeat"])
    return {"distinct_2": statistics.mean(distinct), "longest": statistics.mean(longest)}


def test_greedy_repeats_more_than_nucleus_on_open_prompts(char_lm) -> None:
    greedy = mean_open_metrics(char_lm, Decoder.greedy())
    nucleus = mean_open_metrics(char_lm, Decoder.nucleus(get_settings().decoding.default_top_p))
    assert greedy["distinct_2"] < nucleus["distinct_2"] - 0.2
    assert greedy["longest"] > 2 * nucleus["longest"]


def test_greedy_falls_into_a_loop_on_every_open_prompt(char_lm) -> None:
    for prompt in char_prompts("open"):
        ids = char_lm.encode(prompt.text)
        text = char_lm.decode(Decoder.greedy().generate(char_lm, ids, N_NEW)[0, ids.size(1) :])
        assert longest_repeated_span(words(text)) >= 8, (prompt.text, text)


def test_repetition_penalty_reduces_greedy_loops(char_lm) -> None:
    plain = mean_open_metrics(char_lm, Decoder.greedy())
    penalised = mean_open_metrics(
        char_lm, Decoder.greedy(get_settings().decoding.default_repetition_penalty)
    )
    assert penalised["distinct_2"] > plain["distinct_2"] + 0.1
    assert penalised["longest"] < plain["longest"]


def test_beam_search_is_repetitive_on_open_prompts(char_lm) -> None:
    beam = mean_open_metrics(char_lm, BeamSearch(get_settings().decoding.default_beam_width))
    nucleus = mean_open_metrics(char_lm, Decoder.nucleus(get_settings().decoding.default_top_p))
    assert beam["distinct_2"] < nucleus["distinct_2"]


def test_find_cycle() -> None:
    assert find_cycle("abcabcabc", window=2) == (0, 3)
    assert find_cycle("xyabab", window=2) == (2, 2)
    assert find_cycle("abcdef", window=2) is None


def test_greedy_on_the_char_model_is_a_finite_state_machine(char_lm) -> None:
    """With a fixed window, greedy must cycle; every open prompt reaches a loop."""
    for prompt in char_prompts("open"):
        ids = char_lm.encode(prompt.text)
        out = Decoder.greedy().generate(char_lm, ids, N_NEW)[0].tolist()
        cycle = find_cycle(out, char_lm.context)
        assert cycle is not None, prompt.text
        start, period = cycle
        assert out[start : start + period] == out[start + period : start + 2 * period]
