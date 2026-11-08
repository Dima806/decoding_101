"""Repetition metrics: the loop, measured instead of eyeballed.

All functions take any sequence of hashable units: words (the default for text), characters,
or token ids.
"""

from __future__ import annotations

import re
from collections.abc import Hashable, Sequence

_WORD = re.compile(r"[a-z0-9']+")


def words(text: str) -> list[str]:
    """Lowercase word tokens (letters, digits, apostrophes). Punctuation is dropped."""
    return _WORD.findall(text.lower())


def ngrams(units: Sequence[Hashable], n: int) -> list[tuple[Hashable, ...]]:
    """All contiguous ``n``-grams, in order."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return [tuple(units[i : i + n]) for i in range(len(units) - n + 1)]


def distinct_n(units: Sequence[Hashable], n: int) -> float:
    """Unique ``n``-grams / total ``n``-grams (Li et al., 2016). 1.0 = nothing repeats.

    Sequences too short to contain an ``n``-gram return 1.0 (no repetition observed).
    """
    grams = ngrams(units, n)
    return len(set(grams)) / len(grams) if grams else 1.0


def repetition_rate(units: Sequence[Hashable], n: int) -> float:
    """``1 - distinct_n``: the share of ``n``-grams that are repeats."""
    return 1.0 - distinct_n(units, n)


def longest_repeated_span(units: Sequence[Hashable]) -> int:
    """Length of the longest contiguous run that occurs at least twice (overlaps allowed).

    If a span of length L repeats, so does every shorter piece of it, so the answer is found by
    binary search over L with one set of n-grams per probe: O(n log n) set operations.
    """

    def repeats(length: int) -> bool:
        grams = ngrams(units, length)
        return len(set(grams)) < len(grams)

    lo, hi = 0, max(len(units) - 1, 0)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if repeats(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def repetition_metrics(text: str) -> dict[str, float]:
    """Word-level repetition summary of one generated text."""
    units = words(text)
    return {
        "n_words": float(len(units)),
        "distinct_1": distinct_n(units, 1),
        "distinct_2": distinct_n(units, 2),
        "distinct_3": distinct_n(units, 3),
        "longest_repeat": float(longest_repeated_span(units)),
    }


def valid_word_rate(text: str, lexicon: set[str]) -> float:
    """Share of words in ``text`` that are real words, i.e. appear in ``lexicon``.

    For the char model the lexicon is the corpus vocabulary, which makes "fluent" versus
    "gibberish" a number. Empty text returns ``nan``.
    """
    units = words(text)
    return sum(w in lexicon for w in units) / len(units) if units else float("nan")


def find_cycle(units: Sequence[Hashable], window: int) -> tuple[int, int] | None:
    """Detect where a deterministic decoder starts looping, and the loop's period.

    A model that sees only the last ``window`` tokens, decoded greedily, is a finite-state
    machine: the next token is a fixed function of the current window. The first time a window
    repeats, everything after it repeats too. This finds that first repeat.

    Args:
        units: the generated sequence (prompt included is fine).
        window: the model's context length.

    Returns:
        ``(start, period)``: the sequence is periodic with ``period`` from index ``start`` on;
        ``None`` if no window repeats.
    """
    seen: dict[tuple[Hashable, ...], int] = {}
    for end in range(window - 1, len(units)):
        state = tuple(units[end - window + 1 : end + 1])
        if state in seen:
            first = seen[state]
            return first - window + 1, end - first
        seen[state] = end
    return None
