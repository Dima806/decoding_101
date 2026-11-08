"""Prompts, in the two kinds the quality-diversity trade-off is measured on.

* ``open`` - open-ended continuation. Diversity is wanted; greedy and beam loop and go bland.
* ``single`` - exactly one right answer. Determinism is wanted; sampling only adds ways to be
  wrong. Each carries the expected first word of the answer, so accuracy is a number.

``CHAR_*`` prompts match the built-in corpus of the char model (lowercase, its alphabet).
``HF_*`` prompts are ordinary English for the small real model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PromptKind = Literal["open", "single"]


@dataclass(frozen=True)
class Prompt:
    """A prompt and, for single-answer prompts, the expected first word of the completion."""

    text: str
    kind: PromptKind
    answer: str | None = None

    def is_correct(self, completion: str) -> bool | None:
        """Whether the completion's first word is the answer; ``None`` for open prompts."""
        if self.answer is None:
            return None
        words = completion.strip().split()
        return bool(words) and words[0].strip(".,!?;:\"'").lower() == self.answer.lower()


CHAR_OPEN_ENDED: tuple[Prompt, ...] = (
    Prompt("the keeper", "open"),
    Prompt("in the morning", "open"),
    Prompt("the sea was", "open"),
    Prompt("one winter", "open"),
    Prompt("the fox", "open"),
)

CHAR_SINGLE_ANSWER: tuple[Prompt, ...] = (
    Prompt("the capital of france is", "single", "paris"),
    Prompt("the capital of japan is", "single", "tokyo"),
    Prompt("the opposite of hot is", "single", "cold"),
    Prompt("the colour of grass is", "single", "green"),
    Prompt("three plus four is", "single", "seven"),
    Prompt("the french word for dog is", "single", "chien"),
)

# Chosen because distilgpt2's greedy output visibly loops on them (it does not on every prompt).
HF_OPEN_ENDED: tuple[Prompt, ...] = (
    Prompt("In the morning, the sea was", "open"),
    Prompt("Once upon a time, in a small village by the sea,", "open"),
    Prompt("The meaning of life is", "open"),
)

HF_SINGLE_ANSWER: tuple[Prompt, ...] = (
    Prompt("Monday, Tuesday, Wednesday,", "single", "Thursday"),
    Prompt("January, February, March,", "single", "April"),
    Prompt("one, two, three, four,", "single", "five"),
)


def char_prompts(kind: PromptKind | None = None) -> tuple[Prompt, ...]:
    """Prompts for the char model, optionally filtered by kind."""
    prompts = CHAR_OPEN_ENDED + CHAR_SINGLE_ANSWER
    return tuple(p for p in prompts if kind is None or p.kind == kind)


def hf_prompts(kind: PromptKind | None = None) -> tuple[Prompt, ...]:
    """Prompts for the small real model, optionally filtered by kind."""
    prompts = HF_OPEN_ENDED + HF_SINGLE_ANSWER
    return tuple(p for p in prompts if kind is None or p.kind == kind)
