"""Built-in corpus for the char model and the open-ended / single-answer prompt sets."""

from src.datasets.corpus import TINY_CORPUS
from src.datasets.prompts import (
    CHAR_OPEN_ENDED,
    CHAR_SINGLE_ANSWER,
    HF_OPEN_ENDED,
    HF_SINGLE_ANSWER,
    Prompt,
    PromptKind,
    char_prompts,
    hf_prompts,
)

__all__ = [
    "CHAR_OPEN_ENDED",
    "CHAR_SINGLE_ANSWER",
    "HF_OPEN_ENDED",
    "HF_SINGLE_ANSWER",
    "TINY_CORPUS",
    "Prompt",
    "PromptKind",
    "char_prompts",
    "hf_prompts",
]
