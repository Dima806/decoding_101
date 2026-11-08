"""The arena: same model, same prompts, many decoders, objective metrics.

Every row holds the model fixed and varies only the decoder. Metrics per generated text:

* repetition - within-text distinct-n and the longest repeated span (loops);
* fluency - share of real words (needs a lexicon; the char model has one);
* quality - mean log-probability under the raw model (perplexity-style);
* accuracy - single-answer prompts only: is the first word the right answer?

and per (decoder, task) after :func:`summarize_arena`: diversity across samples and the
open-ended score ``valid_word_rate * distinct_2`` (high only if fluent *and* varied).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd

from src.analysis.diversity import pooled_distinct_n, sequence_logprob
from src.analysis.repetition import repetition_metrics, valid_word_rate, words
from src.config import Settings, get_settings
from src.datasets.corpus import TINY_CORPUS
from src.decoders import BeamSearch, Decoder, DecodingStrategy, make_generator

if TYPE_CHECKING:
    from src.datasets.prompts import Prompt, PromptKind
    from src.model.logits import LanguageModel


def corpus_lexicon(corpus: str = TINY_CORPUS) -> set[str]:
    """Every word in the char model's training corpus."""
    return set(words(corpus))


def standard_strategies(settings: Settings | None = None) -> list[DecodingStrategy]:
    """The line-up used across the notebooks and the app, from the settings file."""
    d = (settings or get_settings()).decoding
    return [
        Decoder.greedy(),
        Decoder.greedy(d.default_repetition_penalty),
        BeamSearch(d.default_beam_width, d.length_penalty),
        Decoder.sampling(1.0),
        Decoder.sampling(2.0),
        Decoder.top_k_sampling(d.default_top_k),
        Decoder.nucleus(d.default_top_p),
    ]


def run_arena(
    model: LanguageModel,
    prompts: Sequence[Prompt],
    strategies: Sequence[DecodingStrategy],
    *,
    n_samples: int,
    max_new_tokens: int,
    max_new_tokens_single: int = 16,
    seed: int = 0,
    lexicon: set[str] | None = None,
    score_quality: bool = True,
) -> pd.DataFrame:
    """Generate from every prompt with every strategy and score each completion.

    Deterministic strategies run once per prompt; stochastic ones run ``n_samples`` times with
    seeds ``seed, seed + 1, ...``.

    Args:
        model: the fixed model.
        prompts: open-ended and/or single-answer prompts.
        strategies: decoders to compare.
        n_samples: samples per (prompt, stochastic strategy).
        max_new_tokens: budget for open-ended prompts.
        max_new_tokens_single: budget for single-answer prompts (only the first word counts).
        seed: base seed.
        lexicon: real words, for ``valid_word_rate``; ``None`` leaves it ``nan``.
        score_quality: compute mean log-probability (slower on big models).

    Returns:
        One row per generated text.
    """
    records = []
    for prompt in prompts:
        ids = model.encode(prompt.text)
        prompt_len = ids.size(1)
        budget = max_new_tokens if prompt.kind == "open" else max_new_tokens_single
        for strategy in strategies:
            for sample in range(1 if strategy.is_deterministic else n_samples):
                out = strategy.generate(
                    model,
                    ids,
                    budget,
                    eos_token_id=model.eos_token_id,
                    generator=make_generator(seed + sample),
                )
                completion = model.decode(out[0, prompt_len:])
                records.append(
                    {
                        "prompt": prompt.text,
                        "kind": prompt.kind,
                        "decoder": strategy.label,
                        "deterministic": strategy.is_deterministic,
                        "sample": sample,
                        "completion": completion,
                        **repetition_metrics(completion),
                        "valid_word_rate": (
                            valid_word_rate(completion, lexicon) if lexicon else math.nan
                        ),
                        "mean_logprob": (
                            sequence_logprob(model, out, prompt_len) if score_quality else math.nan
                        ),
                        "correct": prompt.is_correct(completion),
                    }
                )
    df = pd.DataFrame.from_records(records)
    order = list(dict.fromkeys(s.label for s in strategies))
    df["decoder"] = pd.Categorical(df["decoder"], categories=order, ordered=True)
    return df


def summarize_arena(df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    """One row per decoder: open-ended repetition, diversity, fluency, quality, and accuracy.

    Diversity pools distinct-2 across the samples of each prompt. A deterministic decoder would
    return the same text on every run, so its single output is counted ``n_samples`` times.

    Returns:
        Columns ``open_distinct_2, open_longest_repeat, open_diversity, open_valid_words,
        open_logprob, open_score, single_accuracy`` indexed by decoder.
    """
    rows = []
    for decoder, group in df.groupby("decoder", observed=True, sort=True):
        open_ = group[group["kind"] == "open"]
        single = group[group["kind"] == "single"]
        diversities = []
        for _, by_prompt in open_.groupby("prompt"):
            texts = list(by_prompt["completion"])
            if bool(by_prompt["deterministic"].iloc[0]):
                texts = texts * n_samples
            diversities.append(pooled_distinct_n(texts, 2))
        distinct_2 = _mean(open_["distinct_2"])
        valid = _mean(open_["valid_word_rate"])
        rows.append(
            {
                "decoder": decoder,
                "open_distinct_2": distinct_2,
                "open_longest_repeat": _mean(open_["longest_repeat"]),
                "open_diversity": sum(diversities) / len(diversities) if diversities else math.nan,
                "open_valid_words": valid,
                "open_logprob": _mean(open_["mean_logprob"]),
                "open_score": distinct_2 * valid if not math.isnan(valid) else distinct_2,
                "single_accuracy": _mean(single["correct"].astype(float)),
            }
        )
    return pd.DataFrame.from_records(rows).set_index("decoder")


def _mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else math.nan


RECOMMENDATIONS: dict[str, dict[str, str]] = {
    "single": {
        "use": "greedy or beam search, temperature 0",
        "why": (
            "There is one right answer and it is usually the most probable continuation. "
            "Sampling only adds ways to be wrong; beam search helps when the best answer "
            "needs a less likely first token."
        ),
        "examples": "translation, extraction, classification, structured output, arithmetic",
    },
    "open": {
        "use": "nucleus (top-p 0.8-0.95) with temperature 0.7-1.0; add a repetition penalty "
        "if loops appear",
        "why": (
            "You want variety without nonsense. Temperature controls how adventurous the "
            "sampler is; top-p cuts the long tail of absurd tokens and adapts to how sure the "
            "model is. Greedy and beam loop or go bland here."
        ),
        "examples": "stories, brainstorming, dialogue, marketing copy",
    },
}


def recommend(kind: PromptKind) -> dict[str, str]:
    """The decision-framework rule for a task kind."""
    return RECOMMENDATIONS[kind]


def best_decoder(summary: pd.DataFrame, kind: PromptKind) -> str:
    """The arena winner for a task kind.

    Single-answer: highest accuracy (ties go to the earlier, simpler decoder).
    Open-ended: highest ``open_score`` = fluent words x non-repeated bigrams.
    """
    column = "single_accuracy" if kind == "single" else "open_score"
    return str(summary[column].astype(float).idxmax())
