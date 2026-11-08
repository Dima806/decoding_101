"""Measure what the decoders do: distribution shape, repetition, quality vs diversity."""

from src.analysis.distribution import (
    effective_choices,
    entropy,
    kept_set_sizes,
    probs_from_logits,
    renormalize,
    support_size,
    temperature_sweep,
    token_probabilities,
    top_mass,
    top_tokens,
)
from src.analysis.diversity import (
    pareto_front,
    perplexity,
    pooled_distinct_n,
    sequence_logprob,
    within_distinct_n,
)
from src.analysis.repetition import (
    distinct_n,
    find_cycle,
    longest_repeated_span,
    ngrams,
    repetition_metrics,
    repetition_rate,
    valid_word_rate,
    words,
)

__all__ = [
    "distinct_n",
    "effective_choices",
    "entropy",
    "find_cycle",
    "kept_set_sizes",
    "longest_repeated_span",
    "ngrams",
    "pareto_front",
    "perplexity",
    "pooled_distinct_n",
    "probs_from_logits",
    "renormalize",
    "repetition_metrics",
    "repetition_rate",
    "sequence_logprob",
    "support_size",
    "temperature_sweep",
    "token_probabilities",
    "top_mass",
    "top_tokens",
    "valid_word_rate",
    "within_distinct_n",
    "words",
]
