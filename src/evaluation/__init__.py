"""Check the decoders against transformers and race them against each other."""

from src.evaluation.comparison import (
    RECOMMENDATIONS,
    best_decoder,
    corpus_lexicon,
    recommend,
    run_arena,
    standard_strategies,
    summarize_arena,
)
from src.evaluation.vs_reference import (
    DEFAULT_WARPER_CASES,
    compare_generation,
    compare_warpers,
    reference_generate,
    reference_warp,
)

__all__ = [
    "DEFAULT_WARPER_CASES",
    "RECOMMENDATIONS",
    "best_decoder",
    "compare_generation",
    "compare_warpers",
    "corpus_lexicon",
    "recommend",
    "reference_generate",
    "reference_warp",
    "run_arena",
    "standard_strategies",
    "summarize_arena",
]
