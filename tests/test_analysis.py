"""Distribution, diversity, and arena metrics, plus settings loading."""

from __future__ import annotations

import math

import pandas as pd
import pytest
import torch

from src.analysis import (
    effective_choices,
    entropy,
    kept_set_sizes,
    pareto_front,
    perplexity,
    pooled_distinct_n,
    sequence_logprob,
    support_size,
    temperature_sweep,
    token_probabilities,
    top_mass,
    top_tokens,
    valid_word_rate,
    within_distinct_n,
)
from src.config import PROJECT_ROOT, load_settings
from src.datasets import Prompt, char_prompts
from src.decoders import BeamSearch, Decoder, apply_top_k, make_generator
from src.evaluation import (
    best_decoder,
    corpus_lexicon,
    recommend,
    run_arena,
    standard_strategies,
    summarize_arena,
)

# --- distribution ---------------------------------------------------------------------------


def test_entropy_of_uniform_is_log_vocab() -> None:
    logits = torch.zeros(1, 8)
    assert entropy(logits).item() == pytest.approx(math.log(8))
    assert entropy(logits, base=2).item() == pytest.approx(3.0)
    assert effective_choices(logits).item() == pytest.approx(8.0)


def test_entropy_ignores_masked_tokens() -> None:
    masked = apply_top_k(torch.tensor([[1.0, 1.0, -5.0, -9.0]]), 2)
    assert entropy(masked, base=2).item() == pytest.approx(1.0)
    assert not math.isnan(entropy(torch.tensor([[0.0, float("-inf")]])).item())


def test_top_mass_and_support_size() -> None:
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.2]]))
    assert top_mass(logits, 2).item() == pytest.approx(0.8)
    assert support_size(apply_top_k(logits, 2)).item() == 2


def test_temperature_sweep_table() -> None:
    df = temperature_sweep(torch.randn(50, generator=torch.Generator().manual_seed(0)), [0, 1, 5])
    assert list(df.columns) == ["temperature", "entropy_bits", "effective_choices", "top1_prob"]
    assert df["top1_prob"].iloc[0] == pytest.approx(1.0)
    assert df["entropy_bits"].is_monotonic_increasing


def test_top_tokens_table(char_lm) -> None:
    df = top_tokens(char_lm, char_lm.next_token_logits(char_lm.encode("the sea")), n=5)
    assert list(df.columns) == ["rank", "token_id", "token", "logit", "prob"]
    assert len(df) == 5
    assert df["prob"].is_monotonic_decreasing


def test_fixed_k_ignores_confidence_but_nucleus_adapts() -> None:
    tail = -0.01 * torch.arange(19, dtype=torch.float32)  # untied, so top-k keeps exactly k
    peaked = torch.cat([torch.tensor([5.0]), tail])  # top token ~0.9 of the mass
    flat = -0.01 * torch.arange(20, dtype=torch.float32)  # near-uniform
    df = kept_set_sizes({"peaked": peaked, "flat": flat}, ks=[5], ps=[0.8])
    k = df[df["method"] == "top-k"].set_index("distribution")["kept"]
    p = df[df["method"] == "top-p"].set_index("distribution")["kept"]
    assert k["peaked"] == k["flat"] == 5
    assert p["peaked"] == 1
    assert p["flat"] >= 15


# --- diversity and quality ------------------------------------------------------------------


def test_greedy_text_is_more_likely_than_high_temperature_text(char_lm) -> None:
    ids = char_lm.encode("the keeper")
    n = ids.size(1)
    greedy = Decoder.greedy().generate(char_lm, ids, 80)
    hot = Decoder.sampling(3.0).generate(char_lm, ids, 80, generator=make_generator(0))
    lp_greedy = sequence_logprob(char_lm, greedy, n)
    lp_hot = sequence_logprob(char_lm, hot, n)
    assert lp_greedy > lp_hot
    assert perplexity(lp_greedy) < perplexity(lp_hot)
    assert math.isnan(sequence_logprob(char_lm, ids, n))


def test_token_probabilities_and_summed_logprob(char_lm) -> None:
    ids = char_lm.encode("the fox")
    out = Decoder.greedy().generate(char_lm, ids, 20)
    trace = token_probabilities(char_lm, out, ids.size(1))
    assert len(trace) == 20
    assert (trace["p_chosen"] == trace["p_top1"]).all()  # greedy always takes the top token
    total = sequence_logprob(char_lm, out, ids.size(1), reduction="sum")
    assert total == pytest.approx(float(trace["p_chosen"].map(math.log).sum()), rel=1e-5)


def test_pooled_versus_within_diversity() -> None:
    same = ["the sea is calm"] * 3
    varied = ["the sea is calm", "a fox ran home", "bread for sale today"]
    assert pooled_distinct_n(same) == pytest.approx(1 / 3)
    assert within_distinct_n(same) == 1.0  # each text alone has no repeats
    assert pooled_distinct_n(varied) == 1.0
    assert pooled_distinct_n([]) == 1.0


def test_valid_word_rate() -> None:
    lexicon = {"the", "sea"}
    assert valid_word_rate("the sea the sea", lexicon) == 1.0
    assert valid_word_rate("the zqx", lexicon) == 0.5
    assert math.isnan(valid_word_rate("", lexicon))


def test_pareto_front() -> None:
    df = pd.DataFrame({"diversity": [0.1, 0.5, 0.9, 0.4], "quality": [0.9, 0.7, 0.2, 0.3]})
    front = pareto_front(df)
    assert list(front.index) == [0, 1, 2]  # row 3 is dominated by row 1


# --- arena ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def arena(char_lm) -> tuple[pd.DataFrame, pd.DataFrame]:
    prompts = char_prompts("open")[:3] + char_prompts("single")[:3]
    df = run_arena(
        char_lm,
        prompts,
        standard_strategies(),
        n_samples=3,
        max_new_tokens=200,
        lexicon=corpus_lexicon(),
        score_quality=False,
    )
    return df, summarize_arena(df, n_samples=3)


def test_arena_rows_and_columns(arena) -> None:
    df, summary = arena
    # 3 deterministic strategies run once, 4 stochastic run 3 times: 15 rows per prompt.
    assert len(df) == 6 * (3 + 4 * 3)
    assert {"completion", "distinct_2", "valid_word_rate", "correct"} <= set(df.columns)
    assert list(summary.index) == [s.label for s in standard_strategies()]


def test_beam_wins_single_answer_and_loses_open_ended(arena) -> None:
    _, summary = arena
    beam = summary.loc[BeamSearch(4).label]
    nucleus = summary.loc[Decoder.nucleus(0.8).label]
    hot = summary.loc[Decoder.sampling(2.0).label]
    assert beam["single_accuracy"] == 1.0
    assert beam["single_accuracy"] > hot["single_accuracy"]
    assert beam["open_score"] < nucleus["open_score"]
    assert hot["open_valid_words"] < nucleus["open_valid_words"]  # high T: not real words


def test_task_matcher_picks_a_deterministic_decoder_for_single_answers(arena) -> None:
    _, summary = arena
    assert best_decoder(summary, "single") in {"greedy", "greedy, rep=1.3", "beam (B=4)"}
    assert best_decoder(summary, "open") not in {"greedy", "beam (B=4)", "T=2"}
    assert "greedy" in recommend("single")["use"]
    assert "nucleus" in recommend("open")["use"]


def test_prompt_scoring() -> None:
    p = Prompt("the capital of france is", "single", "paris")
    assert p.is_correct(" paris.\nthe") is True
    assert p.is_correct(" pairs") is False
    assert p.is_correct("") is False
    assert Prompt("the sea", "open").is_correct("anything") is None


# --- settings -------------------------------------------------------------------------------


def test_settings_load_from_yaml() -> None:
    s = load_settings()
    assert s.char_lm.context == 8
    assert 0.0 in s.decoding.temperatures
    assert s.figures_dir == PROJECT_ROOT / "outputs" / "figures"
    assert s.results_dir == PROJECT_ROOT / "outputs" / "results"


def test_save_json_converts_everything(tmp_path) -> None:
    import json

    import numpy as np

    from src.results import save_json

    df = pd.DataFrame({"a": [1.0, math.nan]}, index=pd.Index(["x", "y"], name="k"))
    data = {"df": df, "t": torch.tensor([1, 2]), "np": np.float32(0.5), "inf": math.inf}
    path = save_json("smoke", data, directory=tmp_path)
    payload = json.loads(path.read_text())
    assert payload["name"] == "smoke"
    assert payload["seed"] == load_settings().seed
    assert payload["data"] == {
        "df": [{"k": "x", "a": 1.0}, {"k": "y", "a": None}],
        "t": [1, 2],
        "np": 0.5,
        "inf": None,
    }


def test_environment_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DECODING101_GENERATION__MAX_NEW_TOKENS", "42")
    s = load_settings()
    assert s.generation.max_new_tokens == 42
    assert s.generation.n_samples == 5  # sibling keys still come from the YAML
