"""Decoding 101 - the app.

Same model, same prompt, different decoder. Four views:
distribution viewer, decoder race, temperature sweep, task matcher.
All logic lives in ``src/``; this file only lays it out.
"""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from src.analysis import (
    entropy,
    repetition_metrics,
    temperature_sweep,
    top_tokens,
    valid_word_rate,
)
from src.config import get_settings
from src.datasets import char_prompts, hf_prompts
from src.decoders import Decoder, make_generator
from src.evaluation import (
    best_decoder,
    corpus_lexicon,
    recommend,
    run_arena,
    standard_strategies,
    summarize_arena,
)
from src.model import LanguageModel, ModelSource, load_model, next_token_logits
from src.visualisation import plotly_decoder_bars, plotly_lines, plotly_top_tokens

SETTINGS = get_settings()
LEXICON = corpus_lexicon()

st.set_page_config(page_title="Decoding 101", layout="wide")


@st.cache_resource(show_spinner="Loading the model (the char model trains in a few seconds)...")
def get_model(source: ModelSource) -> LanguageModel:
    return load_model(source)


def generate_text(
    source: ModelSource, prompt: str, decoder: Decoder, max_new_tokens: int, seed: int
) -> str:
    model = get_model(source)
    ids = model.encode(prompt)
    out = decoder.generate(
        model, ids, max_new_tokens, model.eos_token_id, generator=make_generator(seed)
    )
    return model.decode(out[0, ids.size(1) :])


def text_metrics(source: ModelSource, text: str) -> dict[str, float]:
    metrics = repetition_metrics(text)
    if source == "char":
        metrics["valid_word_rate"] = valid_word_rate(text, LEXICON)
    return metrics


@st.cache_data(show_spinner=False)
def race(source: ModelSource, prompt: str, max_new_tokens: int, seed: int) -> pd.DataFrame:
    model = get_model(source)
    ids = model.encode(prompt)
    rows = []
    for strategy in standard_strategies(SETTINGS):
        out = strategy.generate(
            model, ids, max_new_tokens, model.eos_token_id, generator=make_generator(seed)
        )
        text = model.decode(out[0, ids.size(1) :])
        rows.append({"decoder": strategy.label, "text": text, **text_metrics(source, text)})
    return pd.DataFrame.from_records(rows).set_index("decoder")


@st.cache_data(show_spinner=False)
def sweep_generations(
    source: ModelSource, prompt: str, max_new_tokens: int, seed: int
) -> pd.DataFrame:
    rows = []
    for t in SETTINGS.decoding.temperatures:
        text = generate_text(source, prompt, Decoder.sampling(t), max_new_tokens, seed)
        rows.append({"temperature": t, **text_metrics(source, text)})
    return pd.DataFrame.from_records(rows)


@st.cache_data(show_spinner="Running the arena...")
def arena_summary(source: ModelSource, kind: str, seed: int) -> pd.DataFrame:
    model = get_model(source)
    is_char = source == "char"
    prompts = char_prompts("open" if kind == "open" else "single")
    if not is_char:
        prompts = hf_prompts("open" if kind == "open" else "single")
    n_samples = SETTINGS.generation.n_samples if is_char else 2
    df = run_arena(
        model,
        prompts,
        standard_strategies(SETTINGS),
        n_samples=n_samples,
        max_new_tokens=SETTINGS.generation.max_new_tokens
        if is_char
        else SETTINGS.hf.max_new_tokens,
        max_new_tokens_single=16 if is_char else 4,
        seed=seed,
        lexicon=LEXICON if is_char else None,
        score_quality=False,
    )
    return summarize_arena(df, n_samples)


# --- sidebar --------------------------------------------------------------------------------

st.sidebar.title("Decoding 101")
st.sidebar.caption("The model emits a fixed distribution. The sampler decides what gets said.")
source: ModelSource = st.sidebar.radio(
    "Model",
    options=["char", "hf"],
    format_func=lambda s: "From-scratch char model" if s == "char" else SETTINGS.hf.model_id,
    help="The char model trains in seconds and has a ~30-token vocabulary you can read in full. "
    f"{SETTINGS.hf.model_id} is real English but downloads ~350 MB on first use.",
)
seed = int(st.sidebar.number_input("Seed", value=SETTINGS.seed, step=1))
is_char = source == "char"
default_budget = SETTINGS.generation.max_new_tokens if is_char else SETTINGS.hf.max_new_tokens
max_new_tokens = st.sidebar.slider(
    "New tokens" + (" (characters)" if is_char else ""),
    min_value=10,
    max_value=400 if is_char else 100,
    value=default_budget,
)
default_prompt = "the keeper" if is_char else "In the morning, the sea was"

tab_dist, tab_race, tab_sweep, tab_task = st.tabs(
    ["Distribution viewer", "Decoder race", "Temperature sweep", "Task matcher"]
)

# --- 1. distribution viewer -----------------------------------------------------------------

with tab_dist:
    st.subheader("One forward pass, one fixed distribution")
    prompt = st.text_input("Prompt", default_prompt, key="dist_prompt")
    c1, c2, c3 = st.columns(3)
    temperature = c1.slider("Temperature", 0.0, 3.0, 1.0, 0.05, key="dist_t")
    top_k = c2.slider("Top-k (0 = off)", 0, 50, 0, key="dist_k")
    top_p = c3.slider("Top-p (1 = off)", 0.05, 1.0, 1.0, 0.05, key="dist_p")

    model = get_model(source)
    logits = next_token_logits(model, prompt)
    warped = Decoder(temperature=temperature, top_k=top_k, top_p=top_p).warp(
        logits, model.encode(prompt)
    )
    raw_table = top_tokens(model, logits, n=20)
    ids = raw_table["token_id"].tolist()
    warped_table = raw_table.assign(prob=warped.softmax(dim=-1)[0, ids].tolist())
    removed = [not math.isfinite(v) for v in warped[0, ids].tolist()]

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Entropy (bits)",
        f"{entropy(warped, base=2).item():.2f}",
        f"{entropy(warped, base=2).item() - entropy(logits, base=2).item():+.2f} vs raw",
    )
    m2.metric("Tokens still allowed", f"{int(warped.isfinite().sum())} / {model.vocab_size}")
    m3.metric("Top-1 probability", f"{warped.softmax(dim=-1).max().item():.2f}")
    left, right = st.columns(2)
    left.plotly_chart(plotly_top_tokens(raw_table, "The model (never changes)"))
    right.plotly_chart(plotly_top_tokens(warped_table, "After the decoder", removed))
    st.caption("Grey bars were cut by top-k / top-p. Temperature never removes a token.")

# --- 2. decoder race ------------------------------------------------------------------------

with tab_race:
    st.subheader("Same model, same prompt, seven decoders")
    race_prompt = st.text_input("Prompt", default_prompt, key="race_prompt")
    results = race(source, race_prompt, max_new_tokens, seed)
    for label, row in results.iterrows():
        with st.container(border=True):
            st.markdown(
                f"**{label}** - distinct-2 `{row['distinct_2']:.2f}`, longest repeat "
                f"`{int(row['longest_repeat'])}` words"
            )
            st.text(race_prompt + row["text"])
    st.plotly_chart(
        plotly_decoder_bars(results, "distinct_2", "Distinct-2 (1 = no repeated word pairs)")
    )

# --- 3. temperature sweep -------------------------------------------------------------------

with tab_sweep:
    st.subheader("Temperature is one line: logits / T")
    sweep_prompt = st.text_input("Prompt", default_prompt, key="sweep_prompt")
    t = st.slider("Temperature", 0.0, 5.0, SETTINGS.decoding.default_temperature, 0.05)
    st.text(
        sweep_prompt
        + generate_text(source, sweep_prompt, Decoder.sampling(t), max_new_tokens, seed)
    )

    model = get_model(source)
    grid = sorted({*SETTINGS.decoding.temperatures, round(t, 2)})
    curve = temperature_sweep(next_token_logits(model, sweep_prompt), grid)
    left, right = st.columns(2)
    left.plotly_chart(
        plotly_lines(
            curve,
            "temperature",
            {"entropy_bits": "entropy (bits)"},
            title="Next-token entropy",
            xlabel="T",
            vline=t,
        )
    )
    gens = sweep_generations(source, sweep_prompt, max_new_tokens, seed)
    ys = {"distinct_2": "distinct-2 (varied)"}
    if is_char:
        ys["valid_word_rate"] = "real-word rate (fluent)"
    right.plotly_chart(
        plotly_lines(gens, "temperature", ys, title="Generated text", xlabel="T", vline=t)
    )
    st.caption("Low T: real words, repeating. High T: never repeats, not words. In between: both.")

# --- 4. task matcher ------------------------------------------------------------------------

with tab_task:
    st.subheader("Match the decoder to the task")
    kind = st.radio(
        "Task",
        options=["single", "open"],
        format_func=lambda k: "Single correct answer" if k == "single" else "Open-ended",
        horizontal=True,
    )
    rule = recommend("single" if kind == "single" else "open")
    st.markdown(f"**Use:** {rule['use']}  \n**Why:** {rule['why']}  \n**For:** {rule['examples']}")
    summary = arena_summary(source, kind, seed)
    metric = "single_accuracy" if kind == "single" else "open_score"
    winner = best_decoder(summary, "single" if kind == "single" else "open")
    st.success(f"Arena winner on this model: **{winner}**")
    st.plotly_chart(
        plotly_decoder_bars(
            summary,
            metric,
            "Accuracy (first word right)"
            if kind == "single"
            else "Open-ended score = real-word rate x distinct-2",
        )
    )
    st.dataframe(summary.round(3))
