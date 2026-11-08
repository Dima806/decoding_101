# Project: decoding_101

> Source of truth for scope: `.llm/PRD_decoding_101.md` (gitignored, local only).
> This file is the operational contract. If the two disagree, the PRD wins on *what*,
> this file wins on *how*. Update this file when either changes.

---

## 1. Identity

LLM **decoding / sampling built from scratch**, CPU-only, for a data-science audience.
Deliverables: 6 notebooks + `src/` library + 1 Streamlit app + tests.

Build-up (each notebook depends on the one before):
`01 logits to text` → `02 greedy and the loop` → `03 temperature is one line` →
`04 top-k and nucleus` → `05 beam search` → `06 decision framework`.

**Thesis.** The model emits a *fixed* probability distribution over the next token. A tiny,
hand-written sampler then decides what gets said. Greedy loops. High temperature rambles.
Nucleus cuts the absurd tail. Beam is safe and bland. The model never changed — the decoder did.

**Showpiece:** notebook 03. Temperature is one line of arithmetic (`logits / T`), and the sweep
from loop → fluent → gibberish happens on one frozen distribution.

**Related projects** (github.com/Dima806; cross-link in prose, never import). The canonical
list with one-line descriptions is README "Related projects", grouped as: *where the logits come
from* (`tokenization_101`, `embeddings_101`, `transformers_101`, `backprop_101`), *ideas decoding
borrows* (`sampling_101`, `information_theory_101`, `bandits_101`, `calibration_arena`), *LLMs in
practice* (`quantization_101`, `lora_101`, `rag_101`, `llm_judge_benchmark`). Refresh with
`gh api "users/Dima806/repos?per_page=100" --paginate`, and before claiming a link (e.g.
"bandits_101 covers X") check that repo's README: `gh api repos/Dima806/<repo>/readme`.
`bandits_101` covers epsilon-greedy/UCB/Thompson, **not** softmax/Boltzmann exploration.

---

## 2. Current state

Built: `src/` (all modules), `app/streamlit_app.py`, `config/settings.yaml`, 6 executed
notebooks, 108 tests (96% coverage), `outputs/figures/` (17 PNGs), `outputs/results/` (35 JSONs).
`make ci` is green. README is written (results table, notebooks, quickstart, limitations,
related projects). **Not built yet:** `.devcontainer/`, `.github/workflows/ci.yml` (deliberately
deferred).

Toolchain as resolved in `uv.lock`: Python 3.14, torch 2.15 (CPU wheel via the pytorch-cpu
index), transformers 5.22. Notebook runtime on the 2-CPU box: 20 s to 2.5 min each (05 is the
slowest), ~6.5 min for `make notebooks`. **Keep this section current.**

---

## 3. Stack and hard constraints

| Thing | Decision |
|---|---|
| Python | 3.11+ (`requires-python = ">=3.11"`, ruff `target-version = "py311"`) |
| Package manager | `uv` only. **Never `pip install`**, never `python -m venv`, never edit `.venv` by hand |
| Hardware budget | 2 CPU cores, 8 GB RAM, **no GPU**. Everything must run in a GitHub Codespace |
| Tensors | `torch` on CPU, `float32`, `device="cpu"` always. No `.cuda()`, no `device_map`, no `accelerate` |
| Licensing | 100% open-source. No API keys, no paid endpoints, no network at inference time |
| Notebook runtime | < 5 minutes each, enforced by `make notebooks` (`timeout=300`) |
| Lint / types / tests | `ruff` (line-length 99), `ty`, `pytest`. `make lint` and `make test` must be clean |

Runtime deps: `torch`, `transformers`, `httpx`, `numpy`, `scipy`, `pandas`, `matplotlib`,
`plotly`, `streamlit`, `pydantic`, `pydantic-settings`, `pyyaml`.
Dev deps: `pytest`, `ruff`, `ty`, `jupyter`, `ipykernel`, `nbconvert`, `nbclient`.
Adding a dependency needs a reason in the PR description; prefer stdlib/numpy/torch.

---

## 4. Commands

```
make setup       # first run: install uv, uv sync --all-extras, register ipykernel
make sync        # uv sync --all-extras
make models      # OPTIONAL: install ollama + pull qwen2.5:0.5b (never required by tests/CI)
make lint        # format + check + typecheck
make test        # pytest
make test-cov    # pytest --cov=src --cov-report=term-missing
make test-offline  # HF_HUB_OFFLINE=1, deselects @pytest.mark.network
make notebooks   # execute all notebooks in place, 300s timeout each (~6.5 min total)
make run         # streamlit app on :8501
make lab         # jupyterlab on :8888
make ci          # sync + lint + test   <- must pass before any commit
make dev         # lint + test (fast loop)
```

Run anything ad hoc through `uv run …`. Bare `python foo.py` uses the wrong interpreter.

---

## 5. Layout and module responsibilities

```
src/
  config.py            Settings (pydantic-settings) loaded from config/settings.yaml
  decoders/
    __init__.py        Decoder dataclass, DecodingStrategy protocol, generate(), sampling, RNG
    greedy.py          argmax
    temperature.py     logits / T
    top_k.py           keep k most likely, mask rest
    top_p.py           nucleus: smallest prefix whose cumulative prob >= p
    beam.py            beam search over sequences
    penalties.py       repetition (CTRL, multiplicative) + presence/frequency (OpenAI, additive)
  model/
    char_lm.py         from-scratch char-level LM — fully transparent logits
    logits.py          next-token logits from a small HF model (and optional Ollama) behind one API
  analysis/
    distribution.py    entropy, top-mass, temperature sweep, kept-set sizes, per-step token probs
    repetition.py      distinct-n, longest repeated span, find_cycle, valid_word_rate (fluency)
    diversity.py       sequence log-prob / perplexity, pooled diversity, Pareto front
  datasets/
    corpus.py          built-in training text for the char model (original prose + facts)
    prompts.py         CHAR_* and HF_* prompt sets, open-ended vs single-answer (with answers)
  evaluation/
    vs_reference.py    our decoders vs transformers' warpers / generate()
    comparison.py      arena + summary, standard line-up, recommend(), best_decoder()
  visualisation.py     every figure in the project (matplotlib for PNGs, plotly for the app)
  results.py           save_json(): numbers and text behind every figure -> outputs/results/
app/streamlit_app.py   4 tabs: distribution viewer, decoder race, temperature sweep, task matcher
config/settings.yaml   temperatures, k values, p values, beam widths, seeds, model ids
notebooks/01..06       narrative only — they import from src/, they do not define algorithms
tests/                 the 4 PRD files + test_analysis, test_model, test_visualisation, test_app
tests/conftest.py      session char model, offline tiny GPT-2 factory, TableLM toy model
tests/fixtures/        distilgpt2_logits.npz + record_logits.py (regenerates it)
outputs/figures/       committed PNGs, NN_<slug>.png (budget: 19 total, see §9)
outputs/results/       committed JSON, NN_<slug>.json, same slug as the figure it backs
```

Rules:
- **No algorithm lives in a notebook or in the app.** If a notebook needs it, it goes in `src/`.
- Files outside this tree need a line in the PR description saying why.
- `src/` has no top-level side effects: no model loading, no file I/O, no seeding at import.

---

## 6. The math, stated exactly

Implement these definitions, not approximations of them. Each bullet is also a test.

**Softmax.** `p_i = exp(z_i - max z) / Σ exp(z_j - max z)`. Always subtract the max; never call
`exp` on raw logits. `-inf` logits map to exactly `0.0`.

**Temperature.** `z' = z / T` for `T > 0`. Our API treats `T == 0` as argmax (special-cased
*before* any division); `T < 0` raises `ValueError`. Limits that must hold:
`T → 0⁺` ⇒ one-hot on argmax; `T → ∞` ⇒ uniform over the vocab. Entropy is monotonically
non-decreasing in `T` on a non-degenerate distribution.

**Top-k.** `k ← min(k, vocab_size)`; let `τ` be the k-th largest logit; mask every token with
`z_i < τ`. Ties at `τ` are all kept, so the kept set can exceed `k` — this matches
`transformers.TopKLogitsWarper` and a naive `torch.topk`-indices implementation does not.
`k <= 0` means "disabled".

**Top-p (nucleus).** Sort probs descending. `k* = min{ k : Σ_{i<=k} p_i >= p }`. Keep those `k*`
tokens, mask the rest. The token that *crosses* `p` is **included**. Always keep at least
`min_tokens_to_keep = 1`, so `p` smaller than the top probability still yields one token.
`p >= 1.0` keeps everything. Use `torch.sort(..., stable=True)` so ties are reproducible.
HF writes the same thing as ascending sort + `cumprobs <= 1 - p`; they agree up to float error.

**Repetition penalty** (Keskar et al. 2019, CTRL — what HF implements). Applied to **logits**,
multiplicatively, sign-dependent, to every token already in the context:
`z_t ← z_t * ρ if z_t < 0 else z_t / ρ`. `ρ = 1.0` is off; useful range 1.0–1.3.
Dividing unconditionally is wrong for negative logits — that *boosts* them.

**Presence / frequency penalty** (OpenAI-style, additive, our own implementation since HF has
neither): `z_t ← z_t - presence * 1[c_t > 0] - frequency * c_t`, where `c_t` is the count of token
`t` in the context. `0.0` is off.

**Beam search.** Work in log space, accumulate `Σ log p`. Each step: expand all `B` live beams,
take the top `2B` candidates globally, move any candidate ending in EOS into a finished pool,
keep the best `B` unfinished. Final score is **length-normalised**:
`score = Σ log p / (len ** length_penalty)` with `length_penalty = 1.0` by default, where `len`
is the number of **generated** tokens (prompt excluded, EOS included). That is what
transformers >= 4.36 does (`cur_len + 1 - decoder_prompt_len` in `_update_finished_beams`);
the old `hyp.shape[-1]` form is a legacy branch. Only the top B candidates may finish; stop when
B hypotheses are finished and the best live beam's normalised score cannot beat the worst
(`early_stopping=False` heuristic). Beam search is orthogonal to temperature/top-k/top-p.

**Entropy.** `H(p) = -Σ p_i log p_i` in **nats** by default, with an explicit `base` argument.
Label plot axes with the unit. Skip `p_i == 0` terms rather than producing `nan`.

---

## 7. API conventions

**One shape.** Logits are `torch.Tensor`, `float32`, shape `(batch, vocab)` — 2-D even when
`batch == 1`. Beam search operates on `(num_beams, vocab)`. Token ids are `int64`,
shape `(batch, seq)`. No numpy inside the decoders; numpy is for analysis and plotting only.

**Warpers are pure functions on logits.**

```python
def apply_temperature(logits: Tensor, temperature: float) -> Tensor: ...
def apply_top_k(logits: Tensor, k: int, filter_value: float = -float("inf")) -> Tensor: ...
def apply_top_p(logits: Tensor, p: float, min_tokens_to_keep: int = 1) -> Tensor: ...
def apply_repetition_penalty(logits: Tensor, input_ids: Tensor, penalty: float) -> Tensor: ...
```

They take logits and return **logits** with removed tokens set to `-inf`. Never return
probabilities, never mutate the input (`logits = logits.clone()` first), never renormalize
inside a warper. This keeps them composable and directly comparable to HF's warpers.

**Masking vs renormalizing.** Internally: mask with `-inf`, then one softmax at the end. That is
bit-identical to HF and avoids a second normalization. For *teaching*, `src/analysis/distribution.py`
may expose a "zero the tail and divide by the sum" view; assert the two agree with
`torch.allclose(..., atol=1e-6)` rather than asserting equality.

**Pipeline order** (same as HF `generate`, so the reference tests can pass):
penalties → temperature → top-k → top-p → softmax → sample.
Changing this order changes the output; if a notebook explores a different order, say so explicitly.

**Sampling.** One entry point, one RNG:

```python
def sample_from_logits(logits: Tensor, generator: torch.Generator) -> Tensor: ...  # multinomial
```

`torch.Generator` threaded explicitly through every stochastic call. Never `random`,
never `np.random`, never an unseeded global. One `torch.multinomial(probs, num_samples=1,
generator=g)` draw per step so the RNG stream is predictable.

**The generation loop** lives once, in `src/decoders/__init__.py`:

```python
def generate(model, input_ids, *, decoder, max_new_tokens, eos_token_id=None,
             generator=None) -> Tensor: ...
```

`Decoder` is one frozen dataclass (`temperature, top_k, top_p, repetition_penalty,
presence_penalty, frequency_penalty`) with `warp(logits, ids)` and
`__call__(logits, ids, generator) -> (batch, 1)`; `T == 0` means argmax. Greedy, sampling,
nucleus and penalised variants are just different instances (`Decoder.greedy()`,
`Decoder.nucleus(p, T)`, ...). Beam search has its own loop in `beam.py`; `BeamSearch` and
`Decoder` both satisfy `DecodingStrategy` (`label`, `is_deterministic`, `generate(...)`), which is
what the arena and the app iterate over. Sampling decoders raise without a seeded `Generator`.

**Models.** Both sources implement the `LanguageModel` protocol in `src/model/logits.py`
(`next_token_logits(ids) -> (batch, vocab)`, `encode`, `decode`, `token_labels`, `vocab_size`,
`eos_token_id`). Helpers: `next_token_logits(model, prompt) -> (1, vocab)` and
`analysis.top_tokens(model, logits, n)` for labelled tables. `load_model("char" | "hf")` is
cached. The char model (MLP, **context 8**) is the transparent default: context 8 is chosen so
greedy visibly cycles while all facts are still memorised; 16 memorises the corpus and greedy
just recites it. HF: `distilgpt2` for notebooks; tests build a **random tiny GPT-2 from a config**
(offline) and only the `network` test downloads `sshleifer/tiny-gpt2`. `transformers` is
imported lazily. Ollama (`qwen2.5:0.5b` over `httpx`) is **strictly optional**, text only (no
logits): feature-detect with `ollama_available()`, never let anything hard-fail without it.

**Config.** `src/config.py` defines pydantic models and loads `config/settings.yaml`.
Temperatures, k values, p values, beam widths, seeds, model ids, max_new_tokens all live in YAML —
no magic numbers in `src/`, notebooks, or the app. Env override prefix `DECODING101_`.

**Typing.** Full annotations on every public function; `from __future__ import annotations` not
needed on 3.11+ for builtins generics. Docstrings state shapes and the formula. `ty check src/`
must be clean.

---

## 8. Invariants the tests must assert

Mapped to PRD §8. Deterministic decoders get exact equality; stochastic ones get exact equality
on the *filtered distribution* plus a statistical check on the draws — never a vibes check.

`tests/test_decoders.py`
- top-k keeps exactly the right token set, including the tie case at the threshold.
- top-p keeps the smallest prefix reaching `p`, with the crossing token **included**.
- every warper output renormalizes to `1.0` after softmax (`atol=1e-6`).
- `min_tokens_to_keep` holds for tiny `p` and for `k=1`.
- warpers do not mutate their input.
- penalties: repetition penalty lowers a repeated token's probability for both positive and
  negative logits; presence/frequency are additive and monotone in the count.

`tests/test_temperature.py`
- `T = 1e-6` matches greedy's argmax.
- `T = 1e6` is within `atol` of uniform.
- entropy is non-decreasing across the configured temperature sweep.
- `T = 0` returns the one-hot argmax; `T < 0` raises `ValueError`.

`tests/test_vs_reference.py`
- `apply_top_k` == `transformers.TopKLogitsWarper`, `apply_top_p` == `TopPLogitsWarper`,
  `apply_temperature` == `TemperatureLogitsWarper`, `apply_repetition_penalty` ==
  `RepetitionPenaltyLogitsProcessor`, on random logits and on recorded fixtures (exact).
- greedy `generate` == `model.generate(do_sample=False, num_beams=1)` token for token.
- beam `generate` == `model.generate(num_beams=B, do_sample=False, length_penalty=1.0,
  early_stopping=False)` on short sequences.
- do **not** assert token-level equality for sampled generation — HF's `generate` uses the global
  torch RNG and we thread a `Generator`, so the streams differ by construction. Compare the warped
  distributions (exact) and, if needed, many draws by total-variation distance.

`tests/test_repetition.py`
- greedy's mean distinct-2 on the open-ended prompts is at least 0.2 below nucleus, and its
  longest repeat more than twice as long (fixed seeds); every greedy run contains a cycle.
- adding a repetition penalty raises greedy's distinct-2; beam (B=4) is below nucleus.

Test hygiene: fixed seeds everywhere; `tests/fixtures/*.npz` for recorded logits so the suite runs
with `HF_HUB_OFFLINE=1`; markers `slow` and `network` are registered (`--strict-markers`), and
`network` tests are skipped when `HF_HUB_OFFLINE=1`. Ollama is never touched by tests. CI may
download `sshleifer/tiny-gpt2`. `tests/` is not a package: share helpers through fixtures.

---

## 9. Notebook conventions

- Notebooks **narrate and orchestrate**; all logic is imported from `src`. No copy-pasted algorithms.
- Must run top-to-bottom from a fresh kernel with no hidden state, under 5 minutes, and be
  deterministic given the seed in `config/settings.yaml`.
- Open with: the question, the claim, and what the reader will be able to do at the end.
  Close with a short "what this bought you" cell.
- Hold the model fixed. Every comparison varies **only** the decoder, and says so in prose.
- Print the seed and the settings used for any generated text.
- `make notebooks` executes in place (nbconvert writes outputs back), so **committed notebooks
  carry their outputs**. Re-run it before committing a notebook change. The `.ipynb` files are
  the source (no generator script in the repo): edit them, then execute.
- **Every visual is saved as PNG** (`save_figure(fig, "NN_slug")` -> `outputs/figures/`, 150 dpi)
  and **every number and text sample as JSON** (`save_json("NN_slug", data)` ->
  `outputs/results/`, with the seed). A figure and the JSON behind it share the slug.
  No figure is displayed without being saved.
- **Image budget: at most 19 PNGs across all notebooks** (currently 17). Adding one means
  removing or merging another; print numbers or save JSON instead of plotting marginal results.
- **Open every new or changed PNG and check it reads** before committing: no overlapping labels
  or legends, no clipped markers, labels next to their own points (crowded scatters use
  `numbered=True`), integer ticks for counts, units on axes. Shared plot code changes can
  regress figures in other notebooks: re-check those too.
- Every figure is produced by a function in `src/visualisation.py` — no inline plotting code.
  Those functions return a `Figure` and never call `plt.show()` or `fig.write_image()`.
  `plotly` is for the Streamlit app only.
- Prose must match the executed output. If a number changes on re-run, re-read the markdown.

Per-notebook obligations:

| NB | Must contain |
|---|---|
| 01 | one model call, the full next-token distribution displayed, the model/decoder boundary named |
| 02 | greedy collapsing into a loop on an open prompt, distinct-n measuring it, "the fix is the decoder" |
| 03 | temperature as one line, a sweep, entropy-vs-T plot, loop → fluent → gibberish shown on one distribution |
| 04 | top-k and top-p from scratch, fixed-k failing on peaked *and* flat distributions, nucleus adapting, reference match |
| 05 | beam search from scratch, winning on single-answer prompts, losing on open-ended, the trade-off named as explore-exploit |
| 06 | the decision table, the penalties, the diversity-quality harness, working code the reader can lift |

---

## 10. Streamlit app

Four tabs, exactly the PRD's: **distribution viewer** (prompt → distribution, reshaped live by
T / k / p), **decoder race** (greedy vs temperature vs nucleus vs beam side by side with repetition
and diversity metrics), **temperature sweep** (slider from 0 to high, output and entropy together),
**task matcher** (single-answer vs open-ended → which decoder wins and why).

- `@st.cache_resource` for the model, `@st.cache_data` for computed distributions and sweeps.
- Must start and be usable on 2 CPUs / 8 GB. Default to the char LM; make the HF model opt-in.
- No logic in the app file — it calls `src/`.
- The app must work with Ollama absent.

---

## 11. Honesty guardrails

The project's credibility rests on the claim being *demonstrated*, not asserted. So:

- Never vary the model and the decoder in the same comparison.
- Report seeds, settings, prompt sets, and sample counts for every claim.
- Back subjective words ("fluent", "bland", "gibberish") with a metric: entropy, distinct-n,
  rep-n, or the quality-diversity frontier. "Looks better" is not a result.
- Tiny models produce weak text. Say so where it matters, and keep claims about *how decoders
  reshape a fixed distribution* — those hold at any model scale.
- If a result does not reproduce, write that down rather than tuning the seed until it does.
- Cite the literature in the notebooks where it is doing work: Holtzman et al. 2020 (nucleus,
  the greedy loop), Fan et al. 2018 (top-k), Keskar et al. 2019 (repetition penalty),
  Freitag & Al-Onaizan 2017 (beam), Meister et al. 2023 (typical), Welleck et al. 2020 (repetition).

---

## 12. Terminology (use consistently)

**logits** raw scores from the model · **probs** post-softmax · **logprobs** `log(probs)` ·
**warper / processor** a pure logits → logits transform · **decoder / sampler** the procedure that
picks the next token · **nucleus** = top-p · **greedy** = argmax = `T → 0` ·
**open-ended** prompts (diversity wanted) vs **single-answer** prompts (determinism wanted).
Say "token", not "word" — the vocabulary is subword.

---

## 13. Working agreements

- `make ci` passes before any commit. No exceptions, no "will fix in the next commit".
- Small commits, imperative subject (`add top-p warper`, `fix beam length normalisation`).
- Never commit secrets, model weights, HF caches, `.venv`, or anything from `.llm/`.
- `.llm/`, `.agents/`, `agent/`, `.claude/skills/` are gitignored — this file is tracked.
- Don't touch `uv.lock` by hand; change deps via `uv add` / `uv remove`.
- Prefer deleting code to adding flags. This is a teaching repo: the shortest readable
  implementation beats the clever one, and a 10-line decoder that the reader can follow is the
  whole point.
- Comments explain *why* and name the formula; they do not restate the line below.
- **README numbers trace to `outputs/results/*.json`.** When notebooks are re-run and a number
  moves, update the README "What it shows" table and CLAUDE.md §2 counts (tests, PNGs, JSONs).
  Run the README code snippet after any API change.

### Token rules
- **Caveman mode: ON.** Terse, code-first replies. Lead with the diff or the command, not a preamble.
  No restating the plan, no summarising what the code obviously does.
  (Optional `.claude/skills/caveman.md` is gitignored and may be absent — the rule still applies.)
- `/compact` at notebook boundaries.
- Read the PRD section you need, not the whole file, once the scaffold exists.
