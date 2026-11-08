"""Regenerate ``distilgpt2_logits.npz``: real next-token logits for offline reference tests.

    uv run python tests/fixtures/record_logits.py

Stores, per prompt, the prompt ids and the float32 next-token logits of distilgpt2, so the
warper tests run against a genuine LLM distribution (peaked, heavy-tailed) with no download.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.model import HFLanguageModel

PROMPTS = (
    "The capital of France is",
    "Once upon a time, in a small village by the sea,",
)
OUT = Path(__file__).with_name("distilgpt2_logits.npz")


def main() -> None:
    model = HFLanguageModel.from_pretrained("distilgpt2")
    arrays: dict[str, np.ndarray] = {}
    for i, prompt in enumerate(PROMPTS):
        ids = model.encode(prompt)
        arrays[f"ids_{i}"] = ids.numpy()
        arrays[f"logits_{i}"] = model.next_token_logits(ids).numpy().astype(np.float32)
    np.savez_compressed(OUT, **arrays)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
