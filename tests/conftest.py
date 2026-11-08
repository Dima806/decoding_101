"""Shared fixtures. Everything here is offline: no test downloads a model unless marked network."""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

from collections.abc import Callable, Sequence  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from torch import Tensor  # noqa: E402

from src.model import CharLM, HFLanguageModel, default_char_lm  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        skip = pytest.mark.skip(reason="HF_HUB_OFFLINE=1")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(scope="session")
def char_lm() -> CharLM:
    """The configured char model, trained once for the whole session."""
    return default_char_lm()


def build_tiny_gpt2(seed: int, vocab_size: int = 96, eos_token_id: int = 1) -> HFLanguageModel:
    """A randomly initialised 2-layer GPT-2, built from a config: no download, fully offline."""
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=128,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=0,
        eos_token_id=eos_token_id,
        pad_token_id=eos_token_id,
        initializer_range=0.2,
    )
    return HFLanguageModel(GPT2LMHeadModel(config), name=f"tiny-gpt2-{seed}")


@pytest.fixture(scope="session")
def tiny_gpt2() -> HFLanguageModel:
    return build_tiny_gpt2(seed=0)


@pytest.fixture(scope="session")
def make_tiny_gpt2() -> Callable[..., HFLanguageModel]:
    """Factory fixture: ``make_tiny_gpt2(seed, vocab_size=96)``."""
    return build_tiny_gpt2


@pytest.fixture
def random_logits() -> Tensor:
    """``(4, 200)`` heavy-ish logits, seeded."""
    return torch.randn(4, 200, generator=torch.Generator().manual_seed(0)) * 3


@pytest.fixture(scope="session")
def recorded_logits() -> list[tuple[Tensor, Tensor]]:
    """Real distilgpt2 ``(ids, logits)`` pairs from ``tests/fixtures/distilgpt2_logits.npz``."""
    data = np.load(FIXTURES / "distilgpt2_logits.npz")
    n = len([k for k in data.files if k.startswith("ids_")])
    return [
        (torch.from_numpy(data[f"ids_{i}"]), torch.from_numpy(data[f"logits_{i}"]))
        for i in range(n)
    ]


class TableLM:
    """A hand-written model: next-token logits depend only on the last token.

    Built to make greedy provably myopic. From the start token 0:

    * token 1 is the greedy pick (p=0.6) but every continuation after it is flat;
    * token 2 is less likely (p=0.4) but is followed by a near-certain token.

    So greedy takes 1, and beam search (which scores whole sequences) takes 2.
    """

    vocab = 6

    def __init__(self) -> None:
        log = torch.log
        uniform = log(torch.full((self.vocab,), 1.0 / self.vocab))
        first = log(torch.tensor([1e-6, 0.6, 0.4, 1e-6, 1e-6, 1e-6]))
        peaked = log(torch.tensor([1e-6, 1e-6, 1e-6, 0.97, 0.01, 0.02]))
        self.table = {0: first, 1: uniform, 2: peaked}
        self.default = uniform

    @property
    def name(self) -> str:
        return "table-lm"

    @property
    def vocab_size(self) -> int:
        return self.vocab

    @property
    def eos_token_id(self) -> int | None:
        return None

    def next_token_logits(self, input_ids: Tensor) -> Tensor:
        rows = [self.table.get(int(seq[-1]), self.default) for seq in input_ids]
        return torch.stack(rows)

    def encode(self, text: str) -> Tensor:
        return torch.tensor([[int(ch) for ch in text]], dtype=torch.int64)

    def decode(self, ids: Tensor | Sequence[int]) -> str:
        flat = ids.flatten().tolist() if isinstance(ids, Tensor) else list(ids)
        return "".join(str(i) for i in flat)

    def token_labels(self, ids: Sequence[int]) -> list[str]:
        return [str(i) for i in ids]


@pytest.fixture
def table_lm() -> TableLM:
    return TableLM()
