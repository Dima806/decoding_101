"""A tiny from-scratch character language model with fully inspectable logits.

An MLP over the last ``context`` characters (Bengio et al., 2003), trained in seconds on CPU on
the built-in corpus. The vocabulary is ~30 characters, so the *entire* next-token distribution
fits on one bar chart: the decoders can be watched operating on probabilities you can read.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache

import torch
import torch.nn.functional as F  # noqa: N812 - conventional alias
from torch import Tensor, nn

from src.config import get_settings
from src.datasets.corpus import TINY_CORPUS

VISIBLE = {" ": "␣", "\n": "↵"}


class CharLM(nn.Module):
    """Character-level MLP language model.

    Args:
        alphabet: the vocabulary, one character per token id.
        context: number of previous characters the model sees.
        embed_dim: size of each character embedding.
        hidden_dim: width of the hidden layer.
    """

    def __init__(self, alphabet: str, context: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if len(set(alphabet)) != len(alphabet):
            raise ValueError("alphabet must not contain duplicate characters")
        self.alphabet = alphabet
        self.context = context
        self.stoi = {ch: i for i, ch in enumerate(alphabet)}
        self.pad_id = self.stoi.get(" ", 0)
        self.embed = nn.Embedding(len(alphabet), embed_dim)
        self.hidden = nn.Linear(context * embed_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, len(alphabet))
        self.train_loss: list[float] = []

    def forward(self, windows: Tensor) -> Tensor:
        """``(batch, context)`` ids -> ``(batch, vocab)`` logits."""
        x = self.embed(windows).flatten(start_dim=1)
        return self.out(torch.tanh(self.hidden(x)))

    # --- LanguageModel interface ------------------------------------------------------------

    @property
    def name(self) -> str:
        return "char-lm"

    @property
    def vocab_size(self) -> int:
        return len(self.alphabet)

    @property
    def eos_token_id(self) -> int | None:
        return None

    def window(self, input_ids: Tensor) -> Tensor:
        """Last ``context`` ids of each row, left-padded with the space id if shorter."""
        missing = self.context - input_ids.size(1)
        if missing > 0:
            pad = torch.full((input_ids.size(0), missing), self.pad_id, dtype=input_ids.dtype)
            input_ids = torch.cat([pad, input_ids], dim=1)
        return input_ids[:, -self.context :]

    def next_token_logits(self, input_ids: Tensor) -> Tensor:
        """``(batch, seq)`` ids -> ``(batch, vocab)`` logits for the next character."""
        self.eval()
        with torch.no_grad():
            return self.forward(self.window(input_ids))

    def encode(self, text: str) -> Tensor:
        """Lowercase ``text`` and map it to ``(1, seq)`` ids. Unknown characters become spaces."""
        ids = [self.stoi.get(ch, self.pad_id) for ch in text.lower()]
        return torch.tensor([ids], dtype=torch.int64)

    def decode(self, ids: Tensor | Sequence[int]) -> str:
        """Ids (any shape that flattens to one sequence) -> text."""
        flat = ids.flatten().tolist() if isinstance(ids, Tensor) else list(ids)
        return "".join(self.alphabet[i] for i in flat)

    def token_labels(self, ids: Sequence[int]) -> list[str]:
        """Display labels with whitespace made visible (``␣`` for space, ``↵`` for newline)."""
        return [VISIBLE.get(self.alphabet[i], self.alphabet[i]) for i in ids]


def train_char_lm(
    corpus: str = TINY_CORPUS,
    *,
    context: int = 8,
    embed_dim: int = 24,
    hidden_dim: int = 256,
    steps: int = 1500,
    batch_size: int = 256,
    learning_rate: float = 3e-3,
    seed: int = 42,
) -> CharLM:
    """Train a :class:`CharLM` on ``corpus``. Deterministic for a given seed on a given machine.

    The caller's global torch RNG is left untouched: initialisation runs under ``fork_rng`` and
    batches are drawn from a private generator.
    """
    alphabet = "".join(sorted(set(corpus)))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = CharLM(alphabet, context, embed_dim, hidden_dim)

    data = torch.tensor([model.pad_id] * context + [model.stoi[ch] for ch in corpus])
    windows = data.unfold(0, context, 1)[:-1]
    targets = data[context:]

    generator = torch.Generator().manual_seed(seed)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=steps)
    model.train()
    for _ in range(steps):
        batch = torch.randint(len(targets), (batch_size,), generator=generator)
        loss = F.cross_entropy(model(windows[batch]), targets[batch])
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        schedule.step()
        model.train_loss.append(loss.item())
    model.eval()
    return model


@cache
def default_char_lm() -> CharLM:
    """The char model configured in ``config/settings.yaml``, trained once per process."""
    s = get_settings()
    cfg = s.char_lm
    return train_char_lm(
        context=cfg.context,
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        steps=cfg.train_steps,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        seed=s.seed,
    )
