"""Next-token logits from a model, behind one small interface.

Two sources, so the project is both transparent and realistic:

* :class:`src.model.char_lm.CharLM` - from scratch, ~30-token vocabulary, fully inspectable;
* :class:`HFLanguageModel` - a small real ``transformers`` model (``distilgpt2`` by default)
  for genuine English. ``transformers`` is imported lazily, only when one is built.

An optional Ollama server can also be driven with the same knobs (:func:`ollama_generate`), but
it only returns text, never logits, so it is a black-box comparison and never required.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import httpx
import torch
from torch import Tensor

from src.config import get_settings
from src.model.char_lm import VISIBLE, default_char_lm

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

ModelSource = Literal["char", "hf"]


@runtime_checkable
class LanguageModel(Protocol):
    """What a decoder needs from a model. Nothing here knows about sampling."""

    @property
    def name(self) -> str: ...

    @property
    def vocab_size(self) -> int: ...

    @property
    def eos_token_id(self) -> int | None: ...

    def next_token_logits(self, input_ids: Tensor) -> Tensor:
        """``(batch, seq)`` int64 ids -> ``(batch, vocab)`` float32 logits."""
        ...

    def encode(self, text: str) -> Tensor:
        """Text -> ``(1, seq)`` int64 ids."""
        ...

    def decode(self, ids: Tensor | Sequence[int]) -> str:
        """Ids -> text."""
        ...

    def token_labels(self, ids: Sequence[int]) -> list[str]:
        """Human-readable label per token id, whitespace made visible."""
        ...


def visible(text: str) -> str:
    """Make spaces and newlines visible in a token label."""
    return "".join(VISIBLE.get(ch, ch) for ch in text)


class HFLanguageModel:
    """A ``transformers`` causal LM wrapped in the :class:`LanguageModel` interface.

    Every call is a full forward pass over the whole sequence (no KV cache), so the logits are
    exactly what ``model(input_ids)`` returns and the decoding loop stays transparent.

    Args:
        model: a causal LM, e.g. from ``AutoModelForCausalLM.from_pretrained``.
        tokenizer: its tokenizer, or ``None`` to work on raw ids only (tests).
        name: display name.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase | None = None,
        name: str = "hf",
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self._name = name

    @classmethod
    def from_pretrained(cls, model_id: str) -> HFLanguageModel:
        """Download (or load from cache) ``model_id`` and its tokenizer, CPU, float32."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
        return cls(model, tokenizer, name=model_id)

    @property
    def name(self) -> str:
        return self._name

    @property
    def vocab_size(self) -> int:
        return int(self.model.config.vocab_size)

    @property
    def eos_token_id(self) -> int | None:
        eos: Any = self.model.generation_config.eos_token_id
        if isinstance(eos, list):
            return int(eos[0]) if eos else None
        return None if eos is None else int(eos)

    def next_token_logits(self, input_ids: Tensor) -> Tensor:
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False
            )
        return out.logits[:, -1, :].float()

    def _require_tokenizer(self) -> PreTrainedTokenizerBase:
        if self.tokenizer is None:
            raise ValueError(f"{self.name} has no tokenizer; work with token ids directly")
        return self.tokenizer

    def encode(self, text: str) -> Tensor:
        ids = self._require_tokenizer()(text, return_tensors="pt")["input_ids"]
        return torch.as_tensor(ids, dtype=torch.int64)

    def decode(self, ids: Tensor | Sequence[int]) -> str:
        flat = ids.flatten().tolist() if isinstance(ids, Tensor) else list(ids)
        return str(self._require_tokenizer().decode(flat))

    def token_labels(self, ids: Sequence[int]) -> list[str]:
        if self.tokenizer is None:
            return [f"<{i}>" for i in ids]
        return [self._label(i) for i in ids]

    def _label(self, token_id: int) -> str:
        tokenizer = self._require_tokenizer()
        text = str(tokenizer.decode([token_id]))
        if "\ufffd" in text:
            # A byte-level BPE piece of a multi-byte character (e.g. half of a curly quote):
            # it has no text of its own, so show the raw token string instead.
            return f"{tokenizer.convert_ids_to_tokens(token_id)} (byte piece)"
        return visible(text)


@cache
def load_model(source: ModelSource = "char") -> LanguageModel:
    """The configured char model or small real model, built once per process."""
    if source == "char":
        return default_char_lm()
    if source == "hf":
        return HFLanguageModel.from_pretrained(get_settings().hf.model_id)
    raise ValueError(f"unknown model source {source!r}; expected 'char' or 'hf'")


def next_token_logits(model: LanguageModel, prompt: str) -> Tensor:
    """Run the model once on ``prompt``: its complete, fixed opinion. ``(1, vocab)`` logits."""
    return model.next_token_logits(model.encode(prompt))


# --- optional Ollama server -------------------------------------------------------------------


def ollama_available(url: str | None = None, timeout: float = 1.0) -> bool:
    """True if an Ollama server answers at ``url``. Never raises."""
    base = url or get_settings().ollama.url
    try:
        return httpx.get(f"{base}/api/tags", timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


def ollama_generate(
    prompt: str,
    *,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repeat_penalty: float = 1.0,
    seed: int | None = None,
    num_predict: int = 60,
    model: str | None = None,
    url: str | None = None,
) -> str:
    """Generate text from a local Ollama model with the same knobs this project builds.

    Ollama runs its own sampler server-side and returns text only, so this is for comparing
    behaviour, not for inspecting distributions. Check :func:`ollama_available` first.
    """
    cfg = get_settings().ollama
    options: dict[str, float | int] = {
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repeat_penalty": repeat_penalty,
        "num_predict": num_predict,
    }
    if seed is not None:
        options["seed"] = seed
    response = httpx.post(
        f"{url or cfg.url}/api/generate",
        json={"model": model or cfg.model, "prompt": prompt, "stream": False, "options": options},
        timeout=cfg.timeout_s,
    )
    response.raise_for_status()
    return str(response.json()["response"])
