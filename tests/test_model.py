"""The char model, the HF wrapper, and the optional Ollama hooks."""

from __future__ import annotations

import pytest
import torch

from src.datasets import char_prompts
from src.decoders import Decoder
from src.model import HFLanguageModel, load_model, ollama_available, train_char_lm, visible


def test_char_lm_encode_decode_roundtrip(char_lm) -> None:
    text = "the sea was calm."
    ids = char_lm.encode(text)
    assert ids.shape == (1, len(text))
    assert ids.dtype == torch.int64
    assert char_lm.decode(ids) == text
    assert char_lm.decode(ids[0].tolist()) == text


def test_char_lm_lowercases_and_maps_unknown_characters_to_space(char_lm) -> None:
    assert char_lm.decode(char_lm.encode("The SEA!")) == "the sea "


def test_char_lm_logits_shape_and_padding(char_lm) -> None:
    ids = char_lm.encode("ab")
    logits = char_lm.next_token_logits(ids)
    assert logits.shape == (1, char_lm.vocab_size)
    assert char_lm.window(ids).shape == (1, char_lm.context)
    batch = torch.cat([char_lm.encode("the sea "), char_lm.encode("the fox ")])
    assert char_lm.next_token_logits(batch).shape == (2, char_lm.vocab_size)
    assert char_lm.eos_token_id is None
    assert char_lm.name == "char-lm"


def test_char_lm_token_labels_show_whitespace(char_lm) -> None:
    ids = [char_lm.stoi[" "], char_lm.stoi["\n"], char_lm.stoi["a"]]
    assert char_lm.token_labels(ids) == ["␣", "↵", "a"]
    assert visible("a b\n") == "a␣b↵"


def test_char_lm_memorised_every_fact(char_lm) -> None:
    for prompt in char_prompts("single"):
        ids = char_lm.encode(prompt.text)
        out = Decoder.greedy().generate(char_lm, ids, 12)
        assert prompt.is_correct(char_lm.decode(out[0, ids.size(1) :])), prompt.text


def test_training_is_seeded_and_leaves_the_global_rng_alone() -> None:
    torch.manual_seed(123)
    expected_next = torch.rand(1)
    torch.manual_seed(123)
    a = train_char_lm("abcabcabd " * 20, steps=30, hidden_dim=16, seed=5)
    assert torch.equal(torch.rand(1), expected_next)
    b = train_char_lm("abcabcabd " * 20, steps=30, hidden_dim=16, seed=5)
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(pa, pb)
    assert len(a.train_loss) == 30
    assert a.train_loss[-1] < a.train_loss[0]


def test_char_lm_rejects_duplicate_alphabet() -> None:
    from src.model import CharLM

    with pytest.raises(ValueError):
        CharLM("aab", context=4, embed_dim=2, hidden_dim=4)


def test_load_model_caches_and_validates() -> None:
    assert load_model("char") is load_model("char")
    with pytest.raises(ValueError):
        load_model("gpt-17")  # type: ignore[arg-type]


def test_hf_wrapper_without_tokenizer(tiny_gpt2: HFLanguageModel) -> None:
    ids = torch.tensor([[5, 6, 7]])
    assert tiny_gpt2.next_token_logits(ids).shape == (1, 96)
    assert tiny_gpt2.vocab_size == 96
    assert tiny_gpt2.eos_token_id == 1
    assert tiny_gpt2.token_labels([3, 4]) == ["<3>", "<4>"]
    with pytest.raises(ValueError):
        tiny_gpt2.encode("hello")
    with pytest.raises(ValueError):
        tiny_gpt2.decode([1, 2])


def test_ollama_absent_is_reported_not_raised() -> None:
    assert ollama_available("http://127.0.0.1:9", timeout=0.2) is False
