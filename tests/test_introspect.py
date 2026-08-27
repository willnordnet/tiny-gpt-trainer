"""Plumbing tests for web/introspect.py, per DESIGN.md section 6.1.

The focus is the vocabulary guard. A checkpoint stores weights and a
vocab_size but not the vocabulary itself, so its predicted ids only mean
something when decoded with the tokenizer it was trained on. Training a new
BPE vocab overwrites vocab.json in place, and because the ids stay in range,
nothing crashes -- the panels just relabel every token. That is a silent
wrong answer, which is worse than a crash, so it gets a test.
"""

import json

import mlx.core as mx
import pytest

from tinygpt.config import ModelConfig
from tinygpt.model import TinyGPT
from tinygpt.tokenizer.tokenizer import BPETokenizer
from tinygpt.train import save_checkpoint
from web import introspect


@pytest.fixture
def tiny_cfg() -> ModelConfig:
    # vocab_size 258 = the 256 byte tokens plus two merges, so a real
    # BPETokenizer can be built to match it exactly.
    return ModelConfig(vocab_size=258, n_layers=2, d_model=32, n_heads=4, context_len=16)


@pytest.fixture
def matching_tokenizer() -> BPETokenizer:
    return BPETokenizer(merges=[(116, 104), (256, 101)])


def test_the_fixtures_actually_agree(tiny_cfg, matching_tokenizer):
    """Guards the guard: if these drifted, the mismatch test below would pass
    for the wrong reason."""
    assert matching_tokenizer.vocab_size == tiny_cfg.vocab_size


def test_a_matching_tokenizer_is_accepted(tiny_cfg, matching_tokenizer, monkeypatch):
    model = TinyGPT(tiny_cfg)
    monkeypatch.setattr(introspect, "load_tokenizer", lambda _: matching_tokenizer)
    assert introspect._checked_tokenizer(model, "ckpt", "vocab.json") is matching_tokenizer


def test_a_mismatched_tokenizer_is_refused_rather_than_silently_wrong(
    tiny_cfg, monkeypatch
):
    model = TinyGPT(tiny_cfg)
    wrong = BPETokenizer(merges=[(116, 104)])  # vocab_size 257, not 258
    monkeypatch.setattr(introspect, "load_tokenizer", lambda _: wrong)

    with pytest.raises(introspect.VocabMismatch) as raised:
        introspect._checked_tokenizer(model, "checkpoints/x.safetensors", "vocab.json")

    # The message has to name both numbers, or it is not actionable.
    assert "257" in str(raised.value)
    assert "258" in str(raised.value)


# --- listing checkpoints ---------------------------------------------------


def test_list_checkpoints_reads_step_and_val_loss_from_metadata(
    tiny_cfg, tmp_path, monkeypatch
):
    """No sidecar index: a checkpoint carries its own preset, step and val
    loss, so the list stays correct for files this server never saw written."""
    monkeypatch.setattr(introspect, "REPO_ROOT", tmp_path)
    out = tmp_path / "checkpoints"
    out.mkdir()

    model = TinyGPT(tiny_cfg)
    save_checkpoint(out / "t-step10.safetensors", model, "tiny", 10, 1.5)
    save_checkpoint(out / "t-step20.safetensors", model, "tiny", 20, None)

    found = introspect.list_checkpoints("checkpoints")

    assert [entry["step"] for entry in found] == [20, 10]  # newest first
    assert found[1]["val_loss"] == pytest.approx(1.5)
    assert found[0]["val_loss"] is None  # an absent val loss must not crash
    assert all(entry["preset"] == "tiny" for entry in found)


def test_list_checkpoints_skips_an_unreadable_file(tiny_cfg, tmp_path, monkeypatch):
    """A checkpoint half-written by a run in progress must not 500 the page."""
    monkeypatch.setattr(introspect, "REPO_ROOT", tmp_path)
    out = tmp_path / "checkpoints"
    out.mkdir()
    save_checkpoint(out / "t-step10.safetensors", TinyGPT(tiny_cfg), "tiny", 10, 1.0)
    (out / "t-step99.safetensors").write_bytes(b"not a safetensors file")

    found = introspect.list_checkpoints("checkpoints")
    assert [entry["step"] for entry in found] == [10]


def test_no_checkpoint_directory_is_an_empty_list_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(introspect, "REPO_ROOT", tmp_path)
    assert introspect.list_checkpoints("checkpoints") == []


# --- the distribution ------------------------------------------------------


def test_next_token_distribution_reports_both_before_and_after_the_knobs(
    tiny_cfg, matching_tokenizer, tmp_path, monkeypatch
):
    """Showing only the post-knob distribution would hide what the knobs did,
    which is the entire point of the panel."""
    monkeypatch.setattr(introspect, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(introspect, "load_tokenizer", lambda _: matching_tokenizer)
    introspect._model_cache.clear()

    out = tmp_path / "checkpoints"
    out.mkdir()
    save_checkpoint(out / "t-step5.safetensors", TinyGPT(tiny_cfg), "tiny", 5, 1.0)

    result = introspect.next_token_distribution(
        "checkpoints/t-step5.safetensors", "the", temperature=1.0, top_k=3, top_n=8
    )

    assert len(result["candidates"]) == 8
    # Ranked by the raw distribution, so bars keep a stable order as sliders move.
    raw = [candidate["prob"] for candidate in result["candidates"]]
    assert raw == sorted(raw, reverse=True)
    # top_k=3 must leave exactly three survivors among the top-ranked eight.
    survivors = [c for c in result["candidates"] if not c["eliminated"]]
    assert len(survivors) == 3
    assert all(c["prob_after"] == 0.0 for c in result["candidates"] if c["eliminated"])
    # Truncating the tail always lowers entropy.
    assert result["entropy"] < result["entropy_raw"]


def test_attention_grid_is_strictly_causal(
    tiny_cfg, matching_tokenizer, tmp_path, monkeypatch
):
    monkeypatch.setattr(introspect, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(introspect, "load_tokenizer", lambda _: matching_tokenizer)
    introspect._model_cache.clear()

    out = tmp_path / "checkpoints"
    out.mkdir()
    save_checkpoint(out / "t-step5.safetensors", TinyGPT(tiny_cfg), "tiny", 5, 1.0)

    grid = introspect.attention_grid(
        "checkpoints/t-step5.safetensors", "the theme", layer=1, head=2
    )

    assert grid["n_layers"] == tiny_cfg.n_layers
    assert grid["n_heads"] == tiny_cfg.n_heads
    assert len(grid["weights"]) == len(grid["tokens"])
    for i, row in enumerate(grid["weights"]):
        assert sum(row) == pytest.approx(1.0, abs=1e-5)
        assert all(weight == 0.0 for weight in row[i + 1:]), f"row {i} sees the future"


def test_an_out_of_range_layer_or_head_is_clamped_not_crashed(
    tiny_cfg, matching_tokenizer, tmp_path, monkeypatch
):
    """The UI rebuilds its pickers from the response, so the first request
    after switching to a smaller model can legitimately ask for layer 5 of 2."""
    monkeypatch.setattr(introspect, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(introspect, "load_tokenizer", lambda _: matching_tokenizer)
    introspect._model_cache.clear()

    out = tmp_path / "checkpoints"
    out.mkdir()
    save_checkpoint(out / "t-step5.safetensors", TinyGPT(tiny_cfg), "tiny", 5, 1.0)

    grid = introspect.attention_grid(
        "checkpoints/t-step5.safetensors", "the", layer=99, head=99
    )
    assert grid["layer"] == tiny_cfg.n_layers - 1
    assert grid["head"] == tiny_cfg.n_heads - 1


# --- generation ------------------------------------------------------------


def test_generate_completion_streams_exactly_the_continuation(
    tiny_cfg, matching_tokenizer, tmp_path, monkeypatch
):
    """The sink must receive the continuation and nothing else: the page has
    the prompt on screen already, and echoing it back would render it twice."""
    monkeypatch.setattr(introspect, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(introspect, "load_tokenizer", lambda _: matching_tokenizer)
    introspect._model_cache.clear()

    out = tmp_path / "checkpoints"
    out.mkdir()
    save_checkpoint(out / "t-step5.safetensors", TinyGPT(tiny_cfg), "tiny", 5, 1.0)

    deltas: list[str] = []
    result = introspect.generate_completion(
        "checkpoints/t-step5.safetensors", "the", max_tokens=12,
        temperature=0.0, on_text=deltas.append,
    )

    # The robust statement of "the sink never echoes the prompt": what the sink
    # received is exactly what generate_text appended to it. A prefix check
    # would be fragile -- a greedy untrained model happily emits the prompt's
    # own words back as its first continuation.
    assert "".join(deltas) == result["continuation"]
    assert result["step"] == 5
    assert result["prompt_tokens"] == len(matching_tokenizer.encode("the"))
    assert result["context_len"] == tiny_cfg.context_len


def test_generate_completion_works_without_a_sink(
    tiny_cfg, matching_tokenizer, tmp_path, monkeypatch
):
    monkeypatch.setattr(introspect, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(introspect, "load_tokenizer", lambda _: matching_tokenizer)
    introspect._model_cache.clear()

    out = tmp_path / "checkpoints"
    out.mkdir()
    save_checkpoint(out / "t-step5.safetensors", TinyGPT(tiny_cfg), "tiny", 5, 1.0)

    result = introspect.generate_completion(
        "checkpoints/t-step5.safetensors", "the", max_tokens=6, temperature=0.0
    )
    assert result["continuation"]


def test_generate_completion_flags_a_prompt_that_will_overflow_the_context(
    tiny_cfg, matching_tokenizer, tmp_path, monkeypatch
):
    """generate() re-slices ids[-context_len:] every step with no signal, so a
    long prompt loses its own beginning partway through. Say so rather than
    letting it happen quietly."""
    monkeypatch.setattr(introspect, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(introspect, "load_tokenizer", lambda _: matching_tokenizer)
    introspect._model_cache.clear()

    out = tmp_path / "checkpoints"
    out.mkdir()
    save_checkpoint(out / "t-step5.safetensors", TinyGPT(tiny_cfg), "tiny", 5, 1.0)

    short = introspect.generate_completion(
        "checkpoints/t-step5.safetensors", "the", max_tokens=4, temperature=0.0
    )
    assert short["truncated"] is False

    long_prompt = "the " * tiny_cfg.context_len
    overflowing = introspect.generate_completion(
        "checkpoints/t-step5.safetensors", long_prompt, max_tokens=4, temperature=0.0
    )
    assert overflowing["truncated"] is True


def test_generate_completion_refuses_a_mismatched_vocabulary(
    tiny_cfg, tmp_path, monkeypatch
):
    monkeypatch.setattr(introspect, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        introspect, "load_tokenizer", lambda _: BPETokenizer(merges=[(116, 104)])
    )
    introspect._model_cache.clear()

    out = tmp_path / "checkpoints"
    out.mkdir()
    save_checkpoint(out / "t-step5.safetensors", TinyGPT(tiny_cfg), "tiny", 5, 1.0)

    with pytest.raises(introspect.VocabMismatch):
        introspect.generate_completion("checkpoints/t-step5.safetensors", "the")
