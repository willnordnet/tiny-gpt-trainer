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


@pytest.fixture
def same_size_different_vocab() -> BPETokenizer:
    """vocab_size 258, like matching_tokenizer, but a different merge list.

    This is the case that matters: retraining a vocab on a new corpus targets
    the same size, so size alone cannot tell the two apart.
    """
    return BPETokenizer(merges=[(97, 98), (256, 99)])


def test_a_matching_tokenizer_is_accepted(tiny_cfg, matching_tokenizer, monkeypatch):
    model = TinyGPT(tiny_cfg)
    monkeypatch.setattr(introspect, "load_tokenizer", lambda _: matching_tokenizer)
    meta = {"tokenizer_fingerprint": matching_tokenizer.fingerprint}
    accepted = introspect._checked_tokenizer(model, meta, "ckpt", "vocab.json")
    assert accepted is matching_tokenizer


def test_a_mismatched_tokenizer_is_refused_rather_than_silently_wrong(
    tiny_cfg, monkeypatch
):
    model = TinyGPT(tiny_cfg)
    wrong = BPETokenizer(merges=[(116, 104)])  # vocab_size 257, not 258
    monkeypatch.setattr(introspect, "load_tokenizer", lambda _: wrong)

    with pytest.raises(introspect.VocabMismatch) as raised:
        introspect._checked_tokenizer(model, {}, "checkpoints/x.safetensors", "vocab.json")

    # The message has to name both numbers, or it is not actionable.
    assert "257" in str(raised.value)
    assert "258" in str(raised.value)


def test_a_same_size_but_different_vocabulary_is_refused_on_its_fingerprint(
    tiny_cfg, matching_tokenizer, same_size_different_vocab, monkeypatch
):
    """The failure vocab_size cannot see.

    Retraining BPE on a new corpus overwrites vocab.json with a vocabulary of
    the same configured size. Every id stays in range, so the old checkpoint
    decodes without error into entirely the wrong text. Only the fingerprint
    distinguishes them.
    """
    assert same_size_different_vocab.vocab_size == matching_tokenizer.vocab_size
    assert same_size_different_vocab.fingerprint != matching_tokenizer.fingerprint

    model = TinyGPT(tiny_cfg)
    monkeypatch.setattr(introspect, "load_tokenizer", lambda _: same_size_different_vocab)
    meta = {"tokenizer_fingerprint": matching_tokenizer.fingerprint}

    with pytest.raises(introspect.VocabMismatch) as raised:
        introspect._checked_tokenizer(model, meta, "checkpoints/x.safetensors", "vocab.json")

    # Both fingerprints, or there is no way to tell which vocab to go find.
    assert matching_tokenizer.fingerprint in str(raised.value)
    assert same_size_different_vocab.fingerprint in str(raised.value)


@pytest.mark.parametrize("recorded", [None, "", "   "])
def test_a_checkpoint_without_a_fingerprint_still_loads(
    tiny_cfg, matching_tokenizer, same_size_different_vocab, monkeypatch, recorded
):
    """Checkpoints predate the fingerprint field. Refusing them outright would
    be a regression on the weak check they used to get, so they fall back to
    it -- and describe_vocab_match reports them as unverifiable."""
    model = TinyGPT(tiny_cfg)
    monkeypatch.setattr(introspect, "load_tokenizer", lambda _: same_size_different_vocab)
    meta = {} if recorded is None else {"tokenizer_fingerprint": recorded}
    accepted = introspect._checked_tokenizer(model, meta, "ckpt", "vocab.json")
    assert accepted is same_size_different_vocab


def test_save_checkpoint_records_the_tokenizer_fingerprint(
    tiny_cfg, matching_tokenizer, tmp_path
):
    """Without this the guard above has nothing to compare against."""
    path = tmp_path / "t-step1.safetensors"
    save_checkpoint(path, TinyGPT(tiny_cfg), "tiny", 1, 1.5,
                    tokenizer_fingerprint=matching_tokenizer.fingerprint)
    _, metadata = mx.load(str(path), return_metadata=True)
    assert metadata["tokenizer_fingerprint"] == matching_tokenizer.fingerprint

    # Always present, so a reader can distinguish "not recorded" from "old file".
    plain = tmp_path / "t-step2.safetensors"
    save_checkpoint(plain, TinyGPT(tiny_cfg), "tiny", 2, None)
    _, plain_meta = mx.load(str(plain), return_metadata=True)
    assert plain_meta["tokenizer_fingerprint"] == ""


@pytest.mark.parametrize(
    "recorded_from, expected",
    [("matching", "verified"), ("other", "mismatched"), (None, "unverifiable")],
)
def test_describe_vocab_match_names_the_three_states(
    tiny_cfg, matching_tokenizer, same_size_different_vocab, tmp_path, monkeypatch,
    recorded_from, expected,
):
    """The viewer needs to tell "fine", "cannot tell" and "wrong" apart, which
    a raise-or-not guard cannot express on its own."""
    monkeypatch.setattr(introspect, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(introspect, "load_tokenizer", lambda _: matching_tokenizer)

    fingerprints = {
        "matching": matching_tokenizer.fingerprint,
        "other": same_size_different_vocab.fingerprint,
        None: None,
    }
    path = tmp_path / "t-step1.safetensors"
    save_checkpoint(path, TinyGPT(tiny_cfg), "tiny", 1, 1.5,
                    tokenizer_fingerprint=fingerprints[recorded_from])

    assert introspect.describe_vocab_match(str(path))["status"] == expected


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


def test_an_embedded_vocabulary_is_used_in_preference_to_vocab_json(
    tiny_cfg, matching_tokenizer, same_size_different_vocab, tmp_path, monkeypatch
):
    """The point of embedding: a checkpoint decodes correctly on its own, even
    when the vocab.json sitting next to it is a different vocabulary of the
    same size."""
    monkeypatch.setattr(introspect, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(introspect, "load_tokenizer", lambda _: same_size_different_vocab)

    path = tmp_path / "t-step1.safetensors"
    save_checkpoint(path, TinyGPT(tiny_cfg), "tiny", 1, 1.5, tokenizer=matching_tokenizer)
    _, metadata = mx.load(str(path), return_metadata=True)

    resolved = introspect._resolve_tokenizer(metadata, None)
    assert resolved.merges == matching_tokenizer.merges


def test_an_explicit_vocab_path_still_wins_over_the_embedded_one(
    tiny_cfg, matching_tokenizer, same_size_different_vocab, tmp_path, monkeypatch
):
    """Naming a vocabulary is an instruction. Silently overriding it would make
    "read this checkpoint against that vocab" impossible to ask for -- and that
    is the one way to read a checkpoint whose embedded copy is wrong."""
    monkeypatch.setattr(introspect, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(introspect, "load_tokenizer", lambda _: same_size_different_vocab)

    path = tmp_path / "t-step1.safetensors"
    save_checkpoint(path, TinyGPT(tiny_cfg), "tiny", 1, 1.5, tokenizer=matching_tokenizer)
    _, metadata = mx.load(str(path), return_metadata=True)

    resolved = introspect._resolve_tokenizer(metadata, "somewhere/else.json")
    assert resolved.merges == same_size_different_vocab.merges


def test_a_checkpoint_carrying_its_own_vocabulary_needs_no_vocab_json(
    tiny_cfg, matching_tokenizer, tmp_path, monkeypatch
):
    """sample.py and the panels must work from the checkpoint alone, which is
    the whole reason for paying the 39 KB."""
    monkeypatch.setattr(introspect, "REPO_ROOT", tmp_path)

    def no_vocab_file(_):
        raise FileNotFoundError("vocab.json")

    monkeypatch.setattr(introspect, "load_tokenizer", no_vocab_file)

    path = tmp_path / "t-step1.safetensors"
    save_checkpoint(path, TinyGPT(tiny_cfg), "tiny", 1, 1.5, tokenizer=matching_tokenizer)
    _, metadata = mx.load(str(path), return_metadata=True)

    assert introspect._resolve_tokenizer(metadata, None).merges == matching_tokenizer.merges


def test_list_checkpoints_flags_a_checkpoint_from_another_vocabulary(
    tiny_cfg, matching_tokenizer, same_size_different_vocab, tmp_path, monkeypatch
):
    """The list is where a stale checkpoint should become visible. Waiting for
    a panel to refuse it means picking it first and wondering why the output is
    gibberish."""
    monkeypatch.setattr(introspect, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(introspect, "load_tokenizer", lambda _: matching_tokenizer)
    out = tmp_path / "checkpoints"
    out.mkdir()

    model = TinyGPT(tiny_cfg)
    save_checkpoint(out / "t-step10.safetensors", model, "tiny", 10, 1.5,
                    tokenizer_fingerprint=matching_tokenizer.fingerprint)
    save_checkpoint(out / "t-step20.safetensors", model, "tiny", 20, 1.5,
                    tokenizer_fingerprint=same_size_different_vocab.fingerprint)
    save_checkpoint(out / "t-step30.safetensors", model, "tiny", 30, 1.5)

    status = {e["step"]: e["vocab"] for e in introspect.list_checkpoints("checkpoints")}
    assert status == {10: "verified", 20: "mismatched", 30: "unverifiable"}


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
