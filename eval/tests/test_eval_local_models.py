"""Unit tests for the local-model wrappers, the corpus-embedding cache and the hardware probe.

Neither torch nor sentence-transformers is needed. Where a test wants them
"installed" it puts a fake module in ``sys.modules``; where it wants them
absent it puts ``None`` there, which makes the import fail whatever the
environment really has.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from eval.retrieval import config, local_models
from eval.retrieval.interfaces import Embedder, PairScorer
from eval.retrieval.local_models import (
    BgeM3Embedder,
    BgeReranker,
    describe_hardware,
    load_or_compute_corpus_embeddings,
    pick_device,
)
from eval.retrieval.schema import Conversation, Turn


def _use_settings(monkeypatch, device: str | None) -> None:
    monkeypatch.setattr(config, "get_settings", lambda: SimpleNamespace(device=device))


def _fake_torch(mps: bool, cuda: bool) -> SimpleNamespace:
    return SimpleNamespace(
        __version__="2.6.0",
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        cuda=SimpleNamespace(is_available=lambda: cuda),
    )


# --------------------------------------------------------------------------- #
# Importing, devices, hardware                                                 #
# --------------------------------------------------------------------------- #
def test_the_modules_import_without_the_model_libraries() -> None:
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "sys.modules['sentence_transformers'] = None\n"
        "import eval.retrieval.local_models, eval.retrieval.store, eval.retrieval.db\n"
        "import eval.retrieval.data.load_db\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=config.REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_pick_device_prefers_the_argument_then_the_setting(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps=True, cuda=True))
    _use_settings(monkeypatch, "cuda:1")

    assert pick_device("cpu") == "cpu"
    assert pick_device() == "cuda:1"
    assert pick_device(None) == "cuda:1"


@pytest.mark.parametrize(
    ("mps", "cuda", "expected"),
    [(True, True, "mps"), (True, False, "mps"), (False, True, "cuda"), (False, False, "cpu")],
)
def test_pick_device_falls_back_to_the_best_available(
    monkeypatch,
    mps: bool,
    cuda: bool,
    expected: str,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps=mps, cuda=cuda))
    _use_settings(monkeypatch, None)

    assert pick_device() == expected


def test_pick_device_without_torch_is_the_cpu(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    _use_settings(monkeypatch, None)

    assert pick_device() == "cpu"
    assert pick_device("mps") == "mps"  # an explicit choice is not second-guessed


def test_describe_hardware_without_torch(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    _use_settings(monkeypatch, None)

    hardware = describe_hardware()

    assert list(hardware) == ["machine", "processor", "cpu_count", "ram_gb"]
    assert isinstance(hardware["machine"], str) and hardware["machine"]
    assert isinstance(hardware["processor"], str) and hardware["processor"]
    assert isinstance(hardware["cpu_count"], int) and hardware["cpu_count"] >= 1
    ram_gb = hardware["ram_gb"]  # None only where the OS has no sysconf to ask
    assert ram_gb is None or (isinstance(ram_gb, float) and ram_gb > 0)
    assert json.loads(json.dumps(hardware)) == hardware  # goes into the results JSON as it is


def test_describe_hardware_adds_torch_and_the_device_when_torch_is_there(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps=True, cuda=False))
    _use_settings(monkeypatch, None)

    hardware = describe_hardware()

    assert list(hardware) == ["machine", "processor", "cpu_count", "ram_gb", "torch", "device"]
    assert (hardware["torch"], hardware["device"]) == ("2.6.0", "mps")


def test_processor_name_falls_back_when_the_os_probe_fails(monkeypatch) -> None:
    def no_sysctl(*args: Any, **kwargs: Any) -> None:
        raise FileNotFoundError("sysctl")

    monkeypatch.setattr(local_models.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(local_models.platform, "processor", lambda: "")
    monkeypatch.setattr(local_models.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(local_models.subprocess, "run", no_sysctl)

    assert local_models._processor_name() == "arm64"


# --------------------------------------------------------------------------- #
# The model wrappers, against a fake sentence_transformers                     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def library(monkeypatch) -> SimpleNamespace:
    """A fake ``sentence_transformers`` that logs how its models are built and called."""

    log = SimpleNamespace(loaded=[], encoded=[], predicted=[], overlaps=0)
    busy = {"SentenceTransformer": 0, "CrossEncoder": 0}

    def work_in(model: str) -> None:
        """Spend a moment inside ``model``, noting whether another thread is in it too."""

        busy[model] += 1
        log.overlaps += busy[model] > 1
        time.sleep(0.005)
        busy[model] -= 1

    class SentenceTransformer:
        def __init__(self, model_name: str, **kwargs: Any) -> None:
            log.loaded.append(("SentenceTransformer", model_name, kwargs))
            work_in("SentenceTransformer")

        def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
            log.encoded.append((sentences, kwargs))
            work_in("SentenceTransformer")
            return np.array([[len(text), 0.5, -0.1] for text in sentences], dtype=np.float32)

    class CrossEncoder:
        def __init__(self, model_name: str, **kwargs: Any) -> None:
            log.loaded.append(("CrossEncoder", model_name, kwargs))
            work_in("CrossEncoder")

        def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> np.ndarray:
            log.predicted.append((pairs, kwargs))
            work_in("CrossEncoder")
            return np.array([len(text) / 10 for _, text in pairs], dtype=np.float32)

    fake = SimpleNamespace(SentenceTransformer=SentenceTransformer, CrossEncoder=CrossEncoder)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    monkeypatch.setitem(sys.modules, "torch", None)
    _use_settings(monkeypatch, None)
    return log


# What the backend's BGEM3BiEncoder passes to ``encode``, plus the batch size.
ENCODE_OPTIONS = {"batch_size": 32, "normalize_embeddings": True}
THIRD = float(np.float32(-0.1))  # the fake model's third component, as a float32 comes back


def test_wrappers_satisfy_their_protocols_and_default_to_the_configured_models() -> None:
    embedder, scorer = BgeM3Embedder(), BgeReranker()

    assert isinstance(embedder, Embedder) and isinstance(scorer, PairScorer)
    assert (embedder.model_name, embedder.batch_size) == ("BAAI/bge-m3", 32)
    assert scorer.model_name == "BAAI/bge-reranker-v2-m3"
    assert (scorer.batch_size, scorer.max_length) == (32, 512)


def test_embedder_loads_on_first_use_and_encodes_like_the_backend(library: SimpleNamespace) -> None:
    embedder = BgeM3Embedder()
    assert library.loaded == []

    vectors = embedder.embed_documents(("climbing", "tea"))
    embedder.embed_documents(["again"])

    assert library.loaded == [("SentenceTransformer", "BAAI/bge-m3", {"device": "cpu"})]  # once
    assert library.encoded[0] == (["climbing", "tea"], ENCODE_OPTIONS)
    assert vectors == [[8.0, 0.5, THIRD], [3.0, 0.5, THIRD]]
    assert all(type(vector) is list for vector in vectors)
    assert all(type(value) is float for vector in vectors for value in vector)


def test_embed_query_returns_one_vector(library: SimpleNamespace) -> None:
    assert BgeM3Embedder().embed_query("what did Sam say?") == [17.0, 0.5, THIRD]
    assert library.encoded == [(["what did Sam say?"], ENCODE_OPTIONS)]


def test_embedder_passes_its_model_device_and_batch_size(library: SimpleNamespace) -> None:
    embedder = BgeM3Embedder("local/tiny-encoder", device="cuda:1", batch_size=4)

    embedder.embed_documents(["a"])

    assert embedder.model_name == "local/tiny-encoder"
    assert library.loaded == [("SentenceTransformer", "local/tiny-encoder", {"device": "cuda:1"})]
    assert library.encoded[0][1] == {"batch_size": 4, "normalize_embeddings": True}


def test_reranker_scores_query_text_pairs_like_the_backend(library: SimpleNamespace) -> None:
    scorer = BgeReranker()
    assert library.loaded == []

    scores = scorer.score("climbing?", ["a trip", "tea"])
    scorer.score("climbing?", ["more"])

    assert library.loaded == [
        ("CrossEncoder", "BAAI/bge-reranker-v2-m3", {"device": "cpu", "max_length": 512})
    ]
    pairs = [("climbing?", "a trip"), ("climbing?", "tea")]
    assert library.predicted[0] == (pairs, {"batch_size": 32})
    assert scores == [float(np.float32(0.6)), float(np.float32(0.3))]
    assert all(type(score) is float for score in scores)


def test_reranker_passes_its_model_device_and_limits(library: SimpleNamespace) -> None:
    scorer = BgeReranker("local/tiny-reranker", device="mps", batch_size=8, max_length=256)

    scorer.score("q", ["t"])

    assert library.loaded == [
        ("CrossEncoder", "local/tiny-reranker", {"device": "mps", "max_length": 256})
    ]
    assert library.predicted[0][1] == {"batch_size": 8}


def test_nothing_to_do_does_not_load_a_model(library: SimpleNamespace) -> None:
    assert BgeM3Embedder().embed_documents([]) == []
    assert BgeReranker().score("anything", []) == []
    assert library.loaded == []


def test_models_load_once_and_serve_one_thread_at_a_time(library: SimpleNamespace) -> None:
    embedder, scorer = BgeM3Embedder(), BgeReranker()
    calls = [lambda: embedder.embed_documents(["a", "b"]), lambda: scorer.score("q", ["a", "b"])]
    workers = [threading.Thread(target=call) for call in calls * 6]

    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert sorted(kind for kind, _, _ in library.loaded) == ["CrossEncoder", "SentenceTransformer"]
    assert (len(library.encoded), len(library.predicted)) == (6, 6)
    assert library.overlaps == 0  # no model was entered by two threads at once


# --------------------------------------------------------------------------- #
# The corpus-embedding cache                                                   #
# --------------------------------------------------------------------------- #
class CountingEmbedder:
    """Embeds a text as a few bytes of its hash and logs every call."""

    def __init__(self, model_name: str = "fake/bge-m3") -> None:
        self.model_name = model_name
        self.calls: list[list[str]] = []

    def vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(f"{self.model_name}|{text}".encode()).digest()
        return [byte / 255 for byte in digest[:6]]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.vector(text) for text in texts]


def _conversation(episode_id: str, summary: str) -> Conversation:
    start = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)
    return Conversation(
        episode_id=episode_id,
        person_id="person-sam",
        start_time=start,
        end_time=start + timedelta(minutes=5),
        dialogue=[Turn("Sam", "hi")],
        summary=summary,
    )


CORPUS = [
    _conversation("e2", "Sam booked the climbing gym."),
    _conversation("e1", "Ana is moving to Lisbon in März."),
    _conversation("e3", "Bo recommended a ramen place."),
]
SUMMARIES = [conversation.summary for conversation in CORPUS]


def test_cache_miss_embeds_the_summaries_and_writes_the_file(tmp_path: Path) -> None:
    cache_path = tmp_path / "not" / "there" / "yet" / "corpus.npz"
    embedder = CountingEmbedder()

    embeddings = load_or_compute_corpus_embeddings(CORPUS, embedder, cache_path)

    assert embedder.calls == [SUMMARIES]  # one batch, summaries only, in corpus order
    assert embeddings == {c.episode_id: embedder.vector(c.summary) for c in CORPUS}
    assert list(embeddings) == ["e2", "e1", "e3"]
    # Nothing but the finished cache: the temporary file it was written through is gone.
    assert [path.name for path in cache_path.parent.iterdir()] == ["corpus.npz"]


def test_cache_hit_returns_the_same_embeddings_without_the_embedder(tmp_path: Path) -> None:
    cache_path = tmp_path / "corpus.npz"
    computed = load_or_compute_corpus_embeddings(CORPUS, CountingEmbedder(), cache_path)
    later = CountingEmbedder()

    cached = load_or_compute_corpus_embeddings(CORPUS, later, cache_path)
    reordered = load_or_compute_corpus_embeddings(CORPUS[::-1], later, cache_path)

    assert later.calls == []
    assert cached == computed  # exactly: the cache keeps full precision
    assert all(type(value) is float for vector in cached.values() for value in vector)
    assert reordered == computed and list(reordered) == ["e3", "e1", "e2"]


def test_a_changed_summary_invalidates_the_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "corpus.npz"
    load_or_compute_corpus_embeddings(CORPUS, CountingEmbedder(), cache_path)
    edited = [CORPUS[0], replace(CORPUS[1], summary="Ana is moving to Porto in March."), CORPUS[2]]
    embedder = CountingEmbedder()

    embeddings = load_or_compute_corpus_embeddings(edited, embedder, cache_path)
    load_or_compute_corpus_embeddings(edited, embedder, cache_path)

    assert embedder.calls == [[c.summary for c in edited]]  # recomputed once, then cached again
    assert embeddings["e1"] == embedder.vector("Ana is moving to Porto in March.")
    assert embeddings["e2"] == embedder.vector(CORPUS[0].summary)


@pytest.mark.parametrize(
    ("conversations", "model_name"),
    [
        pytest.param(CORPUS, "fake/other-model", id="model"),
        pytest.param([*CORPUS, _conversation("e4", "New.")], "fake/bge-m3", id="added"),
        pytest.param(CORPUS[:2], "fake/bge-m3", id="removed"),
        pytest.param([replace(CORPUS[0], episode_id="e9"), *CORPUS[1:]], "fake/bge-m3", id="id"),
    ],
)
def test_the_cache_key_covers_the_model_the_ids_and_the_summaries(
    tmp_path: Path,
    conversations: list[Conversation],
    model_name: str,
) -> None:
    cache_path = tmp_path / "corpus.npz"
    load_or_compute_corpus_embeddings(CORPUS, CountingEmbedder("fake/bge-m3"), cache_path)
    embedder = CountingEmbedder(model_name)

    embeddings = load_or_compute_corpus_embeddings(conversations, embedder, cache_path)

    assert embedder.calls == [[c.summary for c in conversations]]
    assert embeddings == {c.episode_id: embedder.vector(c.summary) for c in conversations}


@pytest.mark.parametrize(
    "damaged",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"not an npz archive", id="garbage"),
        pytest.param(b"PK\x03\x04 cut short", id="truncated-zip"),
    ],
)
def test_an_unreadable_cache_is_recomputed_and_replaced(tmp_path: Path, damaged: bytes) -> None:
    cache_path = tmp_path / "corpus.npz"
    cache_path.write_bytes(damaged)
    embedder = CountingEmbedder()

    embeddings = load_or_compute_corpus_embeddings(CORPUS, embedder, cache_path)
    again = load_or_compute_corpus_embeddings(CORPUS, embedder, cache_path)

    assert embedder.calls == [SUMMARIES]  # computed once; the second call read the new cache
    assert embeddings == again == {c.episode_id: embedder.vector(c.summary) for c in CORPUS}


def test_a_cache_file_without_the_expected_arrays_is_recomputed(tmp_path: Path) -> None:
    cache_path = tmp_path / "corpus.npz"
    np.savez(cache_path, something_else=np.arange(3))
    embedder = CountingEmbedder()

    embeddings = load_or_compute_corpus_embeddings(CORPUS, embedder, cache_path)

    assert embedder.calls == [SUMMARIES]
    assert list(embeddings) == ["e2", "e1", "e3"]


def test_an_embedder_returning_the_wrong_number_of_vectors_is_rejected(tmp_path: Path) -> None:
    class ShortEmbedder(CountingEmbedder):
        def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
            return super().embed_documents(texts)[:-1]

    cache_path = tmp_path / "corpus.npz"

    with pytest.raises(ValueError, match="returned 2 vectors for 3 summaries"):
        load_or_compute_corpus_embeddings(CORPUS, ShortEmbedder(), cache_path)

    assert not cache_path.exists()


def test_the_default_cache_lives_in_the_gitignored_cache_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default path follows ``config.CACHE_DIR`` at call time, and that is the ``.cache`` dir."""

    assert config.CACHE_DIR == config.RETRIEVAL_ROOT / ".cache"
    assert inspect.signature(load_or_compute_corpus_embeddings).parameters["cache_path"].default is None
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")

    load_or_compute_corpus_embeddings(CORPUS, CountingEmbedder())

    assert (tmp_path / "cache" / "corpus_embeddings.npz").exists()
