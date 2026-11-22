"""The local models (bge-m3 bi-encoder, bge-reranker-v2-m3 cross-encoder) and what they need.

torch and sentence-transformers are imported inside the functions that use
them. The module therefore imports in the backend environment, where neither
is installed, and unit tests can put fakes in their place.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import threading
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .interfaces import Embedder
from .schema import Conversation


def pick_device(preferred: str | None = None) -> str:
    """The torch device the local models run on.

    ``preferred`` wins, then ``EVAL_DEVICE``; otherwise the best one available:
    Apple's MPS, then CUDA, then the CPU (also the answer without torch).
    """

    configured = preferred or config.get_settings().device
    if configured:
        return configured
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class BgeM3Embedder:
    """The bge-m3 bi-encoder behind ``interfaces.Embedder``.

    It encodes with the call the backend's ``BGEM3BiEncoder`` makes
    (``SentenceTransformer.encode(..., normalize_embeddings=True)``), so the
    benchmark ranks with the vectors the product would compute. The model
    loads on first use.
    """

    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._device = device
        self._model: Any | None = None
        # One caller at a time: loading must happen once, and the tokenizer
        # under ``encode`` is not safe to share between threads.
        self._lock = threading.Lock()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, device=pick_device(self._device))
            vectors = self._model.encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=True,
            )
        return np.asarray(vectors, dtype=float).tolist()


class BgeReranker:
    """The bge-reranker-v2-m3 cross-encoder behind ``interfaces.PairScorer``.

    It scores with the call the backend's ``BGEReranker`` makes
    (``CrossEncoder.predict`` on ``(query, text)`` pairs). The model loads on
    first use.
    """

    def __init__(
        self,
        model_name: str = config.RERANKER_MODEL,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._device = device
        self._model: Any | None = None
        self._lock = threading.Lock()  # as in BgeM3Embedder

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(
                    self.model_name,
                    device=pick_device(self._device),
                    max_length=self.max_length,
                )
            predictions = self._model.predict(
                [(query, text) for text in texts],
                batch_size=self.batch_size,
            )
        return [float(prediction) for prediction in predictions]


def load_or_compute_corpus_embeddings(
    conversations: Sequence[Conversation],
    embedder: Embedder,
    cache_path: Path | None = None,
) -> dict[str, list[float]]:
    """Summary embeddings by episode id, computed once and then read from ``cache_path``.

    The cache is keyed by a hash of the model name and every (episode id,
    summary) pair. Editing a summary, adding a conversation or changing the
    model therefore recomputes, rather than serving vectors of the old text.
    The default path is resolved per call, so it follows ``config.CACHE_DIR``.
    """

    if cache_path is None:
        cache_path = config.CACHE_DIR / "corpus_embeddings.npz"
    key = _corpus_key(embedder.model_name, conversations)
    episode_ids = [conversation.episode_id for conversation in conversations]
    cached = _read_cache(cache_path, key)
    if cached is None:
        vectors = embedder.embed_documents([conversation.summary for conversation in conversations])
        if len(vectors) != len(conversations):
            raise ValueError(
                f"the embedder returned {len(vectors)} vectors for {len(conversations)} summaries"
            )
        matrix = np.asarray(vectors, dtype=float)
        _write_cache(cache_path, key, episode_ids, matrix)
        cached = dict(zip(episode_ids, matrix.tolist()))
    return {episode_id: cached[episode_id] for episode_id in episode_ids}


def describe_hardware() -> dict[str, Any]:
    """The machine the local models run on, for the report's latency section.

    Works without torch; with it, adds the torch version and the device the
    models use.
    """

    hardware: dict[str, Any] = {
        "machine": platform.machine(),
        "processor": _processor_name(),
        "cpu_count": os.cpu_count(),
        "ram_gb": _ram_gb(),
    }
    try:
        import torch
    except ImportError:
        return hardware
    hardware["torch"] = str(torch.__version__)
    hardware["device"] = pick_device()
    return hardware


def _corpus_key(model_name: str, conversations: Sequence[Conversation]) -> str:
    """Hash of what the embeddings depend on; the order of the conversations does not matter."""

    pairs = sorted((record.episode_id, record.summary) for record in conversations)
    payload = json.dumps([model_name, pairs], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_cache(cache_path: Path, key: str) -> dict[str, list[float]] | None:
    """The cached embeddings, or None when they have to be computed.

    That is when there is no cache, when it was written for another key, and
    when the file cannot be read as a cache at all: it is then replaced, since
    a damaged cache is no reason to stop a run.
    """

    try:
        with np.load(cache_path) as cache:
            if str(cache["key"]) != key:
                return None
            return dict(zip(cache["episode_ids"].tolist(), cache["embeddings"].tolist()))
    except (OSError, EOFError, KeyError, ValueError, zipfile.BadZipFile):
        return None


def _write_cache(
    cache_path: Path,
    key: str,
    episode_ids: Sequence[str],
    matrix: np.ndarray,
) -> None:
    """Write the cache through a temporary file, so an interrupted run never leaves half of one."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = cache_path.with_name(cache_path.name + ".partial")
    with partial_path.open("wb") as handle:
        np.savez(
            handle,
            key=np.array(key),
            episode_ids=np.array(episode_ids, dtype=str),
            embeddings=matrix,
        )
    os.replace(partial_path, cache_path)


def _processor_name() -> str:
    """The CPU model as the OS names it (``platform.processor()`` says only "arm" on a Mac)."""

    system = platform.system()
    if system == "Darwin":
        try:
            brand = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            brand = ""
        if brand:
            return brand
    elif system == "Linux":
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        except OSError:
            cpuinfo = ""
        for line in cpuinfo.splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or platform.machine()


def _ram_gb() -> float | None:
    """Installed memory in GiB, or None where the OS does not say (no ``sysconf``)."""

    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (AttributeError, ValueError, OSError):
        return None
