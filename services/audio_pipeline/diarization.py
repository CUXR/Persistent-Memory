"""pyannote speaker diarization wrapper.

Wraps ``pyannote/speaker-diarization-community-1`` and exposes a clean
``DiarizedTurn`` list.  The underlying pipeline is loaded lazily on first use.

The module also provides ``extract_speaker_embedding``, which runs pyannote's
embedding model on a waveform slice to produce a fixed-length speaker vector
suitable for cosine similarity comparison with the enrolled user embedding
stored in ``Person.voice_embedding``.

Model requirements
------------------
- HuggingFace access token with the model licence accepted at
  https://huggingface.co/pyannote/speaker-diarization-community-1
- ``pyannote.audio`` installed (see services/requirements.txt)
- Mono 16 kHz audio input
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .segment import SAMPLE_RATE

logger = logging.getLogger(__name__)

_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
_EMBEDDING_MODEL = "pyannote/embedding"


@dataclass
class DiarizedTurn:
    """One speaker turn produced by pyannote diarization.

    Attributes:
        start:      Turn start in seconds (relative to diarized audio start).
        end:        Turn end in seconds.
        speaker_id: Opaque label assigned by the diarizer (e.g. ``"SPEAKER_00"``).
                    Stable within one diarization call; not meaningful across calls.
    """

    start: float
    end: float
    speaker_id: str

    @property
    def duration(self) -> float:
        return self.end - self.start


class DiarizationEngine:
    """Thin wrapper around the pyannote speaker diarization pipeline.

    Args:
        hf_token:           HuggingFace access token (required by the community
                            model to verify licence acceptance).
        use_gpu:            Move the pipeline to CUDA if ``True`` and a device
                            is available.  Defaults to auto-detect.
    """

    def __init__(
        self,
        hf_token: str,
        use_gpu: Optional[bool] = None,
    ) -> None:
        if not hf_token:
            raise ValueError("hf_token must be a non-empty string")
        self._hf_token = hf_token
        self._use_gpu = use_gpu
        self._pipeline = None
        self._embedding_model = None

    # ------------------------------------------------------------------
    def _ensure_pipeline(self) -> None:
        if self._pipeline is not None:
            return

        import torch
        from pyannote.audio import Pipeline

        logger.info("Loading pyannote diarization pipeline '%s' …", _DIARIZATION_MODEL)
        pipeline = Pipeline.from_pretrained(
            _DIARIZATION_MODEL,
            use_auth_token=self._hf_token,
        )

        should_use_gpu = (
            torch.cuda.is_available()
            if self._use_gpu is None
            else self._use_gpu and torch.cuda.is_available()
        )
        if should_use_gpu:
            pipeline = pipeline.to(torch.device("cuda"))
            logger.info("Diarization pipeline moved to CUDA.")
        else:
            logger.info("Diarization pipeline running on CPU.")

        self._pipeline = pipeline

    # ------------------------------------------------------------------
    def _ensure_embedding_model(self) -> None:
        if self._embedding_model is not None:
            return

        from pyannote.audio import Inference

        logger.info("Loading pyannote embedding model '%s' …", _EMBEDDING_MODEL)
        self._embedding_model = Inference(
            _EMBEDDING_MODEL,
            window="whole",
            use_auth_token=self._hf_token,
        )
        logger.info("Embedding model loaded.")

    # ------------------------------------------------------------------
    def diarize(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> list[DiarizedTurn]:
        """Diarize *audio* and return a time-sorted list of speaker turns.

        The method tries pyannote's ``exclusive_speaker_diarization`` first
        (which ensures non-overlapping speaker segments) and falls back to
        standard diarization if unavailable.

        Args:
            audio:          1-D float32 PCM array at *sample_rate* Hz.
            sample_rate:    Audio sample rate (must be 16 000 for community-1).

        Returns:
            List of :class:`DiarizedTurn` sorted by ``start`` time.
            Returns an empty list if no speech is found.
        """
        import torch

        self._ensure_pipeline()

        if audio.ndim != 1:
            raise ValueError(f"audio must be 1-D, got shape {audio.shape}")
        if len(audio) == 0:
            return []

        waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        input_dict = {"waveform": waveform, "sample_rate": sample_rate}

        # Prefer exclusive_speaker_diarization for cleaner downstream alignment.
        annotation = None
        try:
            annotation = self._pipeline.exclusive_speaker_diarization(input_dict)
            logger.debug("Used exclusive_speaker_diarization.")
        except AttributeError:
            pass

        if annotation is None:
            annotation = self._pipeline(input_dict)
            logger.debug("Used standard diarization.")

        turns: list[DiarizedTurn] = []
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            if segment.end <= segment.start:
                continue
            turns.append(
                DiarizedTurn(
                    start=segment.start,
                    end=segment.end,
                    speaker_id=speaker,
                )
            )

        turns.sort(key=lambda t: t.start)
        logger.debug("Diarization returned %d turn(s).", len(turns))
        return turns

    # ------------------------------------------------------------------
    def extract_speaker_embedding(
        self,
        audio: np.ndarray,
        start_sec: float,
        end_sec: float,
        sample_rate: int = SAMPLE_RATE,
    ) -> np.ndarray:
        """Return a fixed-length embedding vector for the given speaker segment.

        Uses the ``pyannote/embedding`` model.  The vector can be compared with
        ``Person.voice_embedding`` via cosine similarity.

        Args:
            audio:          1-D float32 array for the **full** conversation
                            recording (the model crops internally using the
                            Segment boundaries).
            start_sec:      Segment start in seconds (relative to ``audio``).
            end_sec:        Segment end in seconds.
            sample_rate:    Audio sample rate.

        Returns:
            1-D float32 numpy array (embedding dimension depends on the model,
            typically 512).
        """
        import torch
        from pyannote.core import Segment

        self._ensure_embedding_model()

        waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        input_dict = {"waveform": waveform, "sample_rate": sample_rate}
        region = Segment(start_sec, end_sec)

        embedding = self._embedding_model(input_dict, region)

        # Normalise — pyannote embeddings may or may not be unit-norm.
        arr = np.array(embedding, dtype=np.float32).flatten()
        norm = np.linalg.norm(arr)
        if norm > 1e-10:
            arr = arr / norm
        return arr

    # ------------------------------------------------------------------
    def extract_per_speaker_embeddings(
        self,
        audio: np.ndarray,
        turns: list[DiarizedTurn],
        sample_rate: int = SAMPLE_RATE,
        min_duration: float = 1.0,
    ) -> dict[str, np.ndarray]:
        """Return a mean embedding per unique speaker in *turns*.

        For each speaker, all turns longer than *min_duration* seconds are
        embedded individually and then averaged.  Short turns are skipped to
        avoid noisy embeddings.

        Args:
            audio:          1-D float32 PCM array for the full conversation.
            turns:          Output from :meth:`diarize`.
            sample_rate:    Audio sample rate.
            min_duration:   Minimum turn length (seconds) to include in the
                            per-speaker average.

        Returns:
            Dict mapping ``speaker_id`` → unit-normed mean embedding.
            Speakers with no turns longer than *min_duration* are absent.
        """
        from collections import defaultdict

        embeddings_by_speaker: dict[str, list[np.ndarray]] = defaultdict(list)

        for turn in turns:
            if turn.duration < min_duration:
                continue
            emb = self.extract_speaker_embedding(
                audio, turn.start, turn.end, sample_rate
            )
            embeddings_by_speaker[turn.speaker_id].append(emb)

        result: dict[str, np.ndarray] = {}
        for speaker_id, embs in embeddings_by_speaker.items():
            mean_emb = np.mean(np.stack(embs, axis=0), axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm > 1e-10:
                mean_emb = mean_emb / norm
            result[speaker_id] = mean_emb

        logger.debug(
            "Extracted embeddings for %d/%d speakers.",
            len(result),
            len({t.speaker_id for t in turns}),
        )
        return result
