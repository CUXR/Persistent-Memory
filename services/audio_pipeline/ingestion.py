"""Audio ingestion pipeline — rolling window state machine.

This is the top-level entry point for the audio segmentation pipeline.
Callers push raw audio chunks in real time; the pipeline manages VAD
pre-screening, conversation-buffer accumulation, diarization, and speaker
attribution internally.

State machine
-------------
::

    ┌──────────┐  speech detected   ┌─────────────┐  buf full / flush()
    │   IDLE   │ ─────────────────► │ ACCUMULATING │ ──────────────────►┐
    └──────────┘                    └─────────────┘                     │
         ▲                                 │                            │
         │    SILENCE_WINDOWS_EOC silent   │                   ┌────────▼───────┐
         └─────────────────────────────────┘                   │   PROCESSING   │
                                                               └────────────────┘
                                                                        │
                                                               segments returned
                                                               → caller, state → IDLE

Transitions:

* **IDLE → ACCUMULATING**: When a VAD window returns ``contains_speech=True``.
  The 5-second window that triggered the transition is prepended to the
  conversation buffer so no speech is lost.

* **ACCUMULATING → IDLE** (via PROCESSING): Either the conversation buffer
  reaches ``MAX_CONVERSATION_SAMPLES`` or ``SILENCE_WINDOWS_EOC`` consecutive
  silent 5-second windows are observed.  ``flush()`` also forces this path.

Usage::

    vad = SileroVAD()
    engine = DiarizationEngine(hf_token="hf_...")
    attributor = SpeakerAttributor()

    pipeline = AudioIngestionPipeline(
        vad=vad,
        diarization_engine=engine,
        attributor=attributor,
        enrolled_user_embedding=user_voice_embedding,
    )

    for chunk in audio_stream:
        segments = pipeline.push_audio(chunk)
        for seg in segments:
            print(seg)

    # Force-process whatever is left in the buffer.
    final_segments = pipeline.flush()
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Optional

import numpy as np

from .diarization import DiarizationEngine
from .segment import (
    SAMPLE_RATE,
    VAD_WINDOW_SAMPLES,
    MAX_CONVERSATION_SAMPLES,
    SILENCE_WINDOWS_EOC,
    AudioSegment,
)
from .speaker_attribution import AttributionConfig, SpeakerAttributor
from .vad import SileroVAD

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """Internal state of the ingestion pipeline."""

    IDLE = auto()
    """No speech has been detected; 5-second rolling VAD windows are being discarded."""

    ACCUMULATING = auto()
    """Speech detected; audio is being accumulated into the conversation buffer."""

    PROCESSING = auto()
    """The conversation buffer is being diarized (transient; not observable by caller)."""


class AudioIngestionPipeline:
    """Orchestrates VAD → diarization → attribution in a streaming fashion.

    Args:
        vad:                        Initialised :class:`~.vad.SileroVAD`.
        diarization_engine:         Initialised :class:`~.diarization.DiarizationEngine`.
        attributor:                 Initialised :class:`~.speaker_attribution.SpeakerAttributor`.
        enrolled_user_embedding:    Unit-normed 1-D float32 vector from
                                    ``Person.voice_embedding``.  Pass ``None``
                                    to use relative (best-match) attribution.
        sample_rate:                Audio sample rate (Hz).  Must match the
                                    model requirements (default 16 000).
        silence_windows_eoc:        Consecutive silent VAD windows before
                                    end-of-conversation is declared.
    """

    def __init__(
        self,
        vad: SileroVAD,
        diarization_engine: DiarizationEngine,
        attributor: SpeakerAttributor,
        enrolled_user_embedding: Optional[np.ndarray] = None,
        sample_rate: int = SAMPLE_RATE,
        silence_windows_eoc: int = SILENCE_WINDOWS_EOC,
    ) -> None:
        self._vad = vad
        self._diarization_engine = diarization_engine
        self._attributor = attributor
        self._enrolled_embedding = enrolled_user_embedding
        self._sample_rate = sample_rate
        self._silence_windows_eoc = silence_windows_eoc

        # Conversation buffer.
        self._conversation_buffer: list[np.ndarray] = []
        self._conversation_sample_count: int = 0

        # Offset (in samples) from the start of the recording to the first
        # sample in the current conversation buffer.
        self._conversation_start_sample: int = 0

        # Running total of samples seen since the pipeline was created (or last
        # reset).  Used to set the conversation_start_sample.
        self._total_samples_seen: int = 0

        self._state: ConversationState = ConversationState.IDLE
        self._consecutive_silent_windows: int = 0

        # Partial chunk accumulator (samples not yet forming a full VAD window).
        self._partial: np.ndarray = np.empty(0, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def state(self) -> ConversationState:
        """Current pipeline state (read-only)."""
        return self._state

    @property
    def conversation_duration_seconds(self) -> float:
        """Accumulated conversation audio length in seconds."""
        return self._conversation_sample_count / self._sample_rate

    def push_audio(
        self, chunk: np.ndarray
    ) -> list[AudioSegment]:
        """Feed a raw audio chunk into the pipeline.

        The chunk can be **any length** — the pipeline buffers internally and
        processes only when a complete VAD window (5 s) has been accumulated.

        Args:
            chunk:  1-D float32 PCM array at the configured sample rate.

        Returns:
            A (possibly empty) list of ``AudioSegment`` objects.  Segments are
            returned only when a conversation ends (either via silence
            end-of-conversation detection or conversation-buffer overflow).
        """
        if chunk.ndim != 1:
            raise ValueError(f"chunk must be 1-D, got shape {chunk.shape}")

        # Append to partial buffer.
        self._partial = np.concatenate([self._partial, chunk.astype(np.float32)])

        results: list[AudioSegment] = []

        # Drain full VAD windows from the partial buffer.
        while len(self._partial) >= VAD_WINDOW_SAMPLES:
            window = self._partial[:VAD_WINDOW_SAMPLES]
            self._partial = self._partial[VAD_WINDOW_SAMPLES:]
            segments = self._process_vad_window(window)
            results.extend(segments)

        return results

    def flush(self) -> list[AudioSegment]:
        """Process any remaining audio in the conversation buffer.

        Call at stream end or whenever you want to force a conversation
        boundary.  Includes sub-window partial audio if present.

        Returns:
            List of ``AudioSegment`` objects from the remaining buffer.
        """
        # Absorb any remaining partial audio into the conversation buffer.
        if len(self._partial) > 0:
            if self._state == ConversationState.ACCUMULATING:
                self._append_to_conversation(self._partial)
            self._partial = np.empty(0, dtype=np.float32)
            self._total_samples_seen += len(self._partial)

        if self._state == ConversationState.ACCUMULATING:
            return self._finalize_conversation()

        # Reset to clean state regardless.
        self._reset()
        return []

    def reset(self) -> None:
        """Discard all accumulated audio and return to IDLE state."""
        self._reset()

    # ------------------------------------------------------------------
    # Private: state-machine methods
    # ------------------------------------------------------------------

    def _process_vad_window(self, window: np.ndarray) -> list[AudioSegment]:
        """Handle one complete 5-second VAD window.

        Returns segments if a conversation was finalized.
        """
        window_samples = len(window)
        self._total_samples_seen += window_samples

        has_speech = self._vad.contains_speech(window, self._sample_rate)
        logger.debug(
            "VAD window: state=%s speech=%s (total=%.1fs)",
            self._state.name,
            has_speech,
            self._total_samples_seen / self._sample_rate,
        )

        if self._state == ConversationState.IDLE:
            if has_speech:
                logger.info(
                    "Speech detected at %.1fs — entering ACCUMULATING state.",
                    self._total_samples_seen / self._sample_rate,
                )
                # Record where this conversation starts.
                self._conversation_start_sample = (
                    self._total_samples_seen - window_samples
                )
                self._state = ConversationState.ACCUMULATING
                self._consecutive_silent_windows = 0
                self._append_to_conversation(window)
            # else: stay IDLE, window discarded.
            return []

        if self._state == ConversationState.ACCUMULATING:
            self._append_to_conversation(window)

            if has_speech:
                self._consecutive_silent_windows = 0
            else:
                self._consecutive_silent_windows += 1
                logger.debug(
                    "Silent window %d/%d in ACCUMULATING state.",
                    self._consecutive_silent_windows,
                    self._silence_windows_eoc,
                )

            # End-of-conversation: enough consecutive silence.
            if self._consecutive_silent_windows >= self._silence_windows_eoc:
                logger.info(
                    "End-of-conversation detected after %.1fs of speech.",
                    self.conversation_duration_seconds,
                )
                return self._finalize_conversation()

            # Buffer overflow guard: process immediately, then reset.
            if self._conversation_sample_count >= MAX_CONVERSATION_SAMPLES:
                logger.info(
                    "Conversation buffer full (%.0fs) — processing now.",
                    MAX_CONVERSATION_SAMPLES / self._sample_rate,
                )
                return self._finalize_conversation()

            return []

        return []  # Should not be reached (PROCESSING is transient).

    def _append_to_conversation(self, window: np.ndarray) -> None:
        self._conversation_buffer.append(window)
        self._conversation_sample_count += len(window)

    def _finalize_conversation(self) -> list[AudioSegment]:
        """Diarize + attribute the accumulated conversation buffer.

        Returns the labelled segment list and resets to IDLE.
        """
        self._state = ConversationState.PROCESSING

        if not self._conversation_buffer:
            self._reset()
            return []

        audio = np.concatenate(self._conversation_buffer)
        offset_seconds = self._conversation_start_sample / self._sample_rate

        logger.info(
            "Processing conversation: %.1fs of audio (offset=%.1fs).",
            len(audio) / self._sample_rate,
            offset_seconds,
        )

        try:
            turns = self._diarization_engine.diarize(audio, self._sample_rate)

            if not turns:
                logger.info("Diarization returned no turns.")
                self._reset()
                return []

            speaker_embeddings = self._diarization_engine.extract_per_speaker_embeddings(
                audio, turns, self._sample_rate
            )

            segments = self._attributor.attribute(
                turns=turns,
                speaker_embeddings=speaker_embeddings,
                enrolled_user_embedding=self._enrolled_embedding,
                conversation_offset_seconds=offset_seconds,
            )

            logger.info(
                "Conversation processed: %d segment(s) produced.", len(segments)
            )
        except Exception:
            logger.exception("Error during conversation processing.")
            segments = []

        self._reset()
        return segments

    def _reset(self) -> None:
        self._conversation_buffer = []
        self._conversation_sample_count = 0
        self._conversation_start_sample = 0
        self._consecutive_silent_windows = 0
        self._state = ConversationState.IDLE
        logger.debug("Pipeline reset to IDLE.")
