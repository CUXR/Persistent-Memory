"""Speaker attribution via cosine similarity.

Given diarized speaker turns and the wearer's enrolled voice embedding
(stored in ``Person.voice_embedding``), this module labels each turn as
``"user"``, ``"interlocutor"``, or ``"uncertain"``.

Labelling logic
---------------
For each diarized speaker:

1. Compute cosine similarity between the speaker's mean embedding and the
   enrolled user embedding.
2. If similarity ≥ ``user_threshold``  → label as ``"user"``.
3. If similarity ≤ ``interlocutor_threshold`` → label as ``"interlocutor"``.
4. Otherwise (within the uncertain band) → label as ``"uncertain"``.

When **no** enrolled user embedding is provided (``None``), the speaker with
the highest similarity is labelled ``"user"`` only when two or more speakers
are present and their top similarity exceeds ``user_threshold``; otherwise
both are labelled ``"uncertain"``.

Adjacent same-label turns from the same speaker are stitched into a single
``AudioSegment`` to avoid fragmentation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .diarization import DiarizedTurn
from .segment import AudioSegment, SAMPLE_RATE, SpeakerLabel

logger = logging.getLogger(__name__)


@dataclass
class AttributionConfig:
    """Thresholds that control speaker label assignment.

    Attributes:
        user_threshold:             Cosine similarity at or above which a
                                    speaker is labelled ``"user"``.  Default 0.75.
        interlocutor_threshold:     Cosine similarity at or below which a
                                    speaker is labelled ``"interlocutor"``.
                                    Default 0.45.
        min_turn_duration:          Turns shorter than this (seconds) receive
                                    ``"uncertain"`` regardless of similarity.
        stitch_gap_seconds:         Adjacent same-speaker/same-label turns
                                    within this gap are merged.
    """

    user_threshold: float = 0.75
    interlocutor_threshold: float = 0.45
    min_turn_duration: float = 0.1
    stitch_gap_seconds: float = 0.3


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity in [-1, 1] between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SpeakerAttributor:
    """Labels diarized speaker turns as user / interlocutor / uncertain.

    Args:
        config:  Threshold configuration (uses defaults if not supplied).
    """

    def __init__(self, config: Optional[AttributionConfig] = None) -> None:
        self.config = config or AttributionConfig()

    # ------------------------------------------------------------------
    def attribute(
        self,
        turns: list[DiarizedTurn],
        speaker_embeddings: dict[str, np.ndarray],
        enrolled_user_embedding: Optional[np.ndarray],
        conversation_offset_seconds: float = 0.0,
    ) -> list[AudioSegment]:
        """Map diarized turns to labelled ``AudioSegment`` objects.

        Args:
            turns:                      Output of ``DiarizationEngine.diarize()``.
            speaker_embeddings:         Per-speaker mean embeddings from
                                        ``DiarizationEngine.extract_per_speaker_embeddings()``.
            enrolled_user_embedding:    Unit-normed embedding from
                                        ``Person.voice_embedding``.  Pass ``None``
                                        to use relative (best-match) attribution.
            conversation_offset_seconds: Seconds to add to all turn timestamps
                                        (used when diarizing a sub-slice of a
                                        longer recording).

        Returns:
            List of ``AudioSegment`` objects sorted by ``start_time``.
            Adjacent same-label turns from the same speaker are stitched.
        """
        if not turns:
            return []

        # ------------------------------------------------------------------
        # 1. Determine label for every unique speaker.
        # ------------------------------------------------------------------
        speaker_labels: dict[str, tuple[SpeakerLabel, float]] = {}  # id → (label, confidence)
        speaker_ids = list({t.speaker_id for t in turns})

        if enrolled_user_embedding is not None:
            for sid in speaker_ids:
                emb = speaker_embeddings.get(sid)
                if emb is None:
                    speaker_labels[sid] = ("uncertain", 0.0)
                    continue
                sim = _cosine_similarity(emb, enrolled_user_embedding)
                # Map to [0, 1] range for confidence reporting.
                confidence = (sim + 1.0) / 2.0
                if sim >= self.config.user_threshold:
                    label: SpeakerLabel = "user"
                elif sim <= self.config.interlocutor_threshold:
                    label = "interlocutor"
                else:
                    label = "uncertain"
                speaker_labels[sid] = (label, confidence)
                logger.debug(
                    "Speaker %s: cosine_sim=%.3f → %s", sid, sim, label
                )
        else:
            # Relative attribution: if exactly two speakers exist and we have
            # embeddings for both, the one with the higher internal coherence
            # (norm) gets "user" tentatively, the other gets "interlocutor".
            # With fewer or more speakers, all become "uncertain".
            if len(speaker_ids) == 2:
                sims: list[tuple[str, float]] = []
                for sid in speaker_ids:
                    emb = speaker_embeddings.get(sid)
                    if emb is not None:
                        sims.append((sid, float(np.linalg.norm(emb))))
                if len(sims) == 2:
                    sims.sort(key=lambda x: x[1], reverse=True)
                    speaker_labels[sims[0][0]] = ("user", 0.5)
                    speaker_labels[sims[1][0]] = ("interlocutor", 0.5)
                    logger.debug(
                        "No enrolled embedding; relative attribution used."
                    )
                else:
                    for sid in speaker_ids:
                        speaker_labels[sid] = ("uncertain", 0.0)
            else:
                for sid in speaker_ids:
                    speaker_labels[sid] = ("uncertain", 0.0)

        # ------------------------------------------------------------------
        # 2. Build a raw segment list (one per turn).
        # ------------------------------------------------------------------
        segments: list[AudioSegment] = []
        for turn in sorted(turns, key=lambda t: t.start):
            if turn.duration < self.config.min_turn_duration:
                logger.debug(
                    "Skipping turn %s (%.3fs < min %.3fs)",
                    turn.speaker_id,
                    turn.duration,
                    self.config.min_turn_duration,
                )
                continue

            label, confidence = speaker_labels.get(turn.speaker_id, ("uncertain", 0.0))
            abs_start = turn.start + conversation_offset_seconds
            abs_end = turn.end + conversation_offset_seconds
            segments.append(
                AudioSegment.from_times(
                    start_time=abs_start,
                    end_time=abs_end,
                    speaker_label=label,
                    confidence=confidence,
                    speaker_id=turn.speaker_id,
                )
            )

        # ------------------------------------------------------------------
        # 3. Stitch adjacent same-label/same-speaker segments.
        # ------------------------------------------------------------------
        return _stitch_segments(segments, self.config.stitch_gap_seconds)


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _stitch_segments(
    segments: list[AudioSegment],
    gap_threshold: float,
) -> list[AudioSegment]:
    """Merge consecutive same-speaker same-label segments within *gap_threshold*.

    The stitched segment inherits the maximum confidence of the merged group
    and the combined start/end sample range.

    Args:
        segments:       Sorted list of AudioSegment objects.
        gap_threshold:  Maximum gap in seconds to bridge.

    Returns:
        New list with eligible consecutive segments merged.
    """
    if not segments:
        return []

    stitched: list[AudioSegment] = [segments[0]]

    for seg in segments[1:]:
        prev = stitched[-1]
        gap = seg.start_time - prev.end_time
        same_speaker = seg.speaker_id == prev.speaker_id
        same_label = seg.speaker_label == prev.speaker_label

        if same_speaker and same_label and gap <= gap_threshold:
            # Extend the previous segment.
            stitched[-1] = AudioSegment(
                start_time=prev.start_time,
                end_time=seg.end_time,
                speaker_label=prev.speaker_label,
                start_sample=prev.start_sample,
                end_sample=seg.end_sample,
                confidence=max(prev.confidence, seg.confidence),
                speaker_id=prev.speaker_id,
            )
            logger.debug(
                "Stitched %s + %s → %.2f–%.2fs",
                prev,
                seg,
                prev.start_time,
                seg.end_time,
            )
        else:
            stitched.append(seg)

    return stitched
