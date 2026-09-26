"""Match known voices and discover the sole new speaker in a conversation."""
from dataclasses import dataclass
from uuid import UUID

import numpy as np

from .diarization import DiarizedTurn
from .segment import DiscoveredVoice, Recording, SpeechSegment, VoiceProfile


@dataclass(frozen=True)
class AttributionConfig:
    match_threshold: float = 0.75
    min_margin: float = 0.15

    def __post_init__(self):
        if not 0 <= self.match_threshold <= 1 or not 0 < self.min_margin <= 2:
            raise ValueError("Invalid speaker attribution thresholds")


class SpeakerAttributor:
    def __init__(self, config: AttributionConfig | None = None):
        self.config = config or AttributionConfig()

    def attribute(
        self, turns: list[DiarizedTurn], speaker_embeddings: dict[str, np.ndarray],
        recording: Recording, *, embedding_model: str,
        user_profile: VoiceProfile | None, interlocutor_profile: VoiceProfile | None,
        interlocutor_id: UUID,
    ) -> list[SpeechSegment]:
        profiles = {"user": user_profile, "interlocutor": interlocutor_profile}
        for profile in profiles.values():
            if profile is not None and profile.model != embedding_model:
                raise ValueError("Voice model mismatch; re-enroll with the configured model")
        labels = {}
        for speaker_id in {turn.speaker_id for turn in turns}:
            vector = speaker_embeddings.get(speaker_id)
            if vector is None:
                labels[speaker_id] = ("unknown", None)
                continue
            vector = np.asarray(VoiceProfile(model=embedding_model, embedding=vector.tolist()).embedding)
            scores = sorted(
                [(float(np.clip(np.dot(vector, profile.embedding), -1, 1)), role)
                 for role, profile in profiles.items() if profile is not None], reverse=True,
            )
            role, similarity = "unknown", scores[0][0] if scores else None
            if scores and scores[0][0] >= self.config.match_threshold:
                if len(scores) == 1 or scores[0][0] - scores[1][0] >= self.config.min_margin:
                    role = scores[0][1]
            labels[speaker_id] = (role, similarity)

        segments = []
        for turn in sorted(turns, key=lambda turn: turn.start):
            if turn.start < 0 or turn.end > recording.duration + 1 / recording.sample_rate:
                raise ValueError("Diarization turn outside recording")
            role, similarity = labels[turn.speaker_id]
            segments.append(SpeechSegment(
                start_time=turn.start, end_time=min(turn.end, recording.duration),
                audio_path=recording.audio_path, speaker_id=turn.speaker_id,
                speaker_label=role, speaker_similarity=similarity,
                attribution_method="unknown" if role == "unknown" else "voice_match",
                person_id=interlocutor_id if role == "interlocutor" else None,
            ))
        return segments

    def discover_interlocutor(
        self, segments: list[SpeechSegment], speaker_embeddings: dict[str, np.ndarray], *,
        user_profile: VoiceProfile | None, embedding_model: str,
    ) -> DiscoveredVoice | None:
        """Bootstrap one new interlocutor from ordinary speech, without a prompt.

        The caller supplies the active conversation's person ID. Exactly one
        unknown cluster must remain after wearer matching, with a usable clean
        embedding well separated from the wearer's. Several unknown speakers
        cannot be assigned to one person from audio alone.
        """
        if user_profile is None or user_profile.model != embedding_model:
            return None
        unknown = {segment.speaker_id for segment in segments if segment.speaker_label == "unknown"}
        if len(unknown) != 1:
            return None
        speaker_id = next(iter(unknown))
        vector = speaker_embeddings.get(speaker_id)
        if vector is None:
            return None
        profile = VoiceProfile(model=embedding_model, embedding=vector.tolist())
        similarity = float(np.dot(profile.embedding, user_profile.embedding))
        if similarity > self.config.match_threshold - self.config.min_margin:
            return None
        return DiscoveredVoice(speaker_id=speaker_id, profile=profile)
