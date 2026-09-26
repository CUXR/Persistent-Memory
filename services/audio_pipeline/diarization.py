"""Lazy pyannote.audio 4.x adapters. Enrollment and attribution share one model."""
from dataclasses import dataclass

import numpy as np

from .segment import SAMPLE_RATE, VOICE_MODEL, VoiceProfile, validate_audio


@dataclass(frozen=True)
class DiarizedTurn:
    start: float
    end: float
    speaker_id: str

    @property
    def duration(self):
        return self.end - self.start


class DiarizationEngine:
    embedding_model = VOICE_MODEL

    def __init__(self, hf_token: str, use_gpu: bool = False):
        if not hf_token:
            raise ValueError("A Hugging Face token is required")
        self._token = hf_token
        self._use_gpu = use_gpu
        self._pipeline = None
        self._embedding = None

    def _ensure_pipeline(self):
        if self._pipeline is None:
            import torch
            from pyannote.audio import Pipeline
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1", token=self._token,
            )
            if self._use_gpu:
                self._pipeline.to(torch.device("cuda"))
        return self._pipeline

    def _ensure_embedding(self):
        if self._embedding is None:
            import torch
            from pyannote.audio import Inference, Model
            model = Model.from_pretrained(self.embedding_model, token=self._token)
            self._embedding = Inference(
                model, window="whole", device=torch.device("cuda" if self._use_gpu else "cpu"),
            )
        return self._embedding

    def diarize(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> list[DiarizedTurn]:
        audio = validate_audio(audio, sample_rate)
        if len(audio) == 0:
            return []
        import torch
        output = self._ensure_pipeline()({
            "waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": sample_rate,
        })
        # Preserve overlaps so enrollment and automatic memory can reject cross-talk.
        return sorted([
            DiarizedTurn(segment.start, segment.end, speaker)
            for segment, _, speaker in output.speaker_diarization.itertracks(yield_label=True)
            if segment.end > segment.start
        ], key=lambda turn: turn.start)

    def extract_speaker_embedding(self, audio, start_sec, end_sec, sample_rate=SAMPLE_RATE):
        audio = validate_audio(audio, sample_rate)
        if not 0 <= start_sec < end_sec <= len(audio) / sample_rate + 1 / sample_rate:
            raise ValueError("Embedding crop outside recording")
        import torch
        from pyannote.core import Segment
        vector = self._ensure_embedding().crop(
            {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": sample_rate},
            Segment(start_sec, min(end_sec, len(audio) / sample_rate)),
        )
        return np.asarray(VoiceProfile(
            model=self.embedding_model, embedding=np.asarray(vector).flatten().tolist(),
        ).embedding, dtype=np.float32)

    def extract_per_speaker_embeddings(self, audio, turns, sample_rate=SAMPLE_RATE, min_duration=1.0):
        """Average sufficiently long, non-overlapping turns; never embed cross-talk."""
        grouped = {}
        for turn in turns:
            overlaps = any(
                other.speaker_id != turn.speaker_id and other.start < turn.end and turn.start < other.end
                for other in turns
            )
            if turn.duration < min_duration or overlaps:
                continue
            vector = self.extract_speaker_embedding(audio, turn.start, turn.end, sample_rate)
            grouped.setdefault(turn.speaker_id, []).append((vector, turn.duration))
        return {
            speaker: np.asarray(VoiceProfile(
                model=self.embedding_model,
                embedding=np.average([v for v, _ in samples], axis=0,
                                     weights=[duration for _, duration in samples]).tolist(),
            ).embedding, dtype=np.float32)
            for speaker, samples in grouped.items()
        }
