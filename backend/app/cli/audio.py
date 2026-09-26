"""Enroll voices, capture a file, process a manifest, or retry a retained job."""
import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from uuid import UUID

from audio_pipeline.audio import load_audio
from audio_pipeline.diarization import DiarizationEngine
from audio_pipeline.ingestion import AudioIngestionPipeline
from audio_pipeline.segment import Recording, SAMPLE_RATE
from audio_pipeline.speaker_attribution import AttributionConfig, SpeakerAttributor
from audio_pipeline.vad import SileroVAD

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-id", type=UUID, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    enroll = sub.add_parser("enroll")
    enroll.add_argument("audio", type=Path)
    enroll.add_argument("--person-id", type=UUID, help="Omit to enroll the wearer")
    capture = sub.add_parser("capture-file")
    capture.add_argument("audio", type=Path)
    capture.add_argument("--started-at", type=datetime.fromisoformat, required=True,
                         help="Actual recording start, including timezone")
    capture.add_argument("--person-id", type=UUID, required=True, help="Person ID supplied by the face tracker")
    process = sub.add_parser("process")
    process.add_argument("manifest", type=Path)
    process.add_argument("--person-id", type=UUID, required=True, help="Person ID supplied by the face tracker")
    retry = sub.add_parser("retry")
    retry.add_argument("job_id", type=UUID)
    retry.add_argument("--recover-interrupted", action="store_true",
                       help="Recover a job whose previous worker was stopped")
    status = sub.add_parser("status")
    status.add_argument("job_id", type=UUID)
    args = parser.parse_args()
    # Allow --help without database settings or application model initialization.
    from ..core.config import get_settings
    from ..crud.audio_jobs import AudioJobStore
    from ..crud.memory_store import MemoryStore
    from ..crud.voice import VoiceStore
    from ..services.asr_engine import WhisperEngine
    from ..services.audio_memory import AudioMemoryService
    from ..services.voice_enrollment import VoiceEnrollmentService

    settings = get_settings()
    store = MemoryStore(owner_user_id=args.owner_id)
    store.initialize(create_schema=False)
    try:
        jobs = AudioJobStore(store)
        if args.command == "status":
            print(jobs.get(args.job_id).model_dump_json(indent=2))
            return
        if not settings.hf_token:
            parser.error("HF_TOKEN is required for audio model processing")
        diarizer = DiarizationEngine(settings.hf_token)
        if args.command == "enroll":
            profile = VoiceEnrollmentService(VoiceStore(store), diarizer).enroll(
                args.audio, person_id=args.person_id,
            )
            print(f"Enrolled {args.person_id or args.owner_id}: {profile.model}, {len(profile.embedding)} dimensions")
            return
        service = AudioMemoryService(
            store, diarizer, WhisperEngine(), attributor=SpeakerAttributor(AttributionConfig(
                match_threshold=settings.voice_match_threshold, min_margin=settings.voice_match_margin,
            )),
        )
        if args.command == "retry":
            job = jobs.get(args.job_id)
            result = asyncio.run(service.process(job.recording, job.person_id, retry=True,
                                                recover_interrupted=args.recover_interrupted))
            print(result.model_dump_json(indent=2))
            if result.status == "failed":
                raise SystemExit(1)
            return
        if args.command == "process":
            recordings = [Recording.model_validate_json(args.manifest.read_text())]
        else:
            # Validate person ownership before retaining any capture.
            store.get_person(args.person_id)
            audio = load_audio(args.audio)
            pipeline = AudioIngestionPipeline(SileroVAD(), settings.recording_directory, args.started_at)
            recordings = []
            for offset in range(0, len(audio), SAMPLE_RATE):
                recordings.extend(pipeline.push_audio(audio[offset:offset + SAMPLE_RATE]))
            recordings.extend(pipeline.flush())
            # Register the complete capture before any costly processing starts.
            for recording in recordings:
                jobs.register(recording, args.person_id)
        failed = False
        for recording in recordings:
            result = asyncio.run(service.process(recording, args.person_id))
            print(result.model_dump_json(indent=2))
            failed |= result.status == "failed"
        if failed:
            raise SystemExit(1)
    finally:
        store.close()


if __name__ == "__main__":
    main()
