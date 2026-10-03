# Persistent Memory

A social memory system for documenting conversations. The audio path captures a
conversation, recognizes the wearer and learns the interlocutor's voice, builds a structured
dialog, and stores an episode, summary, and facts about the interlocutor.

```text
16 kHz mono stream → 10-second Silero VAD checks → retained WAV + manifest
    → Community-1 diarization → match wearer / learn or match partner → Whisper ASR
    → structured dialog → validated LLM extraction → atomic memory commit
```

## Setup

Use Python 3.12 and run these commands from the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e . -r backend/requirements.txt
pip install -r services/audio_pipeline/requirements_audio_pipeline.txt
cp backend/.env.example .env
```

Set `DATABASE_URL` in `.env`. For local PostgreSQL with pgvector:

```sh
docker compose -f backend/docker-compose.yml up -d db
# Local URL: postgresql+psycopg://postgres:postgres@localhost:5432/persistent_memory
alembic -c backend/alembic.ini upgrade head
python backend/scripts/seed_memory_store.py
```

The optional seed command creates demo memories and prints the owner and person
UUIDs used in the examples below. It expects migrations to have run first.

For audio processing, accept the Hugging Face access terms for
[Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) and
[pyannote/embedding](https://huggingface.co/pyannote/embedding), and set `HF_TOKEN`.
The requirements pin compatible PyTorch, TorchAudio, and TorchCodec releases.
TorchCodec 0.7 needs FFmpeg 4–7 shared libraries for its file decoder. This pipeline
loads files with SoundFile and passes in-memory waveforms to pyannote, so it also
works when pyannote warns that its built-in decoder is unavailable.
Model weights load lazily; the first use downloads them. Set `OPENAI_API_KEY` for memory extraction;
enrollment and the capture component do not need it. The default Whisper model
is `small`, with English transcription and int8 compute. Pyannote uses CPU by
default; its Python constructor supports CUDA explicitly.

### Existing databases

`0001_baseline` describes the original schema; `0002_audio_memory` adds wearer
voice columns and audio jobs; `0003_voice_discovery` records which conversation
first supplied a partner's voice. For an existing unversioned database created by
`create_all`, back it up and verify that its schema matches `0001_baseline`
before running:

```sh
alembic -c backend/alembic.ini stamp 0001_baseline
alembic -c backend/alembic.ini upgrade head
```

Do not stamp an empty or different schema. The baseline uses the original default
512-dimensional vectors; databases with custom vector dimensions need a tailored
migration. Existing partner voice vectors are preserved. Missing model metadata,
incompatible models, or incompatible vector lengths require re-enrollment.

## Enroll and process audio

The CLI is a trusted local operator interface. `--owner-id` selects an existing
user; it is not an authentication credential. The face tracker creates an owned
person record even when their name is unknown, then supplies that person's UUID
to audio processing. Audio never creates a separate person record.

```sh
# Enroll the wearer during setup. The other person needs no enrollment prompt.
python -m app.cli.audio --owner-id USER_UUID enroll wearer.wav

# Import a recording using its actual start time, including timezone.
python -m app.cli.audio --owner-id USER_UUID capture-file conversation.wav \
  --person-id PERSON_UUID --started-at 2026-09-26T14:00:00-04:00

python -m app.cli.audio --owner-id USER_UUID status RECORDING_UUID
python -m app.cli.audio --owner-id USER_UUID retry RECORDING_UUID

# Recover a saved manifest that was not registered before the process stopped.
python -m app.cli.audio --owner-id USER_UUID process recordings/RECORDING_UUID.json \
  --person-id PERSON_UUID
```

Wearer enrollment accepts at most two minutes of audio and requires at least five
seconds of clear speech from exactly one diarized speaker. File imports downmix
stereo and resample to 16 kHz. The enrollment command also supports `--person-id`
for an optional manual replacement; normal conversations do not require it.

For a person with no voice profile, processing computes a voice embedding from
their normal speech and attaches it to the face tracker's person ID. It requires
one unknown speaker after wearer matching and enough clean audio to extract a
vector (currently at least one non-overlapping turn of one second). The wearer
does not have to speak in that same recording, but their saved profile is required
to rule out their voice. With the default settings, the candidate's similarity
to the wearer must be at most `0.60` (`threshold - margin`). Multiple unknown
speakers, a missing embedding, or a borderline wearer match remain unresolved.

The first profile is saved with its source job before ASR/LLM work, so provider
failure does not lose it. Existing partner profiles are matched, never silently
overwritten. First-conversation turns use `attribution_method="conversation"`
with no identity similarity score; later conversations use `voice_match`.

`capture-file` saves all finalized recordings and registers their jobs before
running the expensive models. It prints job JSON, including structured dialog and
the ingestion result. A job with `needs_review` retains its transcript but creates
no memories. After resolving the ambiguity or providing a usable profile from
another conversation, `retry` reruns processing. Overlapping
speech may still require review; there is no manual transcript-editing UI yet.

If a worker was interrupted while a job was `processing`, stop that worker and use
`retry RECORDING_UUID --recover-interrupted`. Each claim gets a new attempt ID;
an older worker cannot commit after recovery. Completed jobs return their saved
result without creating another episode. Failed jobs report the failing stage.

## Pipeline behavior and decisions

| Area | Implementation |
| --- | --- |
| Capture | `AudioIngestionPipeline.push_audio()` accepts arbitrary chunks of finite mono float32 PCM at 16 kHz. It checks non-overlapping 10-second windows, retaining the whole window that first contains speech. |
| Conversation boundary | One fully silent window ends a conversation; continuous audio splits at 600 seconds. `flush()` checks any partial window, including speech shorter than 10 seconds. These limits are constructor settings. |
| Persistence | Finalization writes a float WAV and JSON manifest before clearing the buffer. Each recording has a UUID, absolute start time, and sample count. Segment offsets always refer to that same WAV. |
| Diarization | Community-1 runs once on a finalized recording. Regular diarization preserves overlap so cross-talk can be detected. It is not called for every VAD window, and speaker count is not forced to two. |
| Voice embeddings | Both `User` and `Person` store normalized 512-dimensional `pyannote/embedding` vectors plus their model identifier. Embeddings average eligible turns of at least one second, weighted by duration; turns overlapping another speaker are excluded. Face-vector configuration is independent. |
| Matching | Compare each diarized speaker against both available profiles. Default cosine similarity must be at least `0.75`, with a lead of at least `0.15` over the other profile when both exist. Configure `VOICE_MATCH_THRESHOLD` and `VOICE_MATCH_MARGIN`. These are starting thresholds to calibrate with real recordings. |
| New interlocutor | When the face-selected person has no profile, learn the sole unknown speaker's embedding if it is clearly distinct from the wearer. Save it under that existing person ID, without a spoken enrollment prompt. |
| Ambiguous speakers | Several unknown speakers or an inconclusive wearer comparison remain `unknown`. Existing profiles must use the configured model and are not replaced automatically. |
| ASR and dialog | Faster Whisper transcribes the actual waveform slice for each turn. Dialog retains speaker ID, role, identified person ID, similarity, timestamps, text, and ASR confidence. Adjacent turns merge only when speaker identity and source file match. Similarity and ASR confidence are separate measurements. |
| Memory policy | Any ASR exception fails the job. Unknown speakers, overlapping speech, or no confirmed interlocutor require review. Eligible dialogs become named transcripts for summary and fact extraction. |
| Commit and retry | The episode, summary, deduplicated facts, relationships, and successful job result commit in one transaction. A failure cannot leave a partially written episode. Audio jobs are owner-scoped and claimed atomically. |

The stream clock advances through silence and across recording splits. Capture
and ML processing are separate components: a live caller should enqueue returned
recordings and keep feeding capture while another worker calls
`AudioMemoryService.process(recording, person_id)`. This repository does not yet
include a microphone transport or background queue. The local file CLI processes
sequentially. Unfinalized audio remains in memory and is vulnerable to process
termination; finalized recordings remain on disk for replay. Recordings are not
automatically deleted, so deployment needs a retention policy.

## Face identity and the audio handoff

`services/face_recog_local.py` uses the existing InsightFace/FAISS recognition
path. An unseen face with detection score at least `0.7` is saved automatically
with a UUID, face embedding, and an `Unknown person <UUID>` display placeholder.
The index updates immediately, so matching subsequent frames reuse the same ID;
restarting loads those IDs from the database. Naming the person later must update
that record, keeping the UUID and voice profile together. Face matching is
owner-scoped and restricted to the same embedding model. A face matches when its
squared L2 distance is at most `FACE_MATCH_L2_THRESHOLD` (default `1.5`, validated
on LFW; see [docs/face_threshold_validation.md](docs/face_threshold_validation.md)).

With the face dependencies (InsightFace, ONNX Runtime, FAISS, OpenCV) in your venv:

```sh
python services/face_recog_local.py --owner-id USER_UUID
```

There is no manual face-enrollment prompt. `recognize_faces(frame)` returns IDs
for all recognized or newly created faces. `get_current_person_id(frame)` returns
the ID when exactly one face is visible and `None` when selection is ambiguous.
The frontend `InterlocutorTracker` now requires a `getCurrentPersonId` adapter
instead of inventing IDs from a hard-coded timeline. The adapter must return
these persisted IDs; profile-context and wearer-state adapters are also explicit.

Pass the selected ID to `AudioMemoryService.process(recording, person_id)`.
On an identity switch, flush and submit the previous person's capture before
starting the new person's buffer. Do not assign the latest observed face to an
entire recording spanning multiple people. Camera-to-browser transport and a
live capture worker are still application integration work; the Python
recognizer, frontend source interface, and backend handoff are independently
callable and tested.

## Code map and remaining scope

- `services/audio_pipeline/`: capture, normalization, Silero VAD, pyannote adapters,
  matching, and the shared `Recording`, `VoiceProfile`, and `SpeechSegment` contracts.
- `backend/app/services/voice_enrollment.py`: validate and enroll either speaker.
- `backend/app/services/audio_memory.py`: recording-to-memory coordinator.
- `backend/app/services/asr*.py`: transcription and dialog assembly.
- `backend/app/crud/voice.py`, `audio_jobs.py`, `conversations.py`: owned profiles,
  durable jobs, and transactional memory writes.
- `backend/app/cli/audio.py`: enrollment, import, status, and recovery commands.
- `backend/alembic/versions/`: baseline and audio migrations.
- [Database structure](backend/DB_STRUCTURE.md): current storage contracts.

Text ingestion resolves people by their single name; audio ingestion uses the
face tracker's person's UUID. The face recognizer automatically creates unknown
people, and the frontend tracker accepts a real identity source. Authenticated HTTP
enrollment, login/onboarding, a conversations UI, and semantic retrieval are not
implemented. The FastAPI users router remains a stub. The unused text-embedding
hook was removed; no semantic vectors were being persisted by it.

## Tests

Inside the venv after installing the dependencies above:

```sh
pip install pytest
python -m pytest -q
# Frontend tracker tests use Node with native TypeScript support.
node --test frontend/interlocutorTracker.test.mjs
```

The default tests use real WAV files and SQLite storage with deterministic model
substitutes. They exercise automatic face creation, face-to-audio identity handoff,
voice discovery and enrollment, matching, capture boundaries, ASR
slices, identity preservation, failures, retry fencing, atomic commits, and real
Alembic upgrades/downgrades. PostgreSQL migration SQL is compiled offline.
`--runslow` enables optional real Whisper tests. Default tests neither download
model weights nor call paid LLM APIs; they do not establish real-world diarization
or voice-matching accuracy.
