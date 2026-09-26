# Database structure

SQLAlchemy models in `app/models/` are the application schema. Alembic revisions
`0001_baseline`, `0002_audio_memory`, and `0003_voice_discovery` establish the schema
and add audio support and voice-discovery provenance. See the root README for
the unversioned-database upgrade procedure.

## People and voice profiles

`users` identifies the wearer. `people.user_id` identifies the owner of an
interlocutor record. Both tables have names and timestamps, plus nullable
`voice_embedding` and `voice_embedding_model` columns. Wearer enrollment and
automatic interlocutor discovery each write a
normalized 512-dimensional `pyannote/embedding` vector and its model ID together.
PostgreSQL uses `vector(512)`; SQLite tests use JSON arrays. Missing profiles are
allowed. The face recognizer creates a person with a stable UUID and placeholder
name for a new face; audio then fills that same person's missing voice profile
from ordinary speech. Audio never creates its own person records.

People additionally have face vectors and model metadata, `last_seen_at`, and
legacy `face_key`, `voice_key`, and `persona90` fields. The audio path uses the
actual voice vector, never `voice_key`. Face dimensions remain configurable;
voice dimensions are fixed by the selected embedding model. Existing voice
vectors with missing or incompatible model metadata need re-enrollment.

## Retained audio jobs

`audio_jobs.id` is also the recording UUID. A job belongs to a `user_id` and one
selected `person_id`; neither can be reassigned by registering the same ID again.

| Column | Meaning |
| --- | --- |
| `recording` | JSON manifest: UUID, absolute WAV path, timezone-aware recording start, 16 kHz sample rate, sample count. Audio bytes remain on disk. |
| `status` | `pending`, `processing`, `complete`, `failed`, `needs_review`, or `no_speech`. |
| `stage`, `error` | Current/failed processing stage and diagnostic message. |
| `attempt_id` | UUID of the worker claim; stale workers cannot update or commit the job. |
| `dialog` | Structured turns and per-segment transcription errors. Turns retain diarization speaker ID, role, person ID when identified, similarity, recording-relative times, text, and ASR confidence. |
| `voice_discovery` | Speaker ID and profile first learned from this recording. Saved atomically with the person's initial voice profile and preserved across retries. |
| `transcript` | Readable named dialog, retained even when review is required. |
| `result`, `episode_id` | Committed ingestion result and episode reference; used for idempotent retries. |

When exactly one unknown speaker remains after wearer matching, a missing partner
profile can be learned from clean speech clearly distinct from the wearer. It is
attached to the person ID already supplied by the face tracker. Existing profiles
are not overwritten, and stale job attempts cannot save a profile. Discovery
turns carry `attribution_method="conversation"` and no independent voice-match
score; later recordings can use `voice_match` against that stored profile.

Jobs with unresolved identities, cross-talk, or no confirmed interlocutor are
retained for review without LLM memory extraction. Failed jobs keep their audio
and any assembled dialog. Terminal successful jobs are not reprocessed. Job
foreign keys prevent silently deleting referenced users, people, or episodes;
any future deletion workflow must handle the associated job and file explicitly.

## Conversation memory

- `episodes`: owner, primary interlocutor, start/end, complete transcript,
  summary, and importance score.
- `episode_participants`: episode/person association.
- `person_summaries`: interlocutor summary with source episode and time range.
- `person_facts`: fact text, category (`visual_descriptor`, `affiliation`, or
  `hobby`), confidence, and source episode; optional source and validity dates.
- `person_edges`: directed relation between two known people, confidence, and
  source episode, unique by source/relation/destination.
- `user_facts`: stored wearer facts supplied as context during extraction.

The conversation writer commits all extracted memory rows together with the
audio job's completion. Fact deduplication normalizes case and whitespace.
Relationships resolve existing people by name and require the same owner.
Older recordings do not move `last_seen_at` backwards.

`person_aliases` and `person_prefs` remain legacy tables; name resolution and the
audio ingestion path do not use them. This migration preserves existing data
rather than dropping unrelated tables.
