# Persistent Memory

A memory system for smart-glasses conversations. It records a conversation,
works out who said what, and stores a summary plus facts about the other person
so the wearer can look them up later.

## Status

The pieces exist and are tested individually, but there is **no end-to-end
runnable pipeline yet**. The HTTP API has one feature so far: owner-scoped
person briefings, which show what to remember about someone by face or by name
([contract](docs/contracts/briefings/README.md)). Missing: live microphone
capture and a background worker, camera and audio upload endpoints, conversation
and people read APIs, a UI, and fact ranking. See the open issues for what's next.

## How it fits together

```text
camera ─→ face recognition ─→ person ID ──────────────┐
                                                      ▼
mic ─→ VAD capture ─→ WAV ─→ diarization ─→ voice matching ─→ Whisper ASR
                                                      │
                     Postgres ◀─ LLM summary + facts ◀┘
```

| Component | Where |
| --- | --- |
| Face recognition. Unseen faces become a saved "Unknown person" with a stable ID. The match threshold is `FACE_MATCH_L2_THRESHOLD` ([validation](docs/face_threshold_validation.md)). | `services/face_recog_local.py` |
| Interlocutor tracker. Debounces the face-recognition person ID. | `frontend/interlocutorTracker.ts` |
| Capture (Silero VAD), diarization and voice embeddings (pyannote), speaker matching | `services/audio_pipeline/` |
| Whisper transcription and dialog assembly | `backend/app/services/asr*.py` |
| Recording-to-memory coordinator, retries, voice enrollment | `backend/app/services/audio_memory.py`, `voice_enrollment.py` |
| LLM extraction and the transactional memory write | `backend/app/services/conversation_ingestion.py`, `llm_client.py`, `crud/conversations.py` |
| Person briefings: name, top facts, and last conversation, paged and owner-scoped | `backend/app/services/briefing.py`, `api/routes/briefing.py`, [contract](docs/contracts/briefings/README.md) |
| Owner scoping: `X-Owner-Id` header until real user accounts exist | `backend/app/api/deps.py` |
| Schema and migrations | `backend/app/models/`, `backend/alembic/`, [DB_STRUCTURE.md](backend/DB_STRUCTURE.md) |
| CLI for processing a recorded file | `python -m app.cli.audio --help` |

## Setup

Requires **Python 3.12**: the pinned pydantic and SQLAlchemy have no 3.13
builds, and torch 2.8 has none for 3.14. From the repository root:

```sh
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]" -r backend/requirements.txt
cp backend/.env.example .env
# In .env: DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/persistent_memory
docker compose -f backend/docker-compose.yml up -d db
alembic -c backend/alembic.ini upgrade head
```

Optional components:

- **Audio models:** `pip install -r services/audio_pipeline/requirements_audio_pipeline.txt`.
  Accept the Hugging Face terms for
  [Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) and
  [pyannote/embedding](https://huggingface.co/pyannote/embedding), then set `HF_TOKEN`.
- **Face recognition:** `pip install -r services/requirements_face.txt`
  (needs a C++ compiler for insightface).
- **LLM extraction:** set `OPENAI_API_KEY`.

Run the API and call it as a user:

```sh
uvicorn app.main:app --reload
curl -H "X-Owner-Id: USER_UUID" "localhost:8000/people/briefing?name=Sam"
```

Interactive docs are at `localhost:8000/docs`.

## Tests

```sh
python -m pytest -q                              # no model downloads or paid API calls
node --test frontend/interlocutorTracker.test.mjs   # Node 22.18+
```

Four tests in `services/audio_pipeline/` need the audio requirements installed
(they import torch). `--runslow` adds tests that load real Whisper weights.
