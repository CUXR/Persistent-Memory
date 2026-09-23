# Persistent Memory

A long-term social memory system for smart glasses. The goal: when the
wearer meets someone, recognize who they are, recall what's known about
them from past encounters, and capture the current conversation to grow
that memory over time.

```
face recognition → identity → load memory
        +
audio capture → VAD/diarization → ASR → LLM fact extraction → memory store
```

## Status

This is an early-stage, actively-developed monorepo. Not everything
described below is wired end-to-end yet — the sections below are marked
with what's actually implemented versus scaffolding.

| Area | Status |
| --- | --- |
| Audio pipeline: VAD + speaker diarization | Implemented (`services/audio_pipeline/`) |
| ASR + dialog assembly | Implemented (`backend/app/services/asr*.py`) |
| Conversation ingestion → LLM fact/summary extraction → memory store | Implemented (`backend/app/services/conversation_ingestion.py`) |
| Person resolution (name → `Person` record) | Implemented (`backend/app/crud/person_resolver.py`) |
| Local face recognition (InsightFace + FAISS) | Implemented as a standalone script (`services/face_recog_local.py`) |
| Interlocutor tracker (glasses-side "who's talking to me now") | Mock only (`frontend/interlocutorTracker.ts`) |
| Auth, login, onboarding, voice enrollment | Not started |
| Frontend app (conversations / people views) | Not started — `frontend/` currently has no app, only the mock tracker |
| Semantic/vector retrieval over facts | Extension point defined (`EmbeddingProvider` protocol), no implementation shipped |

## Project Structure

```
Persistent-Memory/
├── frontend/
│   ├── interlocutorTracker.ts   # Mock interlocutor tracker (polling-based identity switch)
│   └── package.json             # No app scaffolding yet
│
├── backend/                     # FastAPI application
│   ├── app/
│   │   ├── main.py              # FastAPI entrypoint (mounts the users router)
│   │   ├── api/
│   │   │   ├── deps.py
│   │   │   └── routes/user.py   # Empty router stub — no endpoints defined yet
│   │   ├── core/
│   │   │   ├── config.py        # Settings (env-driven: DB URL, OpenAI key/model, embedding dim)
│   │   │   └── database.py      # SQLAlchemy engine/session, declarative base
│   │   ├── crud/
│   │   │   ├── memory_store.py      # Reads/writes people, episodes, facts, summaries, edges
│   │   │   ├── person_resolver.py   # Resolves a name mention in text to a Person
│   │   │   └── user.py
│   │   ├── models/               # SQLAlchemy models: User, Person, Episode, Alias,
│   │   │   │                     #   PersonFact, Pref, Summary, Edge, EpisodeParticipant
│   │   ├── schema/                # Pydantic schemas mirroring the models + ASR/ingestion I/O
│   │   └── services/
│   │       ├── asr.py             # Orchestrates transcription + dialog assembly (issue #24)
│   │       ├── asr_engine.py      # faster-whisper wrapper
│   │       ├── asr_assembly.py    # Filtering, turn-merging, text normalization
│   │       ├── conversation_ingestion.py  # transcript → LLM summary + fact extraction → store
│   │       ├── llm_client.py      # OpenAI wrapper (structured-output calls)
│   │       └── embedding.py       # EmbeddingProvider protocol (no implementation yet)
│   ├── alembic/                   # DB migrations
│   ├── scripts/                   # seed_memory_store.py, live_ingestion.py
│   ├── tests/
│   ├── DB_STRUCTURE.md            # Schema notes (partial — see models/ for the full set of tables)
│   ├── docker-compose.yml         # Postgres + API, both containerized
│   └── requirements.txt
│
└── services/
    ├── face_recog_local.py        # Standalone InsightFace + FAISS face-recognition script
    └── audio_pipeline/            # Streaming VAD → diarization → speaker attribution (issue #23)
        ├── vad.py                 # Silero VAD gating
        ├── diarization.py         # pyannote speaker-diarization-community-1 wrapper
        ├── speaker_attribution.py # Cosine similarity vs. enrolled voice embedding → user/interlocutor/uncertain
        ├── ingestion.py           # Rolling-window state machine (IDLE → ACCUMULATING → PROCESSING)
        ├── segment.py             # AudioSegment output contract
        └── test_audio_pipeline.py
```

## How the pieces fit together

1. **Audio capture** streams into `AudioIngestionPipeline` (`services/audio_pipeline/ingestion.py`).
   A rolling 5-second window is screened by Silero VAD; once speech is
   detected, audio accumulates into a single conversation buffer (capped at
   ~10 minutes) until silence or the cap ends the conversation.
2. **Diarization** (`diarization.py`, pyannote `speaker-diarization-community-1`)
   splits that buffer into speaker turns and extracts a per-speaker embedding.
3. **Speaker attribution** (`speaker_attribution.py`) compares each speaker's
   embedding against the wearer's enrolled `Person.voice_embedding` via
   cosine similarity and labels turns `user`, `interlocutor`, or `uncertain`.
4. **ASR** (`backend/app/services/asr.py`) transcribes each labeled segment
   with `faster-whisper`, filters silence, merges adjacent same-speaker
   segments, and normalizes text into a `Dialog` of speaker-attributed turns.
5. **Conversation ingestion** (`conversation_ingestion.py`) takes a
   transcript, resolves the interlocutor to a `Person` record, and makes
   concurrent LLM calls (via `llm_client.py`, OpenAI structured outputs) to
   produce an episode summary and extract new facts (`visual_descriptor`,
   `affiliation`, `hobby`), deduplicating against what's already stored.
6. **Memory store** (`crud/memory_store.py`) persists people, episodes,