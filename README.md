# Persistent Memory

A multi-service application combining face recognition, interlocutor tracking, and backend APIs.

## Project Structure

```
Persistent-Memory/
├── frontend/                    # Memory browser web app + TypeScript tooling
│   ├── index.html              # App shell (no build step; ES modules)
│   ├── src/                    # Router, API client, session hook, page views
│   ├── tests/                  # node --test unit tests for the app
│   ├── scripts/dev.mjs         # Static dev server on http://localhost:5173
│   ├── eval/                   # Retrieval eval harness (npm run eval)
│   ├── interlocutorTracker.ts  # Mock interlocutor tracker for EgoMem
│   └── package.json            # Node.js dependencies
│
├── backend/                     # Python FastAPI backend
│   └── app/
│       ├── main.py            # FastAPI application entry point (CORS + routers)
│       ├── api/
│       │   ├── deps.py         # DB session + authenticated owner dependency
│       │   └── routes/
│       │       ├── user.py            # GET /users/me
│       │       ├── conversations.py   # GET /conversations/recent, /conversations/{id}
│       │       └── people.py          # GET /people, /people/{id}/profile, /people/{id}/context
│       ├── core/
│       │   ├── config.py       # Configuration
│       │   └── database.py     # Database setup
│       ├── crud/
│       │   └── user.py         # CRUD operations
│       ├── models/
│       │   └── user.py         # Database models
│       └── schema/
│           └── user.py         # Pydantic schemas
│
├── eval/                        # Offline evaluation (own requirements, own venv)
│   └── retrieval/              # Ranker + query-flow benchmark, synthetic dataset, reports
│
└── services/                    # External services and ML
    └── face_recog_local.py     # Local face recognition service
```

## Development

### Running the memory browser locally (SQLite)

The memory browser is a small read-only web app with two pages, **Recent
Conversations** and **People**, backed by the FastAPI API. Everything below
runs against a local SQLite file; no Postgres or OpenAI access is needed to
browse.

1. Configure and seed the backend (creates the schema, an owner user, two people
   and one conversation):

   ```bash
   cd backend
   cat > .env <<'EOF'
   DATABASE_URL=sqlite+pysqlite:///./dev.sqlite
   OPENAI_API_KEY=unused-for-browsing
   ALLOW_DEV_AUTH_FALLBACK=true
   EOF
   pip install -r requirements-dev.txt
   python -m scripts.seed_memory_store
   ```

2. Start the API on http://localhost:8000:

   ```bash
   python -m uvicorn app.main:app --reload
   ```

3. Start the frontend on http://localhost:5173 (a plain static server; the app
   uses browser ES modules, so there is no build step):

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

`CORS_ALLOWED_ORIGINS` defaults to the two dev origins above; set it
(comma-separated) when the frontend is served from somewhere else. Set
`window.__PERSISTENT_MEMORY_API_BASE_URL__` before `src/main.js` loads to point
the app at a different API base URL.

#### Which owner does the app show?

Memory data is keyed by `owner_user_id`, and every API request is scoped to one
authenticated owner. The backend resolves the owner in this order:

1. `request.state.user_id`, set by the session middleware from the account
   creation work. This is the production path; the frontend sends
   `credentials: "include"` so a session cookie is forwarded.
2. Local development only, when `ALLOW_DEV_AUTH_FALLBACK=true`: the
   `X-User-Id` request header, or the single existing user when the header is
   absent. Unknown or malformed ids are rejected with 401.

On the frontend, `src/session.js` reads the user id from
`window.__PERSISTENT_MEMORY_SESSION__.userId` (the hook for the auth work) and
otherwise from `localStorage["persistent-memory-user-id"]`, which is handy when
the local database has more than one user:

```js
localStorage.setItem("persistent-memory-user-id", "<owner uuid printed by the seed script>");
```

#### API endpoints used by the app

| Endpoint | Purpose |
| --- | --- |
| `GET /users/me` | Authenticated owner record for the session chip |
| `GET /conversations/recent?limit=20` | Recent episodes with timestamps, participants, and summaries |
| `GET /conversations/{episode_id}` | One episode with transcript and participant cards |
| `GET /people?limit=100&query=` | People directory (display name, last seen, top facts, counts); optional resolver-backed search |
| `GET /people/{person_id}/profile` | Full stored profile bundle for one person |
| `GET /people/{person_id}/context?query=` | Query-relevant facts, summaries, and relationships for one person |

All endpoints are read-only (`GET`) and return 404 for records that belong to a
different owner.

### Frontend
```bash
cd frontend
npm install
npm run dev    # static dev server for the memory browser on :5173
npm test       # unit tests (node --test)
npm run eval   # retrieval eval harness
```

### Backend
```bash
cd backend
pip install -r requirements.txt       # runtime only
pip install -r requirements-dev.txt   # runtime + pytest + httpx (TestClient)
python -m uvicorn app.main:app --reload
python -m pytest            # from backend/ (or `pytest` from the repo root)
```

`DATABASE_URL` and `OPENAI_API_KEY` are required at startup; see
`backend/.env.example` for the commonly used variables and
`backend/app/core/config.py` for the full list.

### Backend on Neon Postgres (pgvector)

The shared dev database is a [Neon](https://neon.com) project with two branches:
`dev` (the shared schema) and `eval`, a child of `dev` that holds the retrieval
benchmark's synthetic data and can be reset without touching `dev`.

1. Sign in and create the project (the CLI runs with `npx`; `neon login` opens
   a browser, or set `NEON_API_KEY`):

   ```bash
   npx neonctl@latest login
   npx neonctl@latest projects create --name persistent-memory --output json
   npx neonctl@latest branches rename production dev --project-id <project-id>
   ```

   (A new project's default branch is called `production`; a project that
   signup created for you can be reused with `projects update <id> --name persistent-memory`.)

2. Enable pgvector and run the migrations on the **direct** connection string
   (the host without `-pooler`; PgBouncer's transaction pooling does not support
   the session state migrations need). `DATABASE_URL` uses the psycopg 3 driver
   and requires TLS:

   ```bash
   DEV_URL=$(npx neonctl@latest connection-string dev --project-id <project-id>)
   psql "$DEV_URL" -c 'CREATE EXTENSION IF NOT EXISTS vector;'
   DATABASE_URL="postgresql+psycopg://USER:PASSWORD@ep-xxxx.REGION.aws.neon.tech/neondb?sslmode=require" \
   OPENAI_API_KEY=unused alembic -c backend/alembic.ini upgrade head
   ```

   The same `DATABASE_URL=...?sslmode=require` value goes into `backend/.env`
   for the API. Never commit it.

3. Create the eval branch from `dev` (it inherits the schema and the
   extension) and reset it whenever the benchmark data should start over:

   ```bash
   npx neonctl@latest branches create --name eval --parent dev --project-id <project-id>
   npx neonctl@latest branches reset eval --parent --project-id <project-id>   # back to dev's state
   ```

   The eval branch's direct connection string is `EVAL_DATABASE_URL` in
   `eval/.env` (see `eval/.env.example`).

### Retrieval benchmark (`eval/`)

`eval/retrieval` benchmarks which ranker best selects past conversations for a
query (bge-m3 alone, bge-m3 → bge-reranker-v2-m3, bge-m3 → Jev, Jev over the
full pool) and how well the whole query flow (extraction → `PersonResolver` →
padded time filter → ranking → summary-first answering with dialogue fallback)
answers. Its dependencies stay out of `backend/requirements.txt`; the backend
test suite never needs them and `python -m pytest -q` downloads nothing.

```bash
python -m venv eval/.venv && eval/.venv/bin/pip install -r eval/requirements.txt
cp eval/.env.example eval/.env      # OPENAI_API_KEY (+ EVAL_DATABASE_URL, unless you run with --store memory)
```

OpenAI is the only paid service it needs (`EVAL_GATEWAY=openai`, the default):
it writes the dataset text, runs the system under test and judges the answers,
up to `EVAL_MAX_SPEND_USD` per command. The two local rankers (`bge_m3`,
`bge_m3_reranker`) cost nothing. The two Jev rankers are not served by OpenAI:
they run only with `EVAL_GATEWAY=openrouter` or `vercel` and that router's own
key, and are left out of every command otherwise.

The dataset in `eval/retrieval/data/` (one owner, 40 people, ~800 conversations
with dialogue and summary, 300 labelled queries) and, once the Jev study has run,
`eval/retrieval/jev_scoring.json` (the pinned Jev version and how it is applied)
are committed, so the usual entry point is the runner:

```bash
eval/.venv/bin/python -m eval.retrieval.data.load_db --reset   # corpus, bge-m3 embeddings and a copy of the queries (eval_queries) into the eval branch
eval/.venv/bin/python -m eval.retrieval.run --split test       # ranker comparison + system eval
```

It writes `eval/retrieval/results/<date>.md` and `<date>.json`, resumes from
`eval/retrieval/.cache/` after an interruption (re-run the printed command), and
stops at `EVAL_MAX_SPEND_USD`. `--stage rankers|system|report`, `--rankers`,
`--limit N` and `--store memory` (no database; embeds the corpus locally) narrow a
run.

Rebuilding the dataset from the seed, in order:

```bash
python -m eval.retrieval.data.generate --smoke          # optional: six people, to read the prompts' output
python -m eval.retrieval.data.generate                  # people.jsonl + conversations.jsonl
python -m eval.retrieval.data.queries                   # queries.jsonl (30 dev / 270 test)
python -m eval.retrieval.data.load_db --reset
python -m eval.retrieval.data.show_queries --split dev  # read the dev queries by hand
python -m eval.retrieval.jev_study --split dev --pin    # router gateways only: how to apply Jev; writes jev_scoring.json
python -m eval.retrieval.data.pool_labels               # top-30 pooling + relevance judge completes gold labels
#   exit code 2 lists "unanswerable" queries that are answerable: run the printed
#   `queries --regenerate ...` command and pool again until it exits 0
python -m eval.retrieval.run --split dev                # dry run
python -m eval.retrieval.run --split test
python -m eval.retrieval.judge_validation sample --records eval/retrieval/.cache/runs/<date>-test/system_records.jsonl --out sample.json
#   label sample.json by hand, then:
python -m eval.retrieval.judge_validation score --sample sample.json --labels labels.json
python -m eval.retrieval.run --split test --stage report
```

Structure is deterministic from `eval/retrieval/config.py` (`SEED`, the fixed
clock `NOW`); LLM text is reproduced from the response cache. Run the offline
tests with `python -m pytest -q eval/tests` from the repo root.

### Backend with Docker
```bash
cd backend
docker compose up --build
```

The API will be available on `http://localhost:8000` and PostgreSQL on `localhost:5432`.
The compose stack does not create the schema or any data, so on a fresh volume
run the migrations and (optionally) the seed script before opening the app:

```bash
docker compose exec api alembic upgrade head
docker compose exec api python -m scripts.seed_memory_store
```

### Services
The `face_recog_local.py` service handles face recognition tasks.

## Dependencies

- **Frontend**: Node.js with TypeScript
- **Backend**: Python 3.10+ with FastAPI (compose uses 3.12)
- **Services**: Face recognition libraries (see requirements.txt)
