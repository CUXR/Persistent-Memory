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
   pip install -r requirements.txt
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
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
python -m pytest            # from backend/ (or `pytest` from the repo root)
```

`DATABASE_URL` and `OPENAI_API_KEY` are required at startup (see
`backend/.env.example` for every supported variable).

### Backend with Docker
```bash
cd backend
docker compose up --build
```

The API will be available on `http://localhost:8000` and PostgreSQL on `localhost:5432`.

### Services
The `face_recog_local.py` service handles face recognition tasks.

## Dependencies

- **Frontend**: Node.js with TypeScript
- **Backend**: Python 3.8+ with FastAPI
- **Services**: Face recognition libraries (see requirements.txt)
