# Persistent Memory

A multi-service application combining face recognition, interlocutor tracking, and backend APIs.

## Project Structure

```
Persistent-Memory/
├── frontend/                    # TypeScript frontend code
│   ├── interlocutorTracker.ts  # Mock interlocutor tracker for EgoMem
│   └── package.json            # Node.js dependencies
│
├── backend/                     # Python FastAPI backend
│   └── app/
│       ├── main.py            # FastAPI application entry point
│       ├── api/
│       │   ├── deps.py         # Dependency injection
│       │   └── routes/
│       │       └── user.py     # User endpoints
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

### Frontend
```bash
cd frontend
npm install
npm run dev  # or appropriate dev command
```

### Backend
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

### Services
The `face_recog_local.py` service handles face recognition tasks.

## Dependencies

- **Frontend**: Node.js with TypeScript
- **Backend**: Python 3.8+ with FastAPI
- **Services**: Face recognition libraries (see requirements.txt)
