# Homework Helper POC

Minimal, subject/level-aware homework tutor with DB-backed chat history and OpenAI Responses API.

## Features
- Level selector: Kindergarten / Lower Secondary / Upper Secondary
- Subject selector: Math / Science / English
- Backend presets enforce style + safe temperature (math is low temp)
- Chat history persisted per user & chat
- Simple frontend (no build step)

## Quick Start

### 1) Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # add your OpenAI key
python run.py
```

- Server runs at `http://127.0.0.1:5000`
- Endpoints: `/ask`, `/chats`, `/history`

### 2) Frontend (static)
Open `frontend/index.html` directly, or serve with a simple server:
```bash
cd frontend
python -m http.server 5173
```
- If you change the port/origin, set `ALLOWED_ORIGIN` in backend `.env` accordingly.

## Environment Variables
- `OPENAI_API_KEY` — your key
- `OPENAI_DEFAULT_MODEL` — default model (e.g. `gpt-4.1-mini`)
- `DATABASE_URL` — default `sqlite:///homework.db`
- `ALLOWED_ORIGIN` — e.g. `http://localhost:5173`

## Notes
- Some models reject `temperature`. The backend retries without it.
- This POC does not include RAG; you can add it later behind the same API.
- SQLite is fine for development; swap to Postgres by changing `DATABASE_URL`.

## License
MIT (for your internal POC use)
