# MemoryChat: Conversational AI API with Persistent Memory

A FastAPI `/chat` endpoint that sends multi-turn conversations to an LLM (Google Gemini) and stores every thread in a database, so context survives restarts.

**Stack:** Python, FastAPI, Pydantic, SQLAlchemy, PostgreSQL (SQLite for quick local runs), Docker

## Features
- `POST /chat` – send a message; omit `session_id` to start a new thread, pass it to continue one
- `GET /sessions/{id}` – replay the stored conversation
- `DELETE /sessions/{id}` – remove a thread
- Request/response validation with Pydantic
- Message-history windowing (`MAX_HISTORY_MESSAGES`) to stay inside the context window
- System prompt with rules and a few-shot example; `TEMPERATURE` is configurable
- LLM failures return a clean `502` and roll back the stored user message

## Run locally
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your free key from https://aistudio.google.com/apikey
uvicorn app.main:app --reload
```
Open http://localhost:8000/docs to try it.

```bash
curl -X POST localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message": "My name is Sai. Remember it."}'
# then reuse the returned session_id:
curl -X POST localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message": "What is my name?", "session_id": "<id>"}'
```

## Run with PostgreSQL
```bash
export GEMINI_API_KEY=your-key
docker compose up --build
```

## Tests
```bash
pytest
```
Tests mock the LLM, so no API key is needed.
