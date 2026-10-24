from typing import Callable, Dict, List

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import llm
from .config import settings
from .db import Base, engine, get_db
from .models import Conversation, Message
from .schemas import ChatRequest, ChatResponse, HistoryResponse

Base.metadata.create_all(bind=engine)

app = FastAPI(title="MemoryChat API")


def get_llm() -> Callable[[List[Dict[str, str]]], str]:
    return llm.generate_reply


def _window(conv: Conversation) -> List[Dict[str, str]]:
    """Most recent N messages only, so long chats stay inside the context window."""
    recent = conv.messages[-settings.max_history_messages :]
    return [{"role": m.role, "content": m.content} for m in recent]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db), generate=Depends(get_llm)):
    if req.session_id:
        conv = db.get(Conversation, req.session_id)
        if conv is None:
            raise HTTPException(404, "Unknown session_id")
    else:
        conv = Conversation()
        db.add(conv)
        db.flush()

    conv.messages.append(Message(role="user", content=req.message))
    db.flush()
    context = _window(conv)

    try:
        reply = generate(context)
    except llm.LLMError as exc:
        db.rollback()
        raise HTTPException(502, str(exc))

    conv.messages.append(Message(role="assistant", content=reply))
    db.commit()
    return ChatResponse(session_id=conv.id, reply=reply, messages_in_context=len(context))


@app.get("/sessions/{session_id}", response_model=HistoryResponse)
def history(session_id: str, db: Session = Depends(get_db)):
    conv = db.get(Conversation, session_id)
    if conv is None:
        raise HTTPException(404, "Unknown session_id")
    return HistoryResponse(session_id=conv.id, messages=conv.messages)


@app.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str, db: Session = Depends(get_db)):
    conv = db.get(Conversation, session_id)
    if conv is None:
        raise HTTPException(404, "Unknown session_id")
    db.delete(conv)
    db.commit()
