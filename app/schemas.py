from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    messages_in_context: int


class MessageOut(BaseModel):
    role: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


class HistoryResponse(BaseModel):
    session_id: str
    messages: List[MessageOut]
