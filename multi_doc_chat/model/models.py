
from pydantic import BaseModel, Field
from typing import Annotated
from enum import Enum

class ChatAnswer(BaseModel):
    # validate chat answer type and length
    answer: Annotated[str, Field(min_length=1, max_length=4096)]

class PromptType(str, Enum):
    CONTEXTUALIZE_QUESTION = "contextualize_question"
    CONTEXT_QUESTION = "context_question"

class UploadResponse(BaseModel):
    session_id = str
    indexed: bool
    message: str | None = None

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    answer: str






