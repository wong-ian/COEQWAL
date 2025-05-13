# models/models.py
from pydantic import BaseModel
from typing import List, Dict, Any

class QueryRequest(BaseModel):
    session_id: str
    query: str

class QueryResponse(BaseModel):
    answer: str
    local_sources: List[Dict[str, Any]]

class UploadResponse(BaseModel):
    success: bool
    message: str
    session_id: str
    filename: str | None = None

# --- New Models ---
class EndSessionRequest(BaseModel):
    session_id: str

class EndSessionResponse(BaseModel):
    success: bool
    message: str
# --- End New Models ---

class ErrorResponse(BaseModel):
    detail: str