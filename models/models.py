# models/models.py
from pydantic import BaseModel
from typing import List, Dict, Any, Optional # Added Optional

class QueryRequest(BaseModel):
    session_id: str
    query: str
    focus_area: Optional[str] = "general" # Add focus_area, default to "general"
    custom_instructions: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    local_sources: List[Dict[str, Any]]

class UploadResponse(BaseModel):
    success: bool
    message: str
    session_id: str
    filename: str | None = None

class EndSessionRequest(BaseModel):
    session_id: str

class EndSessionResponse(BaseModel):
    success: bool
    message: str

class ErrorResponse(BaseModel):
    detail: str