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
    openai_sources: List[str]

class UploadResponse(BaseModel):
    success: bool
    message: str
    session_id: str
    filename: str
    analysis_status: str = "pending" # initial status

class AnalysisStatusResponse(BaseModel):
    session_id: str
    analysis_status: str # pending, in_progress, completed, failed
    message: Optional[str] = None
    analysis_result_path: Optional[str] = None
    analysis_error: Optional[str] = None
class AnalysisResultResponse(BaseModel):
    session_id: str
    analysis_status: str = "completed"
    analysis_data: Dict[str, Any] # The actual JSON analysis content
    message: Optional[str] = None

class EndSessionRequest(BaseModel):
    session_id: str

class EndSessionResponse(BaseModel):
    success: bool
    message: str

class ErrorResponse(BaseModel):
    detail: str