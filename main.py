# main.py
import os
import uuid
import logging
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends, status, Cookie
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware # Optional: If frontend served differently
from pydantic import BaseModel # Ensure BaseModel is imported

from core.config import settings
from core.openai_interaction import OpenAIInteraction
from core.rag_system import HybridRAGSystem
from core.local_db import load_db_on_startup, get_local_db
# Import new models
from models.models import (
    QueryRequest, QueryResponse, UploadResponse, ErrorResponse,
    EndSessionRequest, EndSessionResponse
)

# --- Logging Setup ---
logger = logging.getLogger("main_app")

# --- Global Variables ---
openai_interface: OpenAIInteraction | None = None
rag_system: HybridRAGSystem | None = None

# --- Temporary File Storage ---
TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# --- FastAPI Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # (Keep existing lifespan logic as is)
    logger.info("Application startup...")
    global openai_interface, rag_system
    if not settings: logger.critical("Settings not loaded."); yield; return
    try: openai_interface = OpenAIInteraction(); logger.info("OpenAI Interaction initialized.")
    except Exception as e: logger.error(f"Failed OpenAI init: {e}", exc_info=True); openai_interface = None
    load_db_on_startup()
    local_db_instance = get_local_db()
    if local_db_instance: logger.info(f"Local DB loaded with {len(local_db_instance.documents)} docs.")
    else: logger.warning("Local DB did not load successfully.")
    if openai_interface:
        try: rag_system = HybridRAGSystem(openai_interaction=openai_interface); logger.info("Hybrid RAG System initialized.")
        except Exception as e: logger.error(f"Failed RAG init: {e}", exc_info=True); rag_system = None
    else: logger.error("Cannot initialize RAG system (OpenAI failed)."); rag_system = None
    logger.info("Startup complete.")
    yield
    logger.info("Application shutdown...")

# --- FastAPI App Initialization ---
app = FastAPI(title="COEQWAL Analysis Bot", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- CORS Middleware (Optional) ---
# (Keep commented out unless needed)

# --- Dependency Checks ---
async def check_system_ready():
    if not rag_system or not openai_interface:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Core system components not initialized.")

# --- Session Management ---
async def get_session_id(session_id: str | None = Cookie(None)) -> str | None:
    if session_id is None: return None
    return session_id

async def ensure_session(response: Response, session_id: str | None = Cookie(None)) -> str:
     if session_id is None:
         session_id = str(uuid.uuid4())
         logger.info(f"New session started: {session_id}")
         response.set_cookie(key="session_id", value=session_id, httponly=True, samesite='lax')
     return session_id

# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- FIX 1: Define the responses dictionary ---
@app.post("/upload",
          response_model=UploadResponse,
          responses={
              status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse, "description": "Core system not ready"},
              status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse, "description": "File processing failed"},
              status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse, "description": "Internal server error during upload"},
          })
async def upload_document(
    response: Response, # To set the cookie
    file: UploadFile = File(...),
    session_id: str | None = Depends(get_session_id),
    _=Depends(check_system_ready)
):
    """Handles upload. Gets or creates session_id and sets cookie."""
    active_session_id = await ensure_session(response, session_id)
    original_filename = file.filename or "uploaded_file"
    temp_file_path = os.path.join(TEMP_UPLOAD_DIR, f"{active_session_id}_{original_filename}")

    try:
        logger.info(f"Receiving file '{original_filename}' for session {active_session_id}")
        with open(temp_file_path, "wb") as buffer: content = await file.read(); buffer.write(content)
        logger.info(f"Temporarily saved file to {temp_file_path}")

        success, message = rag_system.add_user_document_for_session(
            session_id=active_session_id,
            file_path=temp_file_path,
            original_filename=original_filename
        )

        if success:
            return UploadResponse(success=True, message=message, session_id=active_session_id, filename=original_filename)
        else:
            status_code = status.HTTP_400_BAD_REQUEST if "failed processing" in message.lower() else status.HTTP_500_INTERNAL_SERVER_ERROR
            raise HTTPException(status_code=status_code, detail=message)

    except HTTPException as e:
         if os.path.exists(temp_file_path): 
            try: os.remove(temp_file_path); logger.info(f"Cleaned temp file on error: {temp_file_path}")
            except OSError: logger.error(f"Could not remove temp file on error: {temp_file_path}")
         raise e
    except Exception as e:
         logger.error(f"Unexpected error during upload for session {active_session_id}: {e}", exc_info=True)
         if os.path.exists(temp_file_path): 
            try: os.remove(temp_file_path); logger.info(f"Cleaned temp file on error: {temp_file_path}")
            except OSError: logger.error(f"Could not remove temp file on error: {temp_file_path}")
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during upload.")


# --- FIX 2: Define the responses dictionary ---
@app.post("/query",
          response_model=QueryResponse,
          responses={
              status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse, "description": "Core system not ready"},
              status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse, "description": "Internal server error during query"},
          })
async def handle_query(
    query_req: QueryRequest, # QueryRequest now includes custom_instructions
    _=Depends(check_system_ready)
):
    session_id = query_req.session_id
    query = query_req.query
    focus_area = query_req.focus_area
    custom_instructions = query_req.custom_instructions # <-- Extract the new field

    logger.info(f"Received query for session {session_id}, focus '{focus_area}': '{query[:100]}...'")
    if custom_instructions:
        logger.info(f"Custom instructions provided: '{custom_instructions[:100]}...'")
    if session_id not in rag_system.user_sessions:
         logger.warning(f"Query for session {session_id}, but no document info in RAG system. Local context only.")

    try:
        # Pass focus_area to answer_question
        answer, local_sources = rag_system.answer_question(
            session_id=session_id,
            query=query,
            focus_area=focus_area,
            custom_instructions=custom_instructions
        )
        cleaned_answer = answer.strip() if answer else "No answer generated."
        return QueryResponse(answer=cleaned_answer, local_sources=local_sources)
    except Exception as e:
        logger.error(f"Error during query processing for session {session_id}: {e}", exc_info=True)
        detail = str(e) if "Error:" in str(e) else "Internal server error during query processing."
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)



# --- FIX 3: Define the responses dictionary ---
@app.post("/end-session",
          response_model=EndSessionResponse,
          responses={
              status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse, "description": "Core system not ready"},
              status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": EndSessionResponse, "description": "Internal error or cleanup failed"},
          })
async def end_session(
    end_req: EndSessionRequest,
    response: Response, # To potentially clear the cookie
    _=Depends(check_system_ready)
):
    """Cleans up resources for the given session ID."""
    session_id = end_req.session_id
    logger.info(f"Received request to end session and clean up resources for: {session_id}")

    if session_id not in rag_system.user_sessions:
        logger.warning(f"Request to end session {session_id}, but it was not found in active sessions.")
        return EndSessionResponse(success=True, message="Session not found or already cleaned up.")

    try:
        success = rag_system.remove_user_session_resources(session_id, delete_openai_resources=True)

        if success:
            message = "Session ended and associated resources cleaned up successfully."
            logger.info(message + f" (Session ID: {session_id})")
            response.delete_cookie(key="session_id")
            return EndSessionResponse(success=True, message=message)
        else:
            message = "Session ended, but an error occurred during resource cleanup on the backend. Check server logs."
            logger.error(message + f" (Session ID: {session_id})")
            response.delete_cookie(key="session_id")
            # Return 500 error status code along with success=false in body
            return JSONResponse(
                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                 content={"success": False, "message": message}
            )

    except Exception as e:
        logger.error(f"Unexpected error during session end for {session_id}: {e}", exc_info=True)
        response.delete_cookie(key="session_id")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal server error occurred while ending the session.")

# --- Health Check Endpoint ---
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    if rag_system and openai_interface: return {"status": "ok", "rag_system_initialized": True, "openai_initialized": True}
    else: return {"status": "degraded", "rag_system_initialized": bool(rag_system), "openai_initialized": bool(openai_interface)}

# --- Run with Uvicorn ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for local development...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, log_level="info")