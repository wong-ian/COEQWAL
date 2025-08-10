import os
import uuid
import logging
import secrets
from contextlib import asynccontextmanager
import time

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends, status, Cookie, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.config import settings
from core.openai_interaction import OpenAIInteraction
from core.rag_system import HybridRAGSystem
from core.local_db import load_db_on_startup, get_local_db
from core import equity_analyzer
from models.models import (
    QueryRequest,
    QueryResponse,
    UploadResponse, 
    ErrorResponse,
    EndSessionRequest,
    EndSessionResponse,
    AnalysisStatusResponse, 
    AnalysisResultResponse  
)

logger = logging.getLogger("main_app")

openai_interface: OpenAIInteraction | None = None
rag_system: HybridRAGSystem | None = None

TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

ANALYSIS_OUTPUT_FOLDER = "analysis_results_json"
os.makedirs(ANALYSIS_OUTPUT_FOLDER, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup...")
    global openai_interface, rag_system
    if not settings:
        logger.critical("Settings not loaded. Cannot initialize.")
        yield
        return

    try:
        openai_interface = OpenAIInteraction()
        logger.info("OpenAI Interaction layer initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI Interaction: {e}", exc_info=True)
        openai_interface = None

    load_db_on_startup()
    local_db_instance = get_local_db()
    if local_db_instance:
        logger.info(f"Local DB loaded with {len(local_db_instance.documents)} docs.")
    else:
        logger.warning("Local DB did not load successfully.")

    if openai_interface:
        try:
            rag_system = HybridRAGSystem(openai_interaction=openai_interface)
            logger.info("Hybrid RAG System initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize RAG System: {e}", exc_info=True)
            rag_system = None
    else:
        logger.error("Cannot initialize RAG system because OpenAI Interaction failed.")
        rag_system = None

    # Ensure ANALYSIS_OUTPUT_FOLDER exists on startup
    os.makedirs(ANALYSIS_OUTPUT_FOLDER, exist_ok=True)
    logger.info(f"Analysis output folder '{ANALYSIS_OUTPUT_FOLDER}' ensured.")

    logger.info("Startup complete.")
    yield
    logger.info("Application shutdown...")

app = FastAPI(title="COEQWAL Analysis Bot", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

async def check_system_ready():
    if not rag_system or not openai_interface:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Core system components are not initialized. Please check server logs."
        )

async def get_session_id(session_id: str | None = Cookie(None)) -> str | None:
    if session_id is None: return None
    return session_id

async def ensure_session(response: Response, session_id: str | None = Cookie(None)) -> str:
     if session_id is None:
         session_id = str(uuid.uuid4())
         logger.info(f"New session started: {session_id}")
         response.set_cookie(key="session_id", value=session_id, httponly=True, samesite='lax')
     return session_id

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload",
          response_model=UploadResponse,
          responses={
              status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse, "description": "Core system not ready"},
              status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse, "description": "File processing failed"},
              status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse, "description": "Internal server error during upload"},
          })
async def upload_document(
    background_tasks: BackgroundTasks, # Inject BackgroundTasks
    response: Response,
    file: UploadFile = File(...),
    session_id: str | None = Depends(get_session_id),
    _=Depends(check_system_ready)
):
    active_session_id = await ensure_session(response, session_id)
    original_filename = file.filename or "uploaded_file"
    temp_file_path = os.path.join(TEMP_UPLOAD_DIR, f"{active_session_id}_{original_filename}")

    # Ensure rag_system is available
    if rag_system is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized. Cannot process upload."
        )

    try:
        logger.info(f"Receiving file '{original_filename}' for session {active_session_id}")
        # Save the file temporarily for processing
        with open(temp_file_path, "wb") as buffer: content = await file.read(); buffer.write(content)
        logger.info(f"Temporarily saved file to {temp_file_path}")

        # --- CRITICAL CHANGE: Initialize session data HERE before calling rag_system.add_user_document_for_session ---
        # This ensures the session exists in user_sessions before any background task or subsequent rag_system call.
        rag_system.user_sessions[active_session_id] = {
            "file_id": None, # Will be filled by add_user_document_for_session
            "vector_store_id": None, # Will be filled by add_user_document_for_session
            "original_filename": original_filename,
            "status": "initial_upload_pending", # Specific status for main.py to track
            "analysis_status": "pending", # Initial analysis status
            "analysis_result_cached": None,
            "analysis_result_path": None,
            "analysis_error": None,
            "temp_file_path": temp_file_path # Keep track of temp path for cleanup by analyzer
        }

        # Perform the initial document upload and vector store creation (this still takes time)
        # rag_system.add_user_document_for_session will now update the 'file_id', 'vector_store_id', and 'status'
        # fields within rag_system.user_sessions[active_session_id] directly.
        success, message = rag_system.add_user_document_for_session(
            session_id=active_session_id,
            file_path=temp_file_path, # Pass temp_file_path for processing
            original_filename=original_filename
        )
        
        # After rag_system.add_user_document_for_session completes,
        # get the updated status from the session_info
        current_upload_status = rag_system.user_sessions[active_session_id].get("status", "unknown")


        if current_upload_status == "completed": # Check the *actual* status from rag_system
            # If upload and VS setup is successful, schedule the detailed analysis as a background task
            background_tasks.add_task(
                equity_analyzer.perform_equity_analysis,
                session_id=active_session_id,
                temp_file_path=temp_file_path, # Pass the path to the temp file
                original_filename=original_filename,
                title=equity_analyzer.get_pdf_title(temp_file_path, original_filename), # Get title here
                file_size_kb=os.path.getsize(temp_file_path) // 1024,
                upload_date_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(os.path.getmtime(temp_file_path))),
                rag_system_instance=rag_system, # Pass the global instance
                openai_interface_instance=openai_interface, # Pass the global instance
                user_sessions_dict=rag_system.user_sessions, # Pass the dict itself for updates
                analysis_output_dir=ANALYSIS_OUTPUT_FOLDER # Pass the configured output folder
            )
            return UploadResponse(
                success=True,
                message=message + " Detailed analysis started in the background.",
                session_id=active_session_id,
                filename=original_filename,
                analysis_status="pending" # Frontend initial status
            )
        else:
            # If initial upload/VS setup failed (status is not "completed"), update analysis_status to failed directly
            # The 'message' from rag_system.add_user_document_for_session already explains the failure.
            # The rag_system.user_sessions[active_session_id] should have been updated by rag_system itself.
            rag_system.user_sessions[active_session_id]['analysis_status'] = 'failed'
            rag_system.user_sessions[active_session_id]['analysis_error'] = message
            
            # Cleanup OpenAI resources and the temp file immediately if the *initial* upload/VS creation failed.
            # rag_system.add_user_document_for_session will have tried to delete the file_id/vector_store_id if it failed
            # midway. Here, we ensure everything is clean for a failed upload.
            rag_system.remove_user_session_resources(active_session_id, delete_openai_resources=True)
            
            # This temp_file_path should ideally be handled by equity_analyzer's finally block,
            # but if the analysis task *never starts* due to a pre-analysis failure, we need to delete it here.
            if os.path.exists(temp_file_path):
                try: os.remove(temp_file_path); logger.info(f"Cleaned temp file on failed upload: {temp_file_path}")
                except OSError: logger.error(f"Could not remove temp file on failed upload: {temp_file_path}")

            status_code = status.HTTP_400_BAD_REQUEST # Assume client error for failed upload/VS
            raise HTTPException(status_code=status_code, detail=message)
    except HTTPException as e:
         # HTTPExceptions are re-raised immediately, so ensure cleanup before raising
         if active_session_id in rag_system.user_sessions:
             rag_system.user_sessions[active_session_id]['analysis_status'] = 'failed'
             rag_system.user_sessions[active_session_id]['analysis_error'] = str(e.detail)
             # Attempt to clean up resources related to the session as it failed early
             rag_system.remove_user_session_resources(active_session_id, delete_openai_resources=True)
         
         # Clean up temp file if it's still lingering and not managed elsewhere
         if os.path.exists(temp_file_path):
            try: os.remove(temp_file_path); logger.info(f"Cleaned temp file on HTTPException: {temp_file_path}")
            except OSError: logger.error(f"Could not remove temp file on HTTPException: {temp_file_path}")
         raise e
    except Exception as e:
         logger.error(f"Unexpected error during upload for session {active_session_id}: {e}", exc_info=True)
         if active_session_id in rag_system.user_sessions:
             rag_system.user_sessions[active_session_id]['analysis_status'] = 'failed'
             rag_system.user_sessions[active_session_id]['analysis_error'] = str(e)
             # Attempt to clean up resources related to the session as it failed early
             rag_system.remove_user_session_resources(active_session_id, delete_openai_resources=True)
         
         # Clean up temp file if it's still lingering and not managed elsewhere
         if os.path.exists(temp_file_path):
            try: os.remove(temp_file_path); logger.info(f"Cleaned temp file on unexpected error: {temp_file_path}")
            except OSError: logger.error(f"Could not remove temp file on unexpected error: {temp_file_path}")
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during upload initiation.")

@app.post("/query",
          response_model=QueryResponse,
          responses={
              status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse, "description": "Core system not ready"},
              status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse, "description": "Internal server error during query"},
          })
async def handle_query(
    query_req: QueryRequest,
    _=Depends(check_system_ready)
):
    session_id = query_req.session_id
    query = query_req.query
    focus_area = query_req.focus_area
    custom_instructions = query_req.custom_instructions

    logger.info(f"Received query for session {session_id}, focus '{focus_area}': '{query[:100]}...'")
    if custom_instructions:
        logger.info(f"Custom instructions provided: '{custom_instructions[:100]}...'")
    if session_id not in rag_system.user_sessions:
         logger.warning(f"Query for session {session_id}, but no document info in RAG system. Local context only.")

    try:
        # Call the analysis function ONCE and store all three results
        answer, local_sources, openai_sources = rag_system.answer_question(
            session_id=session_id,
            query=query,
            focus_area=focus_area,
            custom_instructions=custom_instructions
        )
        cleaned_answer = answer.strip() if answer else "No answer generated."
        
        # Log the exact data being sent back for debugging
        logger.info("--- DATA TO BE SENT TO FRONTEND ---")
        logger.info(f"Answer length: {len(cleaned_answer)}")
        logger.info(f"OpenAI Sources found: {len(openai_sources)}")
        for i, source in enumerate(openai_sources):
            clean_source = source.replace('</blockquote>', '').replace('<blockquote>', ' ').strip()
            logger.info(f"  Source {i+1}: {clean_source}")
        logger.info("------------------------------------")
        
        return QueryResponse(
            answer=cleaned_answer,
            local_sources=local_sources,
            openai_sources=openai_sources
        )

    except Exception as e:
        logger.error(f"Error during query processing for session {session_id}: {e}", exc_info=True)
        detail = str(e) if "Error:" in str(e) else "Internal server error during query processing."
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)

@app.post("/end-session",
          response_model=EndSessionResponse,
          responses={
              status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse, "description": "Core system not ready"},
              status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": EndSessionResponse, "description": "Internal error or cleanup failed"},
          })
async def end_session(
    end_req: EndSessionRequest,
    response: Response,
    _=Depends(check_system_ready)
):
    session_id = end_req.session_id
    logger.info(f"Received request to end session and clean up resources for: {session_id}")

    # Ensure rag_system is available
    if rag_system is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized. Cannot end session."
        )

    # Check if session exists in user_sessions
    session_info = rag_system.user_sessions.get(session_id)
    if not session_info:
        logger.warning(f"Request to end session {session_id}, but it was not found in active sessions or already cleaned up.")
        # Attempt to delete cookie even if session not found on backend
        response.delete_cookie(key="session_id")
        return EndSessionResponse(success=True, message="Session not found or already cleaned up.")

    try:
        # Pass the temp_file_path to remove_user_session_resources for comprehensive cleanup
        temp_file_to_delete = session_info.get("temp_file_path")
        
        success = rag_system.remove_user_session_resources(session_id, delete_openai_resources=True)
        
        # Explicitly delete the temporary uploaded file if it still exists and wasn't cleaned by analyzer
        if temp_file_to_delete and os.path.exists(temp_file_to_delete):
            try:
                os.remove(temp_file_to_delete)
                logger.info(f"Deleted lingering temporary file: {temp_file_to_delete}")
            except OSError as e:
                logger.error(f"Error deleting lingering temporary file {temp_file_to_delete}: {e}")
                success = False # Mark overall success as false if file deletion failed

        # Also remove the generated analysis JSON file if it exists
        analysis_json_path = session_info.get("analysis_result_path")
        if analysis_json_path and os.path.exists(analysis_json_path):
            try:
                os.remove(analysis_json_path)
                logger.info(f"Deleted analysis JSON file: {analysis_json_path}")
            except OSError as e:
                logger.error(f"Error deleting analysis JSON file {analysis_json_path}: {e}")
                success = False

        if success:
            message = "Session ended and associated resources cleaned up successfully."
            logger.info(message + f" (Session ID: {session_id})")
            response.delete_cookie(key="session_id")
            return EndSessionResponse(success=True, message=message)
        else:
            message = "Session ended, but an error occurred during resource cleanup on the backend. Check server logs."
            logger.error(message + f" (Session ID: {session_id})")
            response.delete_cookie(key="session_id")
            return JSONResponse(
                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                 content={"success": False, "message": message}
            )
    except Exception as e:
        logger.error(f"Unexpected error during session end for {session_id}: {e}", exc_info=True)
        response.delete_cookie(key="session_id")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal server error occurred while ending the session.")

@app.get("/get_analysis_status/{session_id}", response_model=AnalysisStatusResponse,
         responses={
             status.HTTP_404_NOT_FOUND: {"model": ErrorResponse, "description": "Session not found"},
             status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse, "description": "Core system not ready"},
         })
async def get_analysis_status(session_id: str, _=Depends(check_system_ready)):
    if rag_system is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="RAG system not initialized.")

    session_info = rag_system.user_sessions.get(session_id)
    if not session_info:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {session_id} not found.")
    
    return AnalysisStatusResponse(
        session_id=session_id,
        analysis_status=session_info.get("analysis_status", "unknown"),
        message=f"Analysis status for session {session_id} is {session_info.get('analysis_status', 'unknown')}.",
        analysis_result_path=session_info.get("analysis_result_path"),
        analysis_error=session_info.get("analysis_error")
    )

@app.get("/get_analysis_result/{session_id}", response_model=AnalysisResultResponse,
         responses={
             status.HTTP_404_NOT_FOUND: {"model": ErrorResponse, "description": "Session or analysis not found"},
             status.HTTP_409_CONFLICT: {"model": ErrorResponse, "description": "Analysis not yet completed"},
             status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse, "description": "Error retrieving analysis"},
             status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse, "description": "Core system not ready"},
         })
async def get_analysis_result(session_id: str, _=Depends(check_system_ready)):
    if rag_system is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="RAG system not initialized.")

    session_info = rag_system.user_sessions.get(session_id)
    if not session_info:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {session_id} not found.")

    analysis_status = session_info.get("analysis_status")
    if analysis_status != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, # 409 Conflict indicates the request cannot be completed due to the current state.
            detail=f"Analysis for session {session_id} is not yet completed. Current status: {analysis_status}. Error: {session_info.get('analysis_error', 'N/A')}"
        )
    
    analysis_data = session_info.get("analysis_result_cached")
    analysis_path = session_info.get("analysis_result_path")

    if analysis_data is None and analysis_path and os.path.exists(analysis_path):
        try:
            with open(analysis_path, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
            # Optionally cache it for future quick access
            session_info['analysis_result_cached'] = analysis_data
        except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
            logger.error(f"Error loading analysis result from file {analysis_path} for session {session_id}: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not load analysis result from file: {e}")
    elif analysis_data is None and (analysis_path is None or not os.path.exists(analysis_path)):
         logger.error(f"Analysis result not found in cache or file for session {session_id}. Path: {analysis_path}")
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Analysis result not found.")


    return AnalysisResultResponse(
        session_id=session_id,
        analysis_status="completed",
        analysis_data=analysis_data,
        message="Analysis completed and retrieved successfully."
    )  

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    if rag_system and openai_interface: return {"status": "ok", "rag_system_initialized": True, "openai_initialized": True}
    else: return {"status": "degraded", "rag_system_initialized": bool(rag_system), "openai_initialized": bool(openai_interface)}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for local development...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, log_level="info")