# generate_batch_analysis.py
import os
import json
import logging
import shutil
import tempfile
from datetime import datetime, timezone

try:
    from PyPDF2 import PdfReader
except ImportError:
    print("PyPDF2 is not installed. Please run: pip install PyPDF2")
    PdfReader = None

from core.config import settings
from core.openai_interaction import OpenAIInteraction
from core.rag_system import HybridRAGSystem
from core.local_db import load_db_on_startup, get_local_db

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("batch_analysis_script")

DOCUMENTS_FOLDER = "Documents"
OUTPUT_JSON_FILE = "analysis_results.json"
FOCUS_AREAS = ["general", "vulnerable_groups", "severity_of_impact", "mitigation_strategies"]
ANALYSIS_QUERY = "Analyze this document in detail, concentrating on the specified equity focus area."

def get_pdf_title(file_path, default_filename):
    if PdfReader is None:
        return f"Title could not be extracted (PyPDF2 not installed) - {default_filename}"
    try:
        reader = PdfReader(file_path)
        title = reader.metadata.get('/Title', default_filename)
        if not isinstance(title, str):
            title = str(title) if title else default_filename
        return title.strip()
    except Exception as e:
        logger.error(f"Could not read metadata from PDF '{file_path}': {e}")
        return default_filename

def main():
    logger.info("--- Starting Batch Document Analysis ---")

    if not settings:
        logger.critical("Settings not loaded. Check .env file.")
        return

    try:
        openai_interface = OpenAIInteraction()
        logger.info("OpenAI Interaction layer initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI Interaction: {e}", exc_info=True)
        return

    load_db_on_startup()
    if get_local_db() is None:
        logger.error("Local vector database failed to load. Cannot proceed.")
        return

    try:
        rag_system = HybridRAGSystem(openai_interaction=openai_interface)
        logger.info("Hybrid RAG System initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize RAG System: {e}", exc_info=True)
        return

    if not os.path.exists(DOCUMENTS_FOLDER):
        logger.error(f"Error: Folder '{DOCUMENTS_FOLDER}' not found. Please create it and place your PDFs inside.")
        return

    pdf_files = [f for f in os.listdir(DOCUMENTS_FOLDER) if f.lower().endswith('.pdf')]
    if not pdf_files:
        logger.warning(f"No PDF files found in '{DOCUMENTS_FOLDER}'.")
        return

    logger.info(f"Found {len(pdf_files)} PDF documents to analyze.")
    
    all_document_analyses = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Created temporary directory for processing: {temp_dir}")

        for i, filename in enumerate(pdf_files):
            logger.info(f"\n--- Analyzing document {i+1} of {len(pdf_files)}: {filename} ---")
            
            original_file_path = os.path.join(DOCUMENTS_FOLDER, filename)
            temp_file_path = os.path.join(temp_dir, filename)
            
            try:
                shutil.copy2(original_file_path, temp_file_path)
                logger.info(f"Created temporary copy at: {temp_file_path}")
            except Exception as e:
                logger.error(f"Could not create temporary copy for '{filename}': {e}. Skipping this file.")
                failed_result = {
                    "filename": filename,
                    "title": get_pdf_title(original_file_path, filename),
                    "analyses": {focus: f"ANALYSIS FAILED: Could not create temporary file copy." for focus in FOCUS_AREAS}
                }
                all_document_analyses.append(failed_result)
                continue

            session_id = os.path.splitext(filename)[0].replace(" ", "_")
            title = get_pdf_title(original_file_path, filename)

            document_result = {
                "filename": filename,
                "title": title,
                "analyses": {}
            }

            try:
                logger.info(f"Uploading '{filename}' to OpenAI...")
                success, message = rag_system.add_user_document_for_session(
                    session_id=session_id,
                    file_path=temp_file_path,
                    original_filename=filename
                )
                if not success:
                    error_message = f"Document processing failed: {message}"
                    logger.error(error_message)
                    for focus in FOCUS_AREAS:
                        document_result["analyses"][focus] = f"ANALYSIS FAILED: {error_message}"
                else:
                    for focus in FOCUS_AREAS:
                        logger.info(f"-> Generating analysis for focus: '{focus}'...")
                        
                        answer, _ = rag_system.answer_question(
                            session_id=session_id,
                            query=ANALYSIS_QUERY,
                            focus_area=focus
                        )
                        
                        if "Error:" in answer:
                            logger.error(f"Received an error for focus '{focus}': {answer}")
                        
                        document_result["analyses"][focus] = answer

            except Exception as e:
                error_message = f"An unexpected error occurred: {e}"
                logger.error(f"Error processing '{filename}': {error_message}")
                for focus in FOCUS_AREAS:
                    document_result["analyses"][focus] = f"ANALYSIS FAILED: {error_message}"

            finally:
                logger.info(f"Cleaning up OpenAI resources for session '{session_id}'...")
                rag_system.remove_user_session_resources(session_id, delete_openai_resources=True)
                
                all_document_analyses.append(document_result)
                
                try:
                    with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
                        json.dump(all_document_analyses, f, indent=2, ensure_ascii=False)
                    logger.info(f"Progress saved to '{OUTPUT_JSON_FILE}'.")
                except Exception as e:
                    logger.error(f"Could not write progress to JSON file: {e}")

    logger.info("\n--- Batch Analysis Complete ---")
    logger.info(f"All results saved to '{OUTPUT_JSON_FILE}'.")


if __name__ == "__main__":
    if not PdfReader:
        logger.error("Cannot run script because PyPDF2 is not installed. Please run: pip install PyPDF2")
    else:
        main()