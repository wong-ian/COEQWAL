# generate_batch_analysis.py
import os
import json
import logging
import shutil
import tempfile
import textwrap
import time
from openai import OpenAI

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
DELAY_BETWEEN_REQUESTS_SECONDS = 20

JSON_SKELETON = """
{
  "document": { "filename": "...", "title": "..." },
  "analysis_sections": {
    "general_equity_assessment": { "title": "General Equity Assessment", "summary": "...", "recognitional_equity": { "title": "Recognitional Equity", "positive_findings": "...", "concerns": "...", "conclusion": "..." }, "procedural_equity": { "title": "Procedural Equity", "positive_findings": "...", "concerns": "...", "conclusion": "..." }, "distributional_equity": { "title": "Distributional Equity", "positive_findings": "...", "concerns": "...", "conclusion": "..." }, "structural_equity": { "title": "Structural Equity", "positive_findings": "...", "concerns": "...", "conclusion": "..." } },
    "vulnerable_groups_analysis": { "title": "Vulnerable Groups Analysis", "summary": "...", "identified_groups_and_impacts": "...", "equity_assessment_summary": "...", "conclusion": "..." },
    "severity_impact_analysis": { "title": "Severity of Impact Analysis", "summary": "...", "high_severity_impacts": "...", "moderate_severity_impacts": "...", "equity_implications_of_impacts": "...", "conclusion": "..." },
    "mitigation_strategies_analysis": { "title": "Mitigation Strategies Analysis", "summary": "...", "identified_strategies": "...", "equity_assessment": "...", "conclusion": "..." }
  },
  "overall_summary_and_recommendations": { "title": "Overall Summary & Recommendations", "key_equity_gaps": "...", "key_equity_strengths": "...", "recommendations": "..." }
}
"""

def get_pdf_title(file_path, default_filename):
    if PdfReader is None: return f"Title not extracted (PyPDF2 missing) - {default_filename}"
    try:
        reader = PdfReader(file_path)
        title = reader.metadata.get('/Title', default_filename)
        if not isinstance(title, str): title = str(title) if title else default_filename
        return title.strip()
    except Exception as e:
        logger.error(f"Could not read metadata from PDF '{file_path}': {e}")
        return default_filename

# In generate_batch_analysis.py

def format_analyses_into_json(raw_analyses: dict, filename: str, title: str, client: OpenAI):
    logger.info("-> Synthesizing raw analyses into the final JSON structure...")
    if any("ANALYSIS FAILED" in text for text in raw_analyses.values()):
        logger.error("Skipping JSON formatting due to failure in raw analysis generation.")
        error_json = json.loads(JSON_SKELETON)
        error_json["document"]["filename"] = filename
        error_json["document"]["title"] = title
        error_json["analysis_sections"]["general_equity_assessment"]["summary"] = "Analysis failed during raw text generation. See logs."
        return error_json

    formatter_prompt = textwrap.dedent(f"""
    You are a data structuring expert. Your task is to populate the provided JSON structure using ONLY the information from the four provided text analyses.

    **Instructions:**
    1.  Read the four analyses provided below.
    2.  Fill in every "..." field in the JSON skeleton with **detailed and comprehensive** information synthesized from these analyses. Do not over-summarize; preserve the key details.
    3.  Do NOT invent new information. All content must be derived from the provided texts.
    4.  Ensure the output is a single, valid JSON object and nothing else. Adhere strictly to the schema.
    5.  For the `general_equity_assessment`, break down the 'general' analysis into the sub-fields for each of the four equity dimensions.

    **JSON SKELETON TO POPULATE:**
    {JSON_SKELETON}

    **RAW TEXT ANALYSES TO USE:**

    --- GENERAL ANALYSIS ---
    {raw_analyses.get('general', 'Not provided.')}

    --- VULNERABLE GROUPS ANALYSIS ---
    {raw_analyses.get('vulnerable_groups', 'Not provided.')}

    --- SEVERITY OF IMPACT ANALYSIS ---
    {raw_analyses.get('severity_of_impact', 'Not provided.')}

    --- MITIGATION STRATEGIES ANALYSIS ---
    {raw_analyses.get('mitigation_strategies', 'Not provided.')}
    """)

    try:
        response = client.chat.completions.create(
            model=settings.OPENAI_CHAT_MODEL,
            messages=[{"role": "system", "content": formatter_prompt}],
            response_format={"type": "json_object"}
        )
        json_output_str = response.choices[0].message.content
        structured_data = json.loads(json_output_str)
        structured_data["document"]["filename"] = filename
        structured_data["document"]["title"] = title
        logger.info("-> Successfully synthesized analyses into JSON structure.")
        return structured_data
    except Exception as e:
        logger.error(f"Failed to format analyses into JSON: {e}")
        return None
def main():
    logger.info("--- Starting Batch Document Analysis ---")

    if not settings: logger.critical("Settings not loaded."); return
    try:
        openai_interface = OpenAIInteraction()
        logger.info("OpenAI Interaction layer initialized.")
    except Exception as e: logger.error(f"Failed to initialize OpenAI Interaction: {e}", exc_info=True); return
    
    load_db_on_startup()
    if get_local_db() is None: logger.error("Local DB failed to load."); return
    try:
        rag_system = HybridRAGSystem(openai_interaction=openai_interface)
        logger.info("Hybrid RAG System initialized.")
    except Exception as e: logger.error(f"Failed to initialize RAG System: {e}", exc_info=True); return

    if not os.path.exists(DOCUMENTS_FOLDER): logger.error(f"Folder '{DOCUMENTS_FOLDER}' not found."); return
    pdf_files = [f for f in os.listdir(DOCUMENTS_FOLDER) if f.lower().endswith('.pdf')]
    if not pdf_files: logger.warning(f"No PDF files found in '{DOCUMENTS_FOLDER}'."); return
    logger.info(f"Found {len(pdf_files)} PDF documents to analyze.")
    
    existing_results = []
    if os.path.exists(OUTPUT_JSON_FILE):
        logger.info(f"Found existing results file: {OUTPUT_JSON_FILE}. Will skip already analyzed documents.")
        try:
            with open(OUTPUT_JSON_FILE, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
                if not isinstance(existing_results, list):
                    logger.warning("Existing results file is not a list. Starting fresh.")
                    existing_results = []
        except (json.JSONDecodeError, FileNotFoundError):
            logger.error(f"Could not read or parse existing results file. Starting fresh.")
            existing_results = []
    
    analyzed_filenames = {doc.get('filename') for doc in existing_results if doc.get('filename')}
    
    for filename in pdf_files:
        if filename in analyzed_filenames:
            logger.info(f"Skipping '{filename}' as it is already present in '{OUTPUT_JSON_FILE}'.")
            continue

        logger.info(f"\n--- Analyzing document: {filename} ---")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            original_file_path = os.path.join(DOCUMENTS_FOLDER, filename)
            temp_file_path = os.path.join(temp_dir, filename)
            
            try:
                shutil.copy2(original_file_path, temp_file_path)
            except Exception as e:
                logger.critical(f"Could not create temp copy for '{filename}': {e}. Stopping script.")
                return

            session_id = os.path.splitext(filename)[0].replace(" ", "_")
            title = get_pdf_title(original_file_path, filename)
            raw_analyses = {}
            final_json_result = None

            try:
                logger.info(f"Uploading '{filename}' to OpenAI...")
                success, message = rag_system.add_user_document_for_session(
                    session_id=session_id, file_path=temp_file_path, original_filename=filename
                )
                if not success: raise Exception(f"Document processing failed: {message}")

                for focus_index, focus in enumerate(FOCUS_AREAS):
                    logger.info(f"-> Generating raw analysis for focus: '{focus}'...")
                    answer, _ = rag_system.answer_question(session_id=session_id, query=ANALYSIS_QUERY, focus_area=focus)
                    if "Error:" in answer: raise Exception(f"Received an error for focus '{focus}': {answer}")
                    raw_analyses[focus] = answer
                    
                    if focus_index < len(FOCUS_AREAS) - 1:
                        logger.info(f"Waiting for {DELAY_BETWEEN_REQUESTS_SECONDS} seconds...")
                        time.sleep(DELAY_BETWEEN_REQUESTS_SECONDS)
                
                final_json_result = format_analyses_into_json(raw_analyses, filename, title, openai_interface.client)
                if not final_json_result: raise Exception("Failed to synthesize the final JSON.")
            
            except Exception as e:
                logger.critical(f"A critical error occurred while processing '{filename}': {e}. Stopping the script.")
                logger.info("Cleaning up OpenAI resources for the failed session before exiting...")
                rag_system.remove_user_session_resources(session_id, delete_openai_resources=True)
                return

            logger.info(f"Cleaning up OpenAI resources for session '{session_id}'...")
            rag_system.remove_user_session_resources(session_id, delete_openai_resources=True)
            existing_results.append(final_json_result)
            
            try:
                with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
                    json.dump(existing_results, f, indent=2, ensure_ascii=False)
                logger.info(f"Progress saved to '{OUTPUT_JSON_FILE}'.")
            except Exception as e:
                logger.critical(f"Could not write progress to JSON file: {e}. Stopping script.")
                return

    logger.info("\n--- Batch Analysis Complete ---")

if __name__ == "__main__":
    if not PdfReader:
        logger.error("Cannot run script because PyPDF2 is not installed. Run: pip install PyPDF2")
    else:
        main()