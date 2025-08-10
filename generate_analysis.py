# generate_batch_analysis.py
import os
import json
import logging
import shutil
import tempfile
import textwrap
import time
from openai import OpenAI
from typing import Dict, Any, List, Optional # Added for type hinting

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
# Changed to a directory for per-file JSON outputs
OUTPUT_DIR = "analysis_outputs"
FOCUS_AREAS = ["general", "vulnerable_groups", "severity_of_impact", "mitigation_strategies"]

# ANALYSIS_QUERY is now more generic as the specific framing will come from _get_system_prompt
# The {focus_description} will be populated by a specific detail instruction for each analysis type.
ANALYSIS_QUERY_GENERIC = "Provide an equity analysis of this document, focusing on: {focus_description}"
DELAY_BETWEEN_REQUESTS_SECONDS = 5 # Reduced delay to 5 seconds

# Define the new perspectives and their specific prompts for raw analysis generation
# These descriptions are for the LLM to understand what to focus on for each raw analysis.
# The 'indicative' tone will be baked into the prompts from rag_system.py
PERSPECTIVES = [
    {
        "group_name": "Policy Makers",
        "description": "the overall equity implications from the perspective of federal and state policymakers, considering their regulatory responsibilities and influence on equity outcomes. This assessment could indicate how effectively equity measures are embedded in their processes, or if there might be systematic gaps in addressing social or racial disparities.",
        "dimensions": {
            "recognitional": "how the document suggests policy makers acknowledge how historical or systemic exclusion might impact decision-making access for marginalized groups. The analysis might highlight whether unique water justice needs appear to be adequately recognized.",
            "procedural": "how the document examines indications of how agencies manage pollution control and public input, considering whether affected communities appear to meaningfully influence agenda-setting or enforcement priorities.",
            "distributional": "whether the document suggests funding and technical assistance are directed in a way that prioritizes equity impacts, or if there might be unaddressed distributional disparities.",
            "structural": "how the document hints at perpetuating top-down oversight, and whether it could suggest a reallocation of decision-making power or reduction of structural barriers for historically excluded groups."
        }
    },
    {
        "group_name": "Residents",
        "description": "the overall equity implications from the perspective of ordinary residents, especially those in vulnerable communities, regarding their access to clean and affordable water. The assessment might indicate if community voices appear to be sufficiently represented in decisions affecting their health and well-being.",
        "dimensions": {
            "recognitional": "how the document suggests residents experience inequity when their specific circumstances—such as historic under-investment in infrastructure or exposure to legacy industrial sites—appear overlooked or undervalued in funding and policy priorities.",
            "procedural": "how the document examines indications of meaningful resident engagement in public hearings and notice periods, and if marginalized populations might face capacity issues in effective participation.",
            "distributional": "whether the document suggests residents in distressed or rural communities might benefit less from environmental outcomes, or if they appear to pay higher rates for water services due to infrastructure underfunding or costly upgrades.",
            "structural": "how the document hints at systemic obstacles for residents, particularly from minority and low-income groups, such as under-resourced water utilities, lack of representation in regulatory processes, and enduring legacies of exclusion from environmental policymaking."
        }
    },
    {
        "group_name": "Farmers/Business Owners",
        "description": "the overall equity implications from the perspective of farmers and business owners, particularly small operators, considering their roles as regulated entities and community members. The assessment could indicate if compliance costs or infrastructure funding might impose disproportionate economic hardship.",
        "dimensions": {
            "recognitional": "how the document suggests the Act primarily recognizes large industrial and municipal actors, and if distinct rural or small-operator needs might be missed in the broader regulatory structures.",
            "procedural": "how the document assesses whether adequate input opportunities exist for small businesses and farmers through regulatory consultations and permit processes, or if resource constraints, lack of technical assistance, and geographic isolation might impede their meaningful participation.",
            "distributional": "whether the document suggests compliance costs (such as new filtration or runoff systems) might weigh heavier on small agricultural operations and small businesses—sometimes threatening economic viability, especially in distressed regions.",
            "structural": "how the document hints at small operators generally lacking a seat at high-level regulatory tables; financial and information barriers could prevent them from accessing support or defending their interests."
        }
    }
]

# Updated JSON Skeleton to remove 'transformational_equity' and include 'sources' array
# NOTE: The 'sources' structure will be {"type": "openai", "data": "..."} as local sources are excluded
JSON_SKELETON = """
{
  "document": {
    "filename": "...",
    "title": "...",
    "size_kb": 0,
    "upload_date_utc": "..."
  },
  "analysis_sections": {
    "general_equity_assessment": {
      "title": "General Equity Assessment",
      "summary": "...",
      "sources": [],
      "recognitional_equity": { "title": "Recognitional Equity", "positive_findings": "...", "concerns": "...", "conclusion": "..." },
      "procedural_equity": { "title": "Procedural Equity", "positive_findings": "...", "concerns": "...", "conclusion": "..." },
      "distributional_equity": { "title": "Distributional Equity", "positive_findings": "...", "concerns": "...", "conclusion": "..." },
      "structural_equity": { "title": "Structural Equity", "positive_findings": "...", "concerns": "...", "conclusion": "..." }
    },
    "vulnerable_groups_analysis": {
      "title": "Vulnerable Groups Analysis",
      "summary": "...",
      "identified_groups_and_impacts": "...",
      "equity_assessment_summary": "...",
      "conclusion": "...",
      "sources": []
    },
    "severity_impact_analysis": {
      "title": "Severity of Impact Analysis",
      "summary": "...",
      "high_severity_impacts": "...",
      "moderate_severity_impacts": "...",
      "equity_implications_of_impacts": "...",
      "conclusion": "...",
      "sources": []
    },
    "mitigation_strategies_analysis": {
      "title": "Mitigation Strategies Analysis",
      "summary": "...",
      "identified_strategies": "...",
      "equity_assessment": "...",
      "conclusion": "...",
      "sources": []
    }
  },
  "equity_analysis_by_perspective": [
    {
      "group": "...",
      "general_equity_assessment": {
        "title": "...",
        "narrative": "...",
        "sources": []
      },
      "recognitional_equity": { "description": "...", "sources": [] },
      "procedural_equity": { "description": "...", "sources": [] },
      "distributional_equity": { "description": "...", "sources": [] },
      "structural_equity": { "description": "...", "sources": [] }
    }
  ],
  "overall_summary_and_recommendations": {
    "title": "Overall Summary & Recommendations",
    "key_equity_gaps": "...",
    "key_equity_strengths": "...",
    "recommendations": "...",
    "sources": []
  }
}
"""


def get_pdf_title(file_path: str, default_filename: str) -> str:
    """Extracts title from PDF metadata or returns default filename."""
    if PdfReader is None:
        logger.warning(f"PyPDF2 not installed. Title extraction for '{default_filename}' skipped.")
        return f"Title not extracted (PyPDF2 missing) - {default_filename}"
    try:
        reader = PdfReader(file_path)
        title = reader.metadata.get('/Title', default_filename)
        if not isinstance(title, str):
            title = str(title) if title else default_filename
        return title.strip()
    except Exception as e:
        logger.error(f"Could not read metadata from PDF '{file_path}': {e}")
        return default_filename

def format_analyses_into_json(raw_analyses: Dict[str, Dict[str, Any]], filename: str, title: str,
                              file_size_kb: int, upload_date_utc: str, client: OpenAI) -> Optional[Dict[str, Any]]:
    """
    Synthesizes raw text analyses into the final structured JSON format.
    Sources are inserted *after* LLM generation, purely by Python.
    """
    logger.info("-> Synthesizing raw analyses into the final JSON structure (text portion)...")

    # Check for any failures in raw analysis generation from the `raw_analyses` dict
    if any("ANALYSIS FAILED" in raw_analysis.get("text", "") for raw_analysis in raw_analyses.values()):
        logger.error("Skipping full JSON formatting due to failure in raw analysis generation for one or more sections.")
        error_json = json.loads(JSON_SKELETON)
        error_json["document"]["filename"] = filename
        error_json["document"]["title"] = title
        error_json["document"]["size_kb"] = file_size_kb
        error_json["document"]["upload_date_utc"] = upload_date_utc
        error_json["analysis_sections"]["general_equity_assessment"]["summary"] = "One or more raw analyses failed during text generation. See logs for details."
        # Clear/default other fields to reflect failure
        error_json["equity_analysis_by_perspective"] = []
        error_json["overall_summary_and_recommendations"]["key_equity_gaps"] = "Due to incomplete raw analysis generation."
        # Still attempt to populate sources for any sections that might have succeeded
        _populate_sources_into_json(error_json, raw_analyses)
        return error_json


    # Dynamically build the raw analyses string (text only) to include in the prompt for the formatter LLM
    raw_analyses_text_str = ""
    for k, v in raw_analyses.items():
        raw_analyses_text_str += f"\n--- RAW ANALYSIS TEXT FOR: {k.replace('_', ' ').upper()} ---\n{v.get('text', 'Not provided.')}\n"

    formatter_prompt = textwrap.dedent(f"""
    You are an expert data structurer and equity analyst. Your task is to populate the provided JSON structure.
    All generated content in the JSON must be derived **ONLY** from the raw analysis texts provided below.
    Crucially, all summary, narrative, and description fields must use an **indicative, tentative, or suggestive tone**.
    Avoid definitive or authoritative statements. Employ phrases like "This may indicate...", "It suggests that...",
    "A potential interpretation is...", "It appears to...", "Could be seen as...", "There is an indication that...",
    "The document seems to...", "It might imply...", etc.

    **Instructions:**
    1.  Carefully read all provided raw text analyses.
    2.  Fill in every "..." placeholder in the JSON skeleton with **detailed and comprehensive** information synthesized from these analysis texts. Do not over-summarize; preserve key details and nuanced interpretations.
    3.  Ensure all content strictly adheres to the schema and the required indicative tone.
    4.  **DO NOT ADD ANY SOURCE INFORMATION OR CITATIONS TO THE TEXT FIELDS OR THE 'sources' ARRAYS.** The 'sources' arrays in the JSON skeleton will be populated separately by Python.
    5.  Specifically for the `general_equity_assessment` within `analysis_sections`, break down the 'general' raw analysis into the sub-fields for each of the four equity dimensions (Recognitional, Procedural, Distributional, Structural). For each dimension, aim to identify both "positive_findings" and "concerns" if discernible, and provide a "conclusion". If information is not provided for a sub-field, use "Not explicitly indicated by the document." or similar indicative phrasing.
    6.  For the `equity_analysis_by_perspective` array, create an entry for each perspective mentioned in the raw analyses (e.g., Policy Makers, Residents, Farmers/Business Owners). For each perspective entry:
        *   Populate the `group` name (e.g., "Policy Makers").
        *   Fill the `general_equity_assessment` (title and narrative) using the corresponding raw analysis text (e.g., the content for key `perspective_policymakers_general`). The title should reflect the group's perspective.
        *   Fill the individual equity dimension descriptions (recognitional_equity, procedural_equity, distributional_equity, structural_equity) using the specific raw analysis texts for that dimension and group (e.g., the content for key `perspective_policymakers_recognitional`).
    7.  Ensure the output is a single, valid JSON object and nothing else.

    **JSON SKELETON TO POPULATE (Text fields only):**
    {JSON_SKELETON}

    **RAW TEXT ANALYSES TO USE:**
    {raw_analyses_text_str}
    """)

    try:
        response = client.chat.completions.create(
            model=settings.OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                {"role": "user", "content": formatter_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2 # Keep temperature low for structured output
        )
        json_output_str = response.choices[0].message.content
        structured_data = json.loads(json_output_str)

        # Assign document metadata
        structured_data["document"]["filename"] = filename
        structured_data["document"]["title"] = title
        structured_data["document"]["size_kb"] = file_size_kb
        structured_data["document"]["upload_date_utc"] = upload_date_utc

        # --- Python Logic to Inject Raw Sources (POST-LLM) ---
        _populate_sources_into_json(structured_data, raw_analyses)

        logger.info("-> Successfully synthesized analyses into JSON structure and injected raw sources.")
        return structured_data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from OpenAI response during formatting: {e}. Response was: {json_output_str[:500]}...", exc_info=True)
        # Attempt to return a partial JSON indicating formatting failure and populate sources
        error_json = json.loads(JSON_SKELETON)
        error_json["document"]["filename"] = filename
        error_json["document"]["title"] = title
        error_json["document"]["size_kb"] = file_size_kb
        error_json["document"]["upload_date_utc"] = upload_date_utc
        error_json["analysis_sections"]["general_equity_assessment"]["summary"] = f"JSON formatting failed: {e}. Raw analyses might be incomplete or malformed."
        error_json["equity_analysis_by_perspective"] = []
        error_json["overall_summary_and_recommendations"]["key_equity_gaps"] = "Due to JSON formatting failure."
        _populate_sources_into_json(error_json, raw_analyses) # Still try to populate sources on error
        return error_json
    except Exception as e:
        logger.error(f"Failed to format analyses into JSON for unknown reason: {e}", exc_info=True)
        error_json = json.loads(JSON_SKELETON)
        error_json["document"]["filename"] = filename
        error_json["document"]["title"] = title
        error_json["document"]["size_kb"] = file_size_kb
        error_json["document"]["upload_date_utc"] = upload_date_utc
        error_json["analysis_sections"]["general_equity_assessment"]["summary"] = f"An unexpected error occurred during final JSON synthesis: {e}"
        error_json["equity_analysis_by_perspective"] = []
        error_json["overall_summary_and_recommendations"]["key_equity_gaps"] = "Due to unexpected error during JSON formatting."
        _populate_sources_into_json(error_json, raw_analyses) # Still try to populate sources on error
        return error_json

def _populate_sources_into_json(structured_data: Dict[str, Any], raw_analyses: Dict[str, Dict[str, Any]]):
    """
    Helper function to inject raw openai_sources into the structured_data JSON.
    Local sources are explicitly excluded as per new requirement.
    This is done purely by Python after the LLM has filled the text fields.
    """
    
    # Mapping from raw_analyses keys to structured_data paths for sources
    source_mappings = {
        # Standard Focus Areas
        "general": structured_data["analysis_sections"]["general_equity_assessment"],
        "vulnerable_groups": structured_data["analysis_sections"]["vulnerable_groups_analysis"],
        "severity_of_impact": structured_data["analysis_sections"]["severity_impact_analysis"],
        "mitigation_strategies": structured_data["analysis_sections"]["mitigation_strategies_analysis"],
    }

    # Add Perspective-Based mappings
    for perspective_info in PERSPECTIVES:
        group_key = perspective_info["group_name"].replace(" ", "_").lower()
        
        # Find the correct perspective entry in structured_data
        # This assumes the LLM successfully created the perspective entries.
        perspective_entry = next((item for item in structured_data["equity_analysis_by_perspective"] if item.get("group") == perspective_info["group_name"]), None)
        
        if perspective_entry:
            source_mappings[f"perspective_{group_key}_general"] = perspective_entry["general_equity_assessment"]
            for dim in ["recognitional", "procedural", "distributional", "structural"]:
                source_mappings[f"perspective_{group_key}_{dim}"] = perspective_entry[f"{dim}_equity"]
        else:
            logger.warning(f"Could not find perspective entry for '{perspective_info['group_name']}' in structured_data. Sources for this perspective will not be added.")

    # Populate sources for each mapped section
    for key, target_section in source_mappings.items():
        if key in raw_analyses:
            # ONLY include openai_sources as per requirement
            raw_openai_sources = raw_analyses[key].get("openai_sources", [])
            
            # Create the list of source objects, only for OpenAI type
            combined_sources = []
            for openai_source_str in raw_openai_sources:
                combined_sources.append({"type": "openai", "data": openai_source_str})
            
            # Assign to the target section's "sources" field
            target_section["sources"] = combined_sources
        else:
            logger.debug(f"Raw analysis key '{key}' not found, skipping source injection for corresponding section.")

    # Handle overall_summary_and_recommendations sources
    # For simplicity, copy sources from the main general_equity_assessment if it exists.
    if "overall_summary_and_recommendations" in structured_data and "general" in raw_analyses:
        main_general_openai_sources = []
        for openai_source_str in raw_analyses["general"].get("openai_sources", []):
            main_general_openai_sources.append({"type": "openai", "data": openai_source_str})
        structured_data["overall_summary_and_recommendations"]["sources"] = main_general_openai_sources
    else:
        structured_data["overall_summary_and_recommendations"]["sources"] = [] # Ensure it's an empty list if no sources.


def main():
    logger.info("--- Starting Batch Document Analysis ---")

    if not settings:
        logger.critical("Settings not loaded. Exiting.")
        return
    if not os.path.exists(DOCUMENTS_FOLDER):
        logger.error(f"Folder '{DOCUMENTS_FOLDER}' not found. Exiting.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output directory '{OUTPUT_DIR}' ensured.")

    try:
        openai_interface = OpenAIInteraction()
        logger.info("OpenAI Interaction layer initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI Interaction: {e}", exc_info=True)
        return

    # Load local DB on startup (if configured) - this is for HybridRAGSystem
    load_db_on_startup()
    if get_local_db() is None and settings.ENABLE_LOCAL_RAG:
        logger.warning("Local DB did not load successfully. Local RAG component might be unavailable.")
    elif get_local_db() is not None:
        logger.info(f"Local DB loaded with {len(get_local_db().documents)} documents.")

    try:
        rag_system = HybridRAGSystem(openai_interaction=openai_interface)
        logger.info("Hybrid RAG System initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize RAG System: {e}", exc_info=True)
        return

    pdf_files = [f for f in os.listdir(DOCUMENTS_FOLDER) if f.lower().endswith('.pdf')]
    if not pdf_files:
        logger.warning(f"No PDF files found in '{DOCUMENTS_FOLDER}'. Exiting.")
        return
    logger.info(f"Found {len(pdf_files)} PDF documents to analyze.")

    for filename in pdf_files:
        base_filename_no_ext = os.path.splitext(filename)[0]
        output_json_path = os.path.join(OUTPUT_DIR, f"{base_filename_no_ext}.json")

        if os.path.exists(output_json_path):
            logger.info(f"Skipping '{filename}' as '{output_json_path}' already exists.")
            continue

        logger.info(f"\n--- Analyzing document: {filename} ---")

        with tempfile.TemporaryDirectory() as temp_dir:
            original_file_path = os.path.join(DOCUMENTS_FOLDER, filename)
            temp_file_path = os.path.join(temp_dir, filename)

            try:
                shutil.copy2(original_file_path, temp_file_path)
            except Exception as e:
                logger.critical(f"Could not create temp copy for '{filename}': {e}. Skipping this document.", exc_info=True)
                continue # Skip to next document instead of stopping script

            # Unique session ID per document, incorporating timestamp
            session_id = f"batch_analysis_{base_filename_no_ext.replace('.', '_')}_{int(time.time())}"
            title = get_pdf_title(original_file_path, filename)
            file_size_kb = os.path.getsize(original_file_path) // 1024 # Size in KB
            upload_date_utc = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(os.path.getmtime(original_file_path)))

            raw_analyses: Dict[str, Dict[str, Any]] = {} # Store text, and openai_sources (local sources explicitly excluded)
            final_json_result = None

            try:
                logger.info(f"Uploading '{filename}' to OpenAI and processing locally for session {session_id}...")
                success, message = rag_system.add_user_document_for_session(
                    session_id=session_id, file_path=temp_file_path, original_filename=filename
                )
                if not success:
                    raise Exception(f"Document processing failed: {message}")

                # --- Generate analyses for standard FOCUS_AREAS ---
                focus_description_map = {
                    "general": "the overall equity implications, considering all relevant dimensions of the COEQWAL framework.",
                    "vulnerable_groups": "how vulnerable groups are affected or mentioned.",
                    "severity_of_impact": "the severity of the document's impacts on equity.",
                    "mitigation_strategies": "strategies or solutions for equity concerns."
                }
                for focus in FOCUS_AREAS:
                    logger.info(f"-> Generating raw analysis for focus: '{focus}'...")
                    query_for_rag_system = ANALYSIS_QUERY_GENERIC.format(focus_description=focus_description_map.get(focus, "equity implications."))
                    
                    # NOTE: _ means local_chunks are discarded here as per requirement
                    answer, _, openai_srcs = rag_system.answer_question(session_id=session_id, query=query_for_rag_system, focus_area=focus)
                    
                    if "Error:" in answer:
                        logger.error(f"Received an error for standard focus '{focus}': {answer}. Marking as failed.")
                        raw_analyses[focus] = {"text": f"ANALYSIS FAILED: {answer}", "openai_sources": openai_srcs}
                    else:
                        raw_analyses[focus] = {"text": answer, "openai_sources": openai_srcs}
                    logger.info(f"Generated raw analysis for '{focus}'. Waiting for {DELAY_BETWEEN_REQUESTS_SECONDS}s...")
                    time.sleep(DELAY_BETWEEN_REQUESTS_SECONDS)

                # --- Generate analyses for each PERSPECTIVE ---
                for perspective_info in PERSPECTIVES:
                    perspective_group_key = perspective_info["group_name"].replace(" ", "_").lower()
                    logger.info(f"-> Generating analysis for perspective: '{perspective_info['group_name']}'...")

                    # General Equity Assessment for this perspective
                    query_general_perspective = ANALYSIS_QUERY_GENERIC.format(focus_description=perspective_info['description'])
                    answer_general, _, openai_srcs = rag_system.answer_question(session_id=session_id, query=query_general_perspective, focus_area="general") # Use general focus for retrieval
                    if "Error:" in answer_general:
                        logger.error(f"Error for perspective '{perspective_info['group_name']}' general analysis: {answer_general}. Marking as failed.")
                        raw_analyses[f"perspective_{perspective_group_key}_general"] = {"text": f"ANALYSIS FAILED: {answer_general}", "openai_sources": openai_srcs}
                    else:
                        raw_analyses[f"perspective_{perspective_group_key}_general"] = {"text": answer_general, "openai_sources": openai_srcs}
                    logger.info(f"Generated general analysis for '{perspective_info['group_name']}'. Waiting for {DELAY_BETWEEN_REQUESTS_SECONDS}s...")
                    time.sleep(DELAY_BETWEEN_REQUESTS_SECONDS)

                    # Individual Equity Dimensions for this perspective
                    for dim in ["recognitional", "procedural", "distributional", "structural"]: # Only 4 dimensions
                        prompt_description = perspective_info["dimensions"].get(dim)
                        if prompt_description:
                            query_dim_perspective = ANALYSIS_QUERY_GENERIC.format(focus_description=prompt_description)
                            answer_dim, _, openai_srcs = rag_system.answer_question(session_id=session_id, query=query_dim_perspective, focus_area="general") # Use general focus for retrieval
                            if "Error:" in answer_dim:
                                logger.error(f"Error for perspective '{perspective_info['group_name']}' {dim} analysis: {answer_dim}. Marking as failed.")
                                raw_analyses[f"perspective_{perspective_group_key}_{dim}"] = {"text": f"ANALYSIS FAILED: {answer_dim}", "openai_sources": openai_srcs}
                            else:
                                raw_analyses[f"perspective_{perspective_group_key}_{dim}"] = {"text": answer_dim, "openai_sources": openai_srcs}
                            logger.info(f"Generated '{dim}' analysis for '{perspective_info['group_name']}'. Waiting for {DELAY_BETWEEN_REQUESTS_SECONDS}s...")
                            time.sleep(DELAY_BETWEEN_REQUESTS_SECONDS)

                # --- Synthesize all raw analyses text into final JSON structure (Python injects sources after) ---
                final_json_result = format_analyses_into_json(raw_analyses, filename, title, file_size_kb, upload_date_utc, openai_interface.client)
                if not final_json_result:
                    raise Exception("Failed to synthesize the final JSON structure from raw analyses.")

            except Exception as e:
                logger.critical(f"A critical error occurred while processing '{filename}': {e}. Cleaning up and skipping this document.", exc_info=True)
                # Attempt to save a partial/error JSON before cleaning up
                # Use a simplified error structure that Python can easily populate
                error_json_for_file = {
                    "document": { "filename": filename, "title": title, "size_kb": file_size_kb, "upload_date_utc": upload_date_utc },
                    "analysis_sections": { "general_equity_assessment": { "summary": f"Analysis failed due to critical error: {e}", "sources": [] } },
                    "equity_analysis_by_perspective": [],
                    "overall_summary_and_recommendations": { "key_equity_gaps": f"Due to critical error: {e}", "sources": [] }
                }
                # Even on critical failure, try to populate sources for any successful raw analyses
                _populate_sources_into_json(error_json_for_file, raw_analyses)
                
                try:
                    with open(output_json_path, 'w', encoding='utf-8') as f:
                        json.dump(error_json_for_file, f, indent=2, ensure_ascii=False)
                    logger.info(f"Error report saved to '{output_json_path}'.")
                except Exception as write_e:
                    logger.error(f"Failed to write error report for '{filename}': {write_e}")
                # Ensure OpenAI resources are cleaned even if the analysis failed
                logger.info(f"Cleaning up OpenAI resources for failed session '{session_id}'...")
                rag_system.remove_user_session_resources(session_id, delete_openai_resources=True)
                continue # Move to the next document

            logger.info(f"Cleaning up OpenAI resources for session '{session_id}'...")
            rag_system.remove_user_session_resources(session_id, delete_openai_resources=True)

            try:
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(final_json_result, f, indent=2, ensure_ascii=False)
                logger.info(f"Analysis for '{filename}' saved successfully to '{output_json_path}'.")
            except Exception as e:
                logger.critical(f"Could not write final JSON for '{filename}' to '{output_json_path}': {e}. Continuing to next document, but data might be lost.", exc_info=True)
                continue # Continue to next document even if saving current fails

    logger.info("\n--- Batch Analysis Complete ---")

if __name__ == "__main__":
    # Ensure OUTPUT_DIR exists before starting
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Check for necessary libraries for local processing.
    if not PdfReader:
        logger.warning("PyPDF2 is not installed. PDF title extraction will be limited. Run: pip install PyPDF2")
    # Note: `fitz`, `docx`, and `bs4` are used in `core/text_processing.py`.
    # This script (generate_batch_analysis.py) doesn't directly import/use them for chunking,
    # but `HybridRAGSystem` might rely on `process_document_to_chunks` from `text_processing`.
    # The warnings below are good for user awareness if `text_processing` uses them.
    try:
        import fitz
    except ImportError:
        logger.warning("PyMuPDF (fitz) is not installed. PDF document processing for local RAG might be affected.")
    try:
        from docx import Document
    except ImportError:
        logger.warning("python-docx is not installed. DOCX document processing for local RAG might be affected.")
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("BeautifulSoup4 is not installed. HTML document processing for local RAG might be basic.")

    main()