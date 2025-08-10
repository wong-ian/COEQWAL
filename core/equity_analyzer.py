# core/equity_analyzer.py

import os
import json
import logging
import shutil
import tempfile
import textwrap
import time
import asyncio

from typing import Dict, Any, List, Optional, Tuple
from openai import OpenAI

try:
    from PyPDF2 import PdfReader
except ImportError:
    logging.warning("PyPDF2 is not installed. PDF title extraction might be limited.")
    PdfReader = None

from .config import settings
from .openai_interaction import OpenAIInteraction
from .rag_system import HybridRAGSystem

logger = logging.getLogger("equity_analyzer")

# --- Constants and JSON Skeleton (remain unchanged) ---
FOCUS_AREAS = ["general", "vulnerable_groups", "severity_of_impact", "mitigation_strategies"]
ANALYSIS_QUERY_GENERIC = "Provide an equity analysis of this document, focusing on: {focus_description}"
DELAY_BETWEEN_REQUESTS_SECONDS = 5
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

# --- Helper Functions (remain unchanged) ---
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
            response_format={"type": "json_object"}
            #temperature=0.2 # Keep temperature low for structured output - COMMENTED OUT
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

async def perform_equity_analysis(
    session_id: str,
    temp_file_path: str, # Path to the temporarily saved file (not the original one in Documents)
    original_filename: str,
    title: str,
    file_size_kb: int,
    upload_date_utc: str,
    rag_system_instance: HybridRAGSystem,
    openai_interface_instance: OpenAIInteraction,
    user_sessions_dict: Dict[str, Dict[str, Any]], # Direct reference to the main app's user_sessions dict
    analysis_output_dir: str # Directory to save the final JSON file
):
    """
    Performs the full equity analysis for a single document as a background task.
    Updates session status and saves results to file and in-memory dict.
    """
    logger.info(f"Background task: Starting analysis for session: {session_id}, file: {original_filename}")
    
    session_info = user_sessions_dict.get(session_id)
    if not session_info:
        logger.error(f"Session {session_id} not found in user_sessions_dict during background analysis start.")
        if os.path.exists(temp_file_path):
            try: os.remove(temp_file_path); logger.info(f"[{session_id}] Deleted temp file due to missing session info.")
            except OSError as e: logger.error(f"[{session_id}] Error deleting temp file (missing session info): {e}")
        return

    session_info['analysis_status'] = 'in_progress'
    session_info['analysis_result_cached'] = None
    session_info['analysis_result_path'] = None
    session_info['analysis_error'] = None

    # --- MAIN TRY BLOCK FOR THE ENTIRE ANALYSIS PROCESS ---
    try:
        if settings.SIMULATE_ANALYSIS:
            logger.info(f"[{session_id}] Running analysis in SIMULATION MODE (no OpenAI calls).")
            # Simulate some work time
            await asyncio.sleep(2) 
            
            dummy_json_result = {
                "document": {
                    "filename": original_filename,
                    "title": title or "Simulated Document Title",
                    "size_kb": file_size_kb,
                    "upload_date_utc": upload_date_utc
                },
                "analysis_sections": {
                    "general_equity_assessment": {
                        "title": "General Equity Assessment (Simulated)",
                        "summary": "This is a simulated general equity assessment. It provides a placeholder for detailed analysis content when in simulation mode. The document appears to touch upon various aspects that could pertain to equitable distribution and fair processes, hinting at areas where further real analysis would elaborate on both strengths and potential concerns. This section aims to mimic a comprehensive output without actual LLM processing.",
                        "sources": [{"type": "openai", "data": "Simulated Source from Doc A: <blockquote>This is a dummy chunk that would normally be retrieved by OpenAI file search during real analysis. It demonstrates source inclusion.</blockquote>"}]
                    },
                    "vulnerable_groups_analysis": {
                        "title": "Vulnerable Groups Analysis (Simulated)",
                        "summary": "In simulation, the analysis might suggest a focus on how policies could implicitly affect vulnerable populations, even if not explicitly stated. A real analysis would identify specific groups and detailed impacts.",
                        "identified_groups_and_impacts": "Placeholder for identified groups and their simulated impacts.",
                        "equity_assessment_summary": "Simulated equity summary for vulnerable groups.",
                        "conclusion": "Simulated conclusion for vulnerable groups.",
                        "sources": [{"type": "openai", "data": "Simulated Source from Doc A: <blockquote>Dummy source related to vulnerable groups.</blockquote>"}]
                    },
                    "severity_impact_analysis": {
                        "title": "Severity of Impact Analysis (Simulated)",
                        "summary": "Simulated summary on impact severity. A real analysis would detail high and moderate severity impacts on equity.",
                        "high_severity_impacts": "Placeholder for high severity impacts.",
                        "moderate_severity_impacts": "Placeholder for moderate severity impacts.",
                        "equity_implications_of_impacts": "Simulated equity implications.",
                        "conclusion": "Simulated conclusion on impact severity.",
                        "sources": [{"type": "openai", "data": "Simulated Source from Doc A: <blockquote>Dummy source related to impact severity.</blockquote>"}]
                    },
                    "mitigation_strategies_analysis": {
                        "title": "Mitigation Strategies Analysis (Simulated)",
                        "summary": "Simulated summary of mitigation strategies. Real analysis would identify and assess their fairness and effectiveness comprehensively.",
                        "identified_strategies": "Placeholder for identified strategies.",
                        "equity_assessment": "Simulated equity assessment of strategies.",
                        "conclusion": "Simulated conclusion on strategies.",
                        "sources": [{"type": "openai", "data": "Simulated Source from Doc A: <blockquote>Dummy source related to mitigation strategies.</blockquote>"}]
                    }
                },
                "equity_analysis_by_perspective": [
                    {
                        "group": "Policy Makers",
                        "general_equity_assessment": {
                            "title": "Policy Makers Perspective (Simulated)",
                            "narrative": "A simulated narrative from the policy makers' perspective, indicating their role and potential equity considerations. This would usually be several detailed paragraphs.",
                            "sources": [{"type": "openai", "data": "Simulated Source from Policy Doc: <blockquote>Dummy source from policy maker context.</blockquote>"}]
                        },
                        "recognitional_equity": { "description": "Simulated description of recognitional equity from this perspective.", "sources": [] },
                        "procedural_equity": { "description": "Simulated description of procedural equity from this perspective.", "sources": [] },
                        "distributional_equity": { "description": "Simulated description of distributional equity from this perspective.", "sources": [] },
                        "structural_equity": { "description": "Simulated description of structural equity from this perspective.", "sources": [] }
                    },
                    {
                        "group": "Residents",
                        "general_equity_assessment": {
                            "title": "Residents Perspective (Simulated)",
                            "narrative": "A simulated narrative from the residents' perspective, focusing on their experiences with water policy and potential equity impacts. Expect more detail in a real analysis.",
                            "sources": [{"type": "openai", "data": "Simulated Source from Resident Testimony: <blockquote>Dummy source from resident context.</blockquote>"}]
                        },
                        "recognitional_equity": { "description": "Simulated description of recognitional equity from this perspective.", "sources": [] },
                        "procedural_equity": { "description": "Simulated description of procedural equity from this perspective.", "sources": [] },
                        "distributional_equity": { "description": "Simulated description of distributional equity from this perspective.", "sources": [] },
                        "structural_equity": { "description": "Simulated description of structural equity from this perspective.", "sources": [] }
                    }
                ],
                "overall_summary_and_recommendations": {
                    "title": "Overall Summary & Recommendations (Simulated)",
                    "key_equity_gaps": "Simulated key equity gaps, indicating areas for improvement.",
                    "key_equity_strengths": "Simulated key equity strengths, highlighting positive aspects.",
                    "recommendations": "Simulated recommendations, suggesting future actions for equity enhancement.",
                    "sources": [{"type": "openai", "data": "Simulated Overall Source: <blockquote>A general dummy source for the overall summary.</blockquote>"}]
                }
            }
            # Save to file
            output_file_path = os.path.join(analysis_output_dir, f"{session_id}.json")
            os.makedirs(analysis_output_dir, exist_ok=True)
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(dummy_json_result, f, indent=2, ensure_ascii=False)
            logger.info(f"[{session_id}] Simulated analysis saved to file: {output_file_path}")

            # Update in-memory dict
            session_info['analysis_status'] = 'completed'
            session_info['analysis_result_path'] = output_file_path
            session_info['analysis_result_cached'] = dummy_json_result
            logger.info(f"[{session_id}] Simulated analysis completed and session info updated.")

        else: # --- REAL ANALYSIS LOGIC ---
            logger.info(f"[{session_id}] Running analysis in REAL MODE (making OpenAI calls).")
            raw_analyses: Dict[str, Dict[str, Any]] = {}
            final_json_result: Optional[Dict[str, Any]] = None
            
            # This entire real analysis process runs within the main try block
            # No inner try/except needed here.

            # --- Generate analyses for standard FOCUS_AREAS ---
            focus_description_map = {
                "general": "the overall equity implications, considering all relevant dimensions of the COEQWAL framework.",
                "vulnerable_groups": "how vulnerable groups are affected or mentioned.",
                "severity_of_impact": "the severity of the document's impacts on equity.",
                "mitigation_strategies": "strategies or solutions for equity concerns."
            }
            for focus in FOCUS_AREAS:
                logger.info(f"[{session_id}] -> Generating raw analysis for focus: '{focus}'...")
                query_for_rag_system = ANALYSIS_QUERY_GENERIC.format(focus_description=focus_description_map.get(focus, "equity implications."))
                
                answer, _, openai_srcs = rag_system_instance.answer_question(session_id=session_id, query=query_for_rag_system, focus_area=focus)
                
                if "Error:" in answer:
                    logger.error(f"[{session_id}] Received an error for standard focus '{focus}': {answer}. Marking as failed.")
                    raw_analyses[focus] = {"text": f"ANALYSIS FAILED: {answer}", "openai_sources": openai_srcs}
                else:
                    raw_analyses[focus] = {"text": answer, "openai_sources": openai_srcs}
                logger.info(f"[{session_id}] Generated raw analysis for '{focus}'. Waiting for {DELAY_BETWEEN_REQUESTS_SECONDS}s...")
                time.sleep(DELAY_BETWEEN_REQUESTS_SECONDS)

            # --- Generate analyses for each PERSPECTIVE ---
            for perspective_info in PERSPECTIVES:
                perspective_group_key = perspective_info["group_name"].replace(" ", "_").lower()
                logger.info(f"[{session_id}] -> Generating analysis for perspective: '{perspective_info['group_name']}'...")

                # General Equity Assessment for this perspective
                query_general_perspective = ANALYSIS_QUERY_GENERIC.format(focus_description=perspective_info['description'])
                answer_general, _, openai_srcs = rag_system_instance.answer_question(session_id=session_id, query=query_general_perspective, focus_area="general")
                if "Error:" in answer_general:
                    logger.error(f"[{session_id}] Error for perspective '{perspective_info['group_name']}' general analysis: {answer_general}. Marking as failed.")
                    raw_analyses[f"perspective_{perspective_group_key}_general"] = {"text": f"ANALYSIS FAILED: {answer_general}", "openai_sources": openai_srcs}
                else:
                    raw_analyses[f"perspective_{perspective_group_key}_general"] = {"text": answer_general, "openai_sources": openai_srcs}
                logger.info(f"[{session_id}] Generated general analysis for '{perspective_info['group_name']}'. Waiting for {DELAY_BETWEEN_REQUESTS_SECONDS}s...")
                time.sleep(DELAY_BETWEEN_REQUESTS_SECONDS)

                # Individual Equity Dimensions for this perspective
                for dim in ["recognitional", "procedural", "distributional", "structural"]:
                    prompt_description = perspective_info["dimensions"].get(dim)
                    if prompt_description:
                        query_dim_perspective = ANALYSIS_QUERY_GENERIC.format(focus_description=prompt_description)
                        answer_dim, _, openai_srcs = rag_system_instance.answer_question(session_id=session_id, query=query_dim_perspective, focus_area="general")
                        if "Error:" in answer_dim:
                            logger.error(f"[{session_id}] Error for perspective '{perspective_info['group_name']}' {dim} analysis: {answer_dim}. Marking as failed.")
                            raw_analyses[f"perspective_{perspective_group_key}_{dim}"] = {"text": f"ANALYSIS FAILED: {answer_dim}", "openai_sources": openai_srcs}
                        else:
                            raw_analyses[f"perspective_{perspective_group_key}_{dim}"] = {"text": answer_dim, "openai_sources": openai_srcs}
                        logger.info(f"[{session_id}] Generated '{dim}' analysis for '{perspective_info['group_name']}'. Waiting for {DELAY_BETWEEN_REQUESTS_SECONDS}s...")
                        time.sleep(DELAY_BETWEEN_REQUESTS_SECONDS)

            # --- Synthesize all raw analyses text into final JSON structure (Python injects sources after) ---
            final_json_result = format_analyses_into_json(raw_analyses, original_filename, title, file_size_kb, upload_date_utc, openai_interface_instance.client)
            if not final_json_result:
                raise Exception("Failed to synthesize the final JSON structure from raw analyses.")

            # --- Save result to file and update in-memory dict ---
            output_file_path = os.path.join(analysis_output_dir, f"{session_id}.json")
            os.makedirs(analysis_output_dir, exist_ok=True)
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(final_json_result, f, indent=2, ensure_ascii=False)
            logger.info(f"[{session_id}] Analysis saved to file: {output_file_path}")

            # Update in-memory dict
            session_info['analysis_status'] = 'completed'
            session_info['analysis_result_path'] = output_file_path
            session_info['analysis_result_cached'] = final_json_result
            logger.info(f"[{session_id}] Analysis completed and session info updated.")

    except Exception as e: # This is the specific catch for REAL analysis errors.
        logger.error(f"[{session_id}] Critical error during background REAL analysis: {e}", exc_info=True)
        session_info['analysis_status'] = 'failed'
        session_info['analysis_error'] = str(e)
    # --- END REAL ANALYSIS LOGIC ---

    finally: # This finally block encapsulates the entire function's execution.
        # Ensure the temporary file is deleted after processing (or failure)
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"[{session_id}] Deleted temporary file: {temp_file_path}")
            except OSError as e:
                logger.error(f"[{session_id}] Error deleting temporary file {temp_file_path}: {e}")