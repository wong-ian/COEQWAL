# core/rag_system.py
import os
import textwrap
import logging
import time
import json
from typing import Optional, List, Dict, Any, Tuple
from openai import APIError, APIStatusError, RateLimitError

from .config import settings
from .local_db import VectorDatabase, get_local_db
from .openai_interaction import OpenAIInteraction

logger = logging.getLogger("rag_system")

class HybridRAGSystem:
    def __init__(self, openai_interaction: OpenAIInteraction):
        self.config = settings
        self.openai_interaction = openai_interaction
        self.local_db = get_local_db()
        if self.local_db is None: logger.warning("HybridRAGSystem initialized WITHOUT a functional local database.")
        else: logger.info("HybridRAGSystem initialized WITH local database.")
        self.user_sessions: Dict[str, Dict[str, Any]] = {}

    def add_user_document_for_session(self, session_id: str, file_path: str, original_filename: str) -> Tuple[bool, str]:
        logger.info(f"Processing user document for session '{session_id}': '{original_filename}' from path '{file_path}'")

        # --- ASSUME session_id is ALREADY initialized in self.user_sessions by main.py ---
        # If it's not, that's an error in the calling code (main.py's upload_document)
        if session_id not in self.user_sessions:
            logger.error(f"Session {session_id} not pre-initialized in user_sessions for add_user_document_for_session.")
            return False, "Internal error: Session not tracked correctly."
        
        # Update the status to reflect the current phase
        self.user_sessions[session_id]["status"] = "uploading_file"

        file_id = self.openai_interaction.upload_file(file_path, purpose="assistants")
        if not file_id:
            msg = f"Failed to upload file {original_filename} for session {session_id}."
            logger.error(msg)
            self.user_sessions[session_id]["status"] = "failed_upload"
            return False, msg
        
        self.user_sessions[session_id]["file_id"] = file_id 
        self.user_sessions[session_id]["status"] = "creating_vs"

        vs_name = f"vs_{session_id}_{original_filename}".replace(" ", "_")[:100]
        vector_store_id = self.openai_interaction.create_vector_store_with_files(name=vs_name, file_ids=[file_id])
        if not vector_store_id:
            msg = f"Failed to create Vector Store for file ID {file_id} (session {session_id}). Cleaning up."
            logger.error(msg)
            # Only delete the file if VS creation failed, but keep it tracked in user_sessions for overall session cleanup
            self.openai_interaction.delete_file(file_id) 
            self.user_sessions[session_id]["status"] = "failed_vs_creation"
            return False, msg
        
        self.user_sessions[session_id]["vector_store_id"] = vector_store_id 
        self.user_sessions[session_id]["status"] = "vs_processing"

        processing_success = self.openai_interaction.wait_for_vector_store_file_processing(vector_store_id=vector_store_id, file_id=file_id)
        
        # Final status update for the upload/VS processing part
        self.user_sessions[session_id]["status"] = "completed" if processing_success else "failed_vs_processing"

        if not processing_success:
            msg = f"File '{original_filename}' (ID: {file_id}) failed processing in VS {vector_store_id} for session {session_id}. File search may fail."
            logger.error(msg)
            return False, msg
        else:
            msg = f"User document '{original_filename}' processed successfully for session '{session_id}'. Ready for queries."
            logger.info(msg)
            return True, msg

    def _get_system_prompt(self, focus_area: str, original_filename: str, local_context_str: str, query: str, custom_instructions: Optional[str] = None) -> str:
        """
        Selects or generates the appropriate system prompt based on the focus area.
        Ensures an indicative, tentative, or suggestive tone while preserving directives.
        **Emphasizes generating more content for sections.**
        """
        base_intro = f"""
            **Framework Definitions (from COEQWAL Context - Four Equity Dimensions):**
            - **Recognitional Equity:** Concerns the fair and inclusive recognition of diverse groups, their unique identities, histories, and cultural values in policies, processes, and outcomes. It asks if all voices are seen and valued.
            - **Procedural Equity:** Focuses on fair and inclusive processes for decision-making. It examines whether all affected parties have meaningful opportunities to participate, influence, and access information.
            - **Distributional Equity:** Addresses the fair and just distribution of benefits and burdens. It questions whether resources, services, and environmental risks are equitably shared among all groups, avoiding disproportionate impacts on any particular community.
            - **Structural Equity:** Seeks to identify and address the underlying systemic barriers, institutional practices, and power imbalances that perpetuate inequities. It aims to transform these structures to create a more just society.

            --- START CONTEXT FROM COEQWAL DOCUMENT ---
            {local_context_str if local_context_str else 'No definitions or context from the COEQWAL document were retrieved from the COEQWAL Framework document.'}
            --- END CONTEXT FROM COEQWAL DOCUMENT ---

            **User Document Name:** {original_filename}
            **User Query:** {query}

            **IMPORTANT - Output Style & Tone:** Please write your analysis in clear, accessible language. Avoid overly academic or technical jargon.
            Crucially, your analysis should be presented in a **tentative, suggestive, or indicative tone**. Avoid definitive or authoritative statements. You might use phrases such as: "This could suggest...", "It may indicate...", "A possible interpretation is...", "It appears to...", "Could be seen as...", "There seems to be an indication that...", "The document seems to imply...", "It might be perceived as...", "It might suggest the presence of...". If information is not explicitly available in the document, you may state that it is not directly mentioned or that the document does not appear to provide sufficient detail.
            **Provide a comprehensive response, aiming for detailed analysis and multiple paragraphs where appropriate for each section/point.** Elaborate thoroughly on each finding. Conclude your response with a bulleted summary of the key findings.
        """
        
        if focus_area == "custom" and custom_instructions:
            return textwrap.dedent(f"""
                **Your Task:** You are an equity analyst. Your goal is to analyze the uploaded 'User Document' based on a specific set of custom instructions provided by the user. You should strive to follow these instructions while using the COEQWAL Equity Framework as a guiding lens.

                {base_intro}

                **User's Custom Focus Instructions:**
                --- START OF USER INSTRUCTIONS ---
                {custom_instructions}
                --- END OF USER INSTRUCTIONS ---

                **Your Analysis Steps:**
                1.  Thoroughly consider the User's Custom Focus Instructions.
                2.  You may identify and extract all parts from the User Document that appear relevant to these instructions and the user's query.
                3.  Where applicable, you might consider how the COEQWAL dimensions (Recognition, Procedure, Distribution, Structure) could help illuminate the analysis as per the user's instructions.
                4.  Strive to provide a balanced view, discussing both potential strengths and possible weaknesses that you might identify.
                5.  Where possible, you may refer to instances or examples from the User Document that could support your observations.
                6.  If the document seems to lack the necessary information to follow the instructions, it may be appropriate to state this limitation.

                **Final Output:** Provide a detailed analysis that directly addresses the User's Custom Focus Instructions. Start with a clear overview, then offer the detailed analysis, and conclude with a bulleted summary of your key findings. Ensure comprehensive coverage.
            """).strip()

        elif focus_area == "vulnerable_groups":
            return textwrap.dedent(f"""
                **Your Task:** You are an equity analyst. Your primary goal is to analyze how the uploaded 'User Document' discusses or potentially impacts **vulnerable groups**. You should use the user's query and the COEQWAL Equity Framework to guide your analysis.

                {base_intro}

                **Instructions for Vulnerable Group Analysis:**
                1.  **Identify Vulnerable Groups:** You might look for any groups in the User Document that could be negatively affected or appear to have special needs (e.g., based on income, race, location, disability, language, etc.). Elaborate on any groups identified.
                2.  **Recognition (Recognitional Equity):** Consider if the document seems to acknowledge these groups and their unique situations, or if they might be overlooked. Provide comprehensive detail.
                3.  **Fair Process (Procedural Equity):** Does the document appear to describe a fair process for these groups to participate in decisions or potentially receive help? Detail the mechanisms or lack thereof.
                4.  **Fair Outcomes (Distributional Equity):** Could the document suggest whether these groups might receive a fair share of benefits and appear protected from harm? Provide a thorough assessment.
                5.  **Addressing Root Causes (Structural Equity):** Does the document seem to address any long-standing barriers or systems that might disadvantage these groups? Elaborate on the structural aspects.
                6.  **Suggest Evidence:** Where possible, you may refer to specific examples or quotes from the *User Document* that could support your observations.
                7.  **Present a Balanced View:** Discuss both the potential strengths (positive considerations) and possible weaknesses (potential concerns) that you might discern in the User Document comprehensively.
                8.  **Handle Missing Information:** If the User Document appears to lack detail on this topic, it may be appropriate to state this clearly.

                **Final Output:** Provide a **thorough and comprehensive** analysis focused on vulnerable groups, supported by specific examples from the document. Start with a clear overview and ensure each point is well-developed.
                """).strip()

        elif focus_area == "severity_of_impact":
            return textwrap.dedent(f"""
                **Your Task:** You are an equity analyst. Your primary goal is to assess the **potential severity of impacts**—both positive and negative—that might be described or implied in the uploaded 'User Document'. You should use the user's query and the COEQWAL Equity Framework to guide your analysis of how these potential impacts are handled.

                {base_intro}

                **Instructions for Severity of Impact Analysis:**
                1.  **Identify Key Impacts:** You might try to identify the main potential consequences or outcomes (both positive and negative) that could be suggested by the actions or policies described in the User Document. Elaborate on these impacts.
                2.  **Assess Severity:** For each identified impact, consider how serious it might be. This could involve evaluating the potential number of people affected, the possible duration of the impact, and its apparent reversibility. Provide detailed insights.
                3.  **Distribution of Severe Impacts (Distributional Equity):** Does the document seem to indicate if the most severe potential negative impacts might be unfairly concentrated on certain groups? Detail any disproportionate effects.
                4.  **Acknowledgement of Severity (Recognitional Equity):** Could the document suggest whether it acknowledges that some groups might be potentially impacted more severely than others? Provide comprehensive detail on this recognition.
                5.  **Process for Addressing Severe Impacts (Procedural Equity):** Does the document appear to describe a fair process for evaluating and possibly dealing with severe impacts? Elaborate on these processes.
                6.  **Structural Link to Severity (Structural Equity):** Do the severe potential impacts seem to stem from deeper, systemic issues? Discuss any apparent structural connections.
                7.  **Suggest Evidence:** Where possible, you may refer to specific examples or data from the *User Document* that could support your observations.
                8.  **Present a Balanced View:** Discuss both significant potential positive outcomes and possible severe negative impacts comprehensively.
                9.  **Handle Missing Information:** If the User Document appears to lack detail on the severity of impacts, it may be appropriate to state this clearly.
                
                **Final Output:** Provide a **thorough and comprehensive** analysis focused on the potential severity of impact, supported by specific examples from the document. Start with a clear overview and ensure each point is well-developed.
                """).strip()

        elif focus_area == "mitigation_strategies":
            return textwrap.dedent(f"""
                **Your Task:** You are an equity analyst. Your primary goal is to evaluate the **mitigation strategies** (plans to reduce harm) that might be discussed or implied in the uploaded 'User Document'. You should use the user's query and the COEQWAL Equity Framework to assess how fair and potentially effective these strategies appear to be.

                {base_intro}

                **Instructions for Mitigation Strategy Analysis:**
                1.  **Identify Mitigation Strategies:** You might look for any specific plans or actions in the User Document that appear to be designed to prevent, reduce, or address potential negative impacts. Provide a detailed list and description.
                2.  **Evaluate Strategy Fairness and Effectiveness:**
                    *   **Recognition:** Do the strategies seem to consider the unique needs of the people who might be most affected? Elaborate on this consideration.
                    *   **Fair Process:** Was the process for considering or creating these strategies seemingly fair and inclusive? Detail the procedural aspects.
                    *   **Fair Outcomes:** Could the strategies actually help those who might need it most, or do they appear to introduce new potential challenges? Provide a thorough assessment of outcomes.
                    *   **Addressing Root Causes:** Do the strategies seem to address the underlying problem, or could they be perceived as more of a temporary measure? Elaborate on their systemic impact.
                3.  **Consider Unintended Consequences:** Does the User Document hint at potential new problems that the strategies themselves might inadvertently create? Discuss these potential issues.
                4.  **Assess Sufficiency:** Do the strategies appear to be sufficiently robust to address the problem they are meant to target? Provide a comprehensive assessment.
                5.  **Suggest Evidence:** Where possible, you may refer to specific details from the *User Document* that could support your evaluation.
                6.  **Present a Balanced View:** Discuss both the potential strengths and possible weaknesses of the suggested mitigation strategies comprehensively.
                7.  **Handle Missing Information:** If the User Document appears to lack detail on mitigation strategies, it may be appropriate to state this clearly.

                **Final Output:** Provide a **thorough and comprehensive** analysis focused on mitigation strategies, supported by specific examples from the document. Start with a clear overview and ensure each point is well-developed.
            """).strip()

        else: # Default to the general COEQWAL analysis prompt
            return textwrap.dedent(f"""
                **Your Task:** You are an equity analyst. Your goal is to provide a balanced analysis of the uploaded 'User Document' based on the user's query, using the COEQWAL Equity Framework (Recognition, Procedure, Distribution, and Structure) as your guide. You might identify both potential strengths and possible areas of concern.

                {base_intro}

                **Instructions for General Analysis:**
                1.  **Search the User Document:** You may identify sections of the document that appear relevant to the user's query.
                2.  **Apply COEQWAL Framework:** For each relevant part, it might be useful to evaluate it using the four dimensions: Recognition, Procedure, Distribution, and Structure. Provide comprehensive details for each.
                3.  **Identify Strengths and Potential Concerns:** Consider whether certain points could be interpreted as positive alignments with equity or as potential areas of concern. Elaborate thoroughly on these findings.
                4.  **Suggest Evidence:** Where possible, you may refer to instances or examples from the *User Document* that could support your observations.
                5.  **Address Information Gaps:** If the User Document appears to lack detail on a specific aspect, it may be appropriate to note this limitation.

                **Final Output:** Provide a balanced analysis of the User Document. Start with a clear overview, then offer the detailed analysis, and conclude with a bulleted summary of your key potential strengths and concerns. Ensure comprehensive coverage for all aspects.
            """).strip()

    def decode_hex_utf16le(hex_string: str) -> str:
        """
        Try to decode hex-encoded UTF-16LE text snippets to readable string.
        Returns original string on failure.
        """
        try:
            # Clean whitespace/newlines, if any
            hex_str_clean = ''.join(hex_string.split())
            byte_data = bytes.fromhex(hex_str_clean)
            return byte_data.decode('utf-16le')
        except Exception:
            return hex_string


    def answer_question(
        self,
        session_id: str,
        query: str,
        focus_area: str = "general",
        custom_instructions: Optional[str] = None
    ) -> Tuple[str, List[Dict[str, Any]], List[str]]:
        """
        Answers a query using openai.responses.create with the 'include' parameter
        to reliably get source citations and raw search results.

        Returns:
            - final answer string
            - list of local DB source chunks (dictionaries, raw)
            - list of OpenAI source strings (file search results, raw formatted by OpenAI)
        """
        if not query:
            return "Please provide a query.", [], []

        logger.info(f"Answering query for session {session_id} with focus: {focus_area}")

        session_data = self.user_sessions.get(session_id)
        user_vector_store_id = None
        original_filename = "N/A"

        if session_data:
            original_filename = session_data.get("original_filename", "N/A")
            if session_data.get("status") == "completed":
                user_vector_store_id = session_data.get("vector_store_id")
            elif session_data.get("status") == "failed":
                return "Error: Document processing failed for this session. Cannot answer.", [], []
            else: # "in_progress" or other unexpected status
                 logger.warning(f"Session {session_id} document status is '{session_data.get('status')}'. May not be fully ready.")


        local_chunks = self.local_db.search(query, top_k=self.config.TOP_K_LOCAL) if (self.local_db and self.local_db.model) else []
        local_context_str = self._format_local_context_for_prompt(local_chunks)
        prompt_content_string = self._get_system_prompt(
            focus_area,
            original_filename,
            local_context_str,
            query,
            custom_instructions
        )

        try:
            tools = []
            if user_vector_store_id: # Only add tool if a vector store is available and ready
                tools.append({
                    "type": "file_search",
                    "vector_store_ids": [user_vector_store_id],
                    "max_num_results": getattr(self.config, "MAX_NUM_RESULTS", 5),
                })
            else:
                logger.warning(f"No valid vector store ID for session {session_id}. OpenAI file search will not be used.")


            kwargs = {
                "model": self.config.RESPONSES_MODEL,
                "input": prompt_content_string,
                #"temperature": self.config.TEMPERATURE,
                "max_output_tokens": self.config.MAX_OUTPUT_TOKENS,
            }

            if tools:
                kwargs["tools"] = tools
                kwargs["include"] = ["file_search_call.results"]

            logger.info(f"Session {session_id}: Calling client.responses.create with include=['file_search_call.results']")
            response = self.openai_interaction.client.responses.create(**kwargs)

            # Log full raw response for debugging
            try:
                resp_dict = response.model_dump()
            except Exception:
                resp_dict = response if isinstance(response, dict) else response.__dict__
            logger.info(f"Full OpenAI response dump:\n{json.dumps(resp_dict, indent=2)}")

            final_answer: Optional[str] = None
            
            # --- Parse message annotations for the answer text itself ---
            # This is specifically for extracting the main textual answer.
            # Sources extracted from file_search_call.results are handled separately below.
            for item in response.output:
                if getattr(item, "type", None) == "message" and hasattr(item, "content"):
                    for content_item in item.content:
                        if getattr(content_item, "type", None) == "output_text":
                            if final_answer is None:
                                final_answer = getattr(content_item, "text", "").strip() or "Model returned empty answer."
                            # Removed openai_sources.append(f"Reference from file...") from here
                            # as user wants sources exclusively from file_search_call.results for JSON output.
                            # If you still want file citations for other purposes (e.g. debugging, a separate field),
                            # they could be collected here.

            # --- Parse retrieved chunks from file_search_call.results (as per user's explicit instruction) ---
            retrieved_chunks_from_openai_tool: List[str] = []
            for item in response.output:
                if getattr(item, "type", None) == "file_search_call":
                    results = getattr(item, "results", None)
                    if results:
                        for res in results:
                            file_name = getattr(res, "file_name", None) or getattr(res, "filename", None) or "Unknown file"
                            chunk_text = getattr(res, "text", "")
                            snippet = chunk_text[:400] + "..." if len(chunk_text) > 400 else chunk_text
                            # Use the exact format requested by the user for OpenAI sources
                            retrieved_chunks_from_openai_tool.append(
                                f"Source from {file_name}:\n<blockquote>{snippet}</blockquote>"
                            )

            if final_answer is None:
                final_answer = "No valid answer returned by the model."

            # Return original local_chunks (raw dicts) - these will be *discarded* by generate_batch_analysis.py
            # But the signature needs to match rag_system's original return.
            return final_answer, local_chunks, retrieved_chunks_from_openai_tool

        except APIError as e:
            logger.error(f"Session {session_id}: APIError: {e}", exc_info=False)
            return f"Error: OpenAI API failed ({getattr(e, 'status_code', 'N/A')}).", [], []
        except Exception as e:
            logger.error(f"Session {session_id}: Unexpected error: {e}", exc_info=True)
            return "Error: An unexpected issue occurred while generating the response.", [], []

    def _format_local_context_for_prompt(self, local_results: List[Dict[str, Any]]) -> str:
        # This method is used to format local context for the *prompt* passed to OpenAI.
        # It's distinct from how sources are ultimately stored in the final JSON.
        if not local_results: return ""
        context_parts = []
        for i, result in enumerate(local_results):
            text = result.get("text", "").strip()
            metadata = result.get("metadata", {})
            headings = metadata.get("headings", [])
            score = result.get("score")
            header_parts = [f"Local Source {i+1}/{len(local_results)}"]
            if score is not None: header_parts.append(f"Score: {score:.4f}")
            if headings: header_parts.append(f"Section: '{headings[-1]}'")
            else: header_parts.append("Section: N/A")
            pos_index = metadata.get("position_index", -1); pos_total = metadata.get("position_total", -1)
            if pos_index != -1 and pos_total != -1: header_parts.append(f"Position: {pos_index+1}/{pos_total}")
            header = f"-- {' | '.join(header_parts)} --"
            context_parts.append(f"{header}\n{text}")
        return "\n\n".join(context_parts)

    def remove_user_session_resources(self, session_id: str, delete_openai_resources: bool = True):
        logger.info(f"Cleanup for session '{session_id}'. Delete OpenAI: {delete_openai_resources}")
        if session_id in self.user_sessions:
            doc_meta = self.user_sessions[session_id]
            vs_id = doc_meta.get("vector_store_id")
            file_id = doc_meta.get("file_id")

            
            if delete_openai_resources:
                if vs_id:
                    deleted_vs = self.openai_interaction.delete_vector_store(vs_id)
                    if not deleted_vs: logger.warning(f"Session {session_id}: Failed to delete VS {vs_id}.")
                if file_id:
                    time.sleep(1) # Small delay before file deletion
                    deleted_file = self.openai_interaction.delete_file(file_id)
                    if not deleted_file: logger.warning(f"Session {session_id}: Failed to delete File {file_id}.")
            
            del self.user_sessions[session_id] # This deletes the session from tracking
            logger.info(f"Removed session '{session_id}' from tracking.")
            return True
        else:
            logger.warning(f"Session ID '{session_id}' not found for cleanup.")
            return False