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
        # ... (keep existing __init__)
        self.config = settings
        self.openai_interaction = openai_interaction
        self.local_db = get_local_db()
        if self.local_db is None: logger.warning("HybridRAGSystem initialized WITHOUT a functional local database.")
        else: logger.info("HybridRAGSystem initialized WITH local database.")
        self.user_sessions: Dict[str, Dict[str, Any]] = {}

    def add_user_document_for_session(self, session_id: str, file_path: str, original_filename: str) -> Tuple[bool, str]:
        # ... (keep existing add_user_document_for_session)
        logger.info(f"Processing user document for session '{session_id}': '{original_filename}' from path '{file_path}'")
        self.remove_user_session_resources(session_id, delete_openai_resources=True)
        file_id = self.openai_interaction.upload_file(file_path, purpose="assistants")
        if not file_id:
            msg = f"Failed to upload file {original_filename} for session {session_id}."
            logger.error(msg)
            try:
                if os.path.exists(file_path): os.remove(file_path)
            except OSError as e: logger.error(f"Error removing temporary file {file_path}: {e}")
            return False, msg
        vs_name = f"vs_{session_id}_{original_filename}".replace(" ", "_")[:100]
        vector_store_id = self.openai_interaction.create_vector_store_with_files(name=vs_name, file_ids=[file_id])
        if not vector_store_id:
            msg = f"Failed to create Vector Store for file ID {file_id} (session {session_id}). Cleaning up."
            logger.error(msg)
            self.openai_interaction.delete_file(file_id)
            try:
                if os.path.exists(file_path): os.remove(file_path)
            except OSError as e: logger.error(f"Error removing temporary file {file_path}: {e}")
            return False, msg
        processing_success = self.openai_interaction.wait_for_vector_store_file_processing(vector_store_id=vector_store_id, file_id=file_id)
        doc_status = "completed" if processing_success else "failed"
        self.user_sessions[session_id] = {"file_id": file_id, "vector_store_id": vector_store_id, "original_filename": original_filename, "status": doc_status}
        try:
            if os.path.exists(file_path): os.remove(file_path); logger.info(f"Removed temporary file: {file_path}")
        except OSError as e: logger.error(f"Error removing temporary file {file_path}: {e}")
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
        """
        base_intro = f"""
            **Framework Definitions (from COEQWAL Context):**
            --- START CONTEXT FROM COEQWAL DOCUMENT ---
            {local_context_str if local_context_str else 'No definitions or context from the COEQWAL document were retrieved.'}
            --- END CONTEXT FROM COEQWAL DOCUMENT ---

            **User Document Name:** {original_filename}
            **User Query:** {query}

            **IMPORTANT - Output Style:** Please write your analysis in clear, accessible language. Avoid overly academic or technical jargon. Conclude your response with a bulleted summary of the key findings.
        """
        
        if focus_area == "custom" and custom_instructions:
            # This prompt is already custom, so a hard word limit might be less useful.
            # We will leave it as is, but you could add a word count here too if desired.
            return textwrap.dedent(f"""
                **Your Task:** You are an equity analyst. Your goal is to analyze the uploaded 'User Document' based on a specific set of custom instructions provided by the user. You must strictly follow these instructions while using the COEQWAL Equity Framework as a guiding lens.

                {base_intro}

                **User's Custom Focus Instructions:**
                --- START OF USER INSTRUCTIONS ---
                {custom_instructions}
                --- END OF USER INSTRUCTIONS ---

                **Your Analysis Steps:**
                1.  Thoroughly understand the User's Custom Focus Instructions.
                2.  Search the User Document for all parts relevant to these instructions and the user's query.
                3.  Apply the COEQWAL dimensions (Recognition, Procedure, Distribution, Structure) where they help illuminate the analysis as per the user's instructions.
                4.  Provide a balanced view, discussing both strengths and weaknesses you find.
                5.  Cite evidence from the User Document to support your points.
                6.  If the document lacks the necessary information to follow the instructions, state this limitation clearly.

                **Final Output:** Provide a detailed analysis that directly addresses the User's Custom Focus Instructions. Start with a clear overview, then provide the detailed analysis, and end with a bulleted summary of your key findings.
            """).strip()

        elif focus_area == "vulnerable_groups":
            return textwrap.dedent(f"""
                **Your Task:** You are an equity analyst. Your primary goal is to analyze how the uploaded 'User Document' discusses or impacts **vulnerable groups**. Use the user's query and the COEQWAL Equity Framework to guide your analysis.

                {base_intro}

                **Instructions for Vulnerable Group Analysis:**
                1.  **Identify Vulnerable Groups:** Look for any groups in the User Document that might be negatively affected or have special needs (e.g., based on income, race, location, disability, language, etc.).
                2.  **Recognition (Recognitional Equity):** Does the document acknowledge these groups and their unique situations? Or are they ignored?
                3.  **Fair Process (Procedural Equity):** Does the document describe a fair process for these groups to participate in decisions or receive help?
                4.  **Fair Outcomes (Distributional Equity):** Does the document show if these groups receive a fair share of benefits and are protected from harm?
                5.  **Addressing Root Causes (Structural Equity):** Does the document address any long-standing barriers or systems that disadvantage these groups?
                6.  **Provide Evidence:** Back up your points with specific examples or quotes from the *User Document*.
                7.  **Present a Balanced View:** Discuss both the strengths (positive considerations) and weaknesses (potential concerns) found in the User Document.
                8.  **Handle Missing Information:** If the User Document lacks detail on this topic, clearly state that.

                **Final Output:** Provide a **thorough and comprehensive** analysis focused on vulnerable groups, supported by specific evidence from the document. Start with a clear overview...
                """).strip()

        elif focus_area == "severity_of_impact":
            return textwrap.dedent(f"""
                **Your Task:** You are an equity analyst. Your primary goal is to assess the **severity of impacts**—both positive and negative—described in the uploaded 'User Document'. Use the user's query and the COEQWAL Equity Framework to analyze how these impacts are handled.

                {base_intro}

                **Instructions for Severity of Impact Analysis:**
                1.  **Identify Key Impacts:** Find the main consequences or outcomes (both good and bad) resulting from the actions or policies described in the User Document.
                2.  **Assess Severity:** For each impact, evaluate how serious it is. Consider how many people are affected, how long the impact lasts, and if it's reversible.
                3.  **Distribution of Severe Impacts (Distributional Equity):** Are the most severe negative impacts unfairly concentrated on certain groups?
                4.  **Acknowledgement of Severity (Recognitional Equity):** Does the document acknowledge that some groups might be impacted more severely than others?
                5.  **Process for Addressing Severe Impacts (Procedural Equity):** Does the document describe a fair process for evaluating and dealing with severe impacts?
                6.  **Structural Link to Severity (Structural Equity):** Do the severe impacts come from deeper, systemic problems?
                7.  **Provide Evidence:** Back up your points with specific examples or data from the *User Document*.
                8.  **Present a Balanced View:** Discuss both significant positive outcomes and severe negative impacts.
                9.  **Handle Missing Information:** If the User Document lacks detail on the severity of impacts, clearly state that.
                
                **Final Output:** Provide a **thorough and comprehensive** analysis focused on the severity of impact, supported by specific evidence from the document. Start with a clear overview...
                """).strip()

        elif focus_area == "mitigation_strategies":
            return textwrap.dedent(f"""
                **Your Task:** You are an equity analyst. Your primary goal is to evaluate the **mitigation strategies** (plans to reduce harm) discussed in the uploaded 'User Document'. Use the user's query and the COEQWAL Equity Framework to assess how fair and effective these strategies are.

                {base_intro}

                **Instructions for Mitigation Strategy Analysis:**
                1.  **Identify Mitigation Strategies:** Find any specific plans or actions in the User Document designed to prevent, reduce, or fix negative impacts.
                2.  **Evaluate Strategy Fairness and Effectiveness:**
                    *   **Recognition:** Do the strategies consider the unique needs of the people most affected?
                    *   **Fair Process:** Was the process for creating these strategies fair and inclusive?
                    *   **Fair Outcomes:** Do the strategies actually help those who need it most, or do they create new problems?
                    *   **Addressing Root Causes:** Do the strategies fix the underlying problem, or are they just a temporary "band-aid"?
                3.  **Consider Unintended Consequences:** Does the User Document mention potential new problems the strategies themselves could create?
                4.  **Assess Sufficiency:** Are the strategies strong enough to solve the problem they are meant to address?
                5.  **Provide Evidence:** Back up your evaluation with specific details from the *User Document*.
                6.  **Present a Balanced View:** Discuss both the strengths and weaknesses of the proposed mitigation strategies.
                7.  **Handle Missing Information:** If the User Document lacks detail on mitigation strategies, clearly state that.

                **Final Output:** Provide a **thorough and comprehensive** analysis focused on vulnerable groups, supported by specific evidence from the document. Start with a clear overview...
            """).strip()

        else: # Default to the general COEQWAL analysis prompt
            return textwrap.dedent(f"""
                **Your Task:** You are an equity analyst. Your goal is to provide a balanced analysis of the uploaded 'User Document' based on the user's query, using the COEQWAL Equity Framework as your guide. Identify both strengths and weaknesses.

                {base_intro}

                **Instructions for General Analysis:**
                1.  **Search the User Document:** Find parts of the document relevant to the user's query.
                2.  **Apply COEQWAL Framework:** For each relevant part, evaluate it using the four dimensions: Recognition, Procedure, Distribution, and Structure.
                3.  **Identify Strengths and Weaknesses:** Clearly label points as positive alignments with equity (pros) or as potential concerns (cons).
                4.  **Provide Evidence:** Back up your points with examples from the *User Document*.
                5.  **Handle Missing Information:** If the User Document lacks detail, clearly state that.

                **Final Output:** Provide a balanced analysis of the User Document. Start with a clear overview, then provide the detailed analysis, and end with a bulleted summary of the key pros and cons.
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
            - list of local DB source chunks (dictionaries)
            - list of OpenAI source strings (file citations + snippet quotes)
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
            if user_vector_store_id:
                tools.append({
                    "type": "file_search",
                    "vector_store_ids": [user_vector_store_id],
                    "max_num_results": getattr(self.config, "MAX_NUM_RESULTS", 5),
                })

            kwargs = {
                "model": self.config.RESPONSES_MODEL,
                "input": prompt_content_string,
                "temperature": self.config.TEMPERATURE,
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
            openai_sources: List[str] = []
            found_search_results = False

            # Extract model answer and file citations from message output (annotations)
            if hasattr(response, "output") and response.output:
                for item in response.output:
                    if getattr(item, "type", None) == "message" and hasattr(item, "content"):
                        for content_item in item.content:
                            if getattr(content_item, "type", None) == "output_text":
                                if final_answer is None:
                                    final_answer = getattr(content_item, "text", "").strip() or "Model returned empty answer."
                                annotations = getattr(content_item, "annotations", [])
                                for ann in annotations:
                                    if hasattr(ann, "type") and ann.type == "file_citation":
                                        file_name = getattr(ann, "filename", "Unknown file")
                                        quote_idx = getattr(ann, "index", None)
                                        openai_sources.append(f"Reference from file '{file_name}', index: {quote_idx}")

                # Extract raw search results from the file_search_call output
                for item in response.output:
                    if getattr(item, "type", None) == "file_search_call":
                        results = getattr(item, "results", None)
                        if results:
                            found_search_results = True
                            for res in results:
                                # Sometimes field might be `filename` or `file_name`
                                file_name = getattr(res, "file_name", None) or getattr(res, "filename", None) or "Unknown file"

                                # Each result can have multiple 'content' chunks with 'text'
                                for part in getattr(res, "content", []):
                                    raw_text = getattr(part, "text", "")
                                    decoded_text = decode_hex_utf16le(raw_text)
                                    openai_sources.append(f"Source from {file_name}:\n<blockquote>{decoded_text}</blockquote>")

            if final_answer is None:
                final_answer = "No valid answer returned by the model."

            if tools and not found_search_results:
                logger.warning(f"Session {session_id}: 'include' data with search results was not found in the response.")

            return final_answer, local_chunks, openai_sources

        except APIError as e:
            logger.error(f"Session {session_id}: APIError: {e}", exc_info=False)
            return f"Error: OpenAI API failed ({getattr(e, 'status_code', 'N/A')}).", [], []
        except Exception as e:
            logger.error(f"Session {session_id}: Unexpected error: {e}", exc_info=True)
            return "Error: An unexpected issue occurred while generating the response.", [], []

    def _format_local_context_for_prompt(self, local_results: List[Dict[str, Any]]) -> str:
        # ... (keep existing _format_local_context_for_prompt)
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
        # ... (keep existing remove_user_session_resources)
        logger.info(f"Cleanup for session '{session_id}'. Delete OpenAI: {delete_openai_resources}")
        if session_id in self.user_sessions:
            doc_meta = self.user_sessions[session_id]
            vs_id = doc_meta.get("vector_store_id"); file_id = doc_meta.get("file_id")
            if delete_openai_resources:
                if vs_id:
                    deleted_vs = self.openai_interaction.delete_vector_store(vs_id)
                    if not deleted_vs: logger.warning(f"Session {session_id}: Failed to delete VS {vs_id}.")
                if file_id:
                    time.sleep(1) # Small delay before file deletion
                    deleted_file = self.openai_interaction.delete_file(file_id)
                    if not deleted_file: logger.warning(f"Session {session_id}: Failed to delete File {file_id}.")
            del self.user_sessions[session_id]
            logger.info(f"Removed session '{session_id}' from tracking.")
            return True
        else:
            logger.warning(f"Session ID '{session_id}' not found for cleanup.")
            return False