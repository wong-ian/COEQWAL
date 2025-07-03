# core/rag_system.py
import os
import textwrap
import logging
import time
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

    def _get_system_prompt(self, focus_area: str, original_filename: str, local_context_str: str, query: str) -> str:
        """
        Selects or generates the appropriate system prompt based on the focus area.
        """
        base_intro = f"""
            **Framework Definitions (from COEQWAL Context):**
            --- START CONTEXT FROM COEQWAL DOCUMENT ---
            {local_context_str if local_context_str else 'No definitions or context from the COEQWAL document were retrieved. Focus analysis solely on the User Document if possible, or state inability to apply the framework critically.'}
            --- END CONTEXT FROM COEQWAL DOCUMENT ---

            **User Document Name:** {original_filename}
            **User Query:** {query}
        """

        if focus_area == "vulnerable_groups":
            return textwrap.dedent(f"""
                **Your Task:** You are a specialized equity analyst. Your primary focus is to identify and evaluate how the uploaded 'User Document' specifically addresses or impacts **vulnerable groups** in relation to the user's query. Apply the COEQWAL Equity Framework (Recognitional, Procedural, Distributional, Structural) as a lens, but critically examine every aspect *through its effect on vulnerable groups*.

                {base_intro}

                **Specific Instructions for Vulnerable Group Analysis:**
                1.  **Identify Vulnerable Groups:** From the User Document and the query, identify any explicitly mentioned or implicitly affected vulnerable groups (e.g., based on socioeconomic status, race, ethnicity, age, disability, language, geographic location, historical marginalization etc.).
                2.  **Recognitional Equity:** How are these groups recognized, or ignored? Are their unique needs, histories, and perspectives acknowledged in the User Document?
                3.  **Procedural Equity:** Does the User Document describe fair and accessible processes for these groups to participate in decisions, raise concerns, or access benefits/mitigation?
                4.  **Distributional Equity:** How are resources, benefits, burdens, and risks distributed with respect to these vulnerable groups according to the User Document? Are there disproportionate negative impacts or unequal access to positive outcomes?
                5.  **Structural Equity:** Does the User Document address or perpetuate systemic/structural barriers that disadvantage these vulnerable groups?
                6.  **Severity of Impact:** Specifically consider the severity of potential or actual impacts on these groups as detailed in the User Document.
                7.  **Mitigation Strategies:** If mitigation strategies are discussed in the User Document, how effectively and equitably do they address the needs and reduce negative impacts on these vulnerable groups?
                8.  **Provide Evidence:** Cite specific examples or text snippets from the *User Document* (using file search citations if provided by the tool).
                9.  **Balanced View (within Vulnerable Group Focus):** Present both positive considerations for vulnerable groups and areas of concern found in the User Document.
                10. **Handle Missing Information:** If the User Document lacks sufficient detail for this focused analysis, state that clearly.

                **Output:** Provide a detailed analysis focused *solely* on vulnerable groups as they relate to the User Query, the User Document, and the COEQWAL framework.
            """).strip()

        elif focus_area == "severity_of_impact":
            return textwrap.dedent(f"""
                **Your Task:** You are a specialized equity analyst. Your primary focus is to assess the **severity of impacts** (both positive and negative) described or implied in the 'User Document' as they relate to the user's query, using the COEQWAL Equity Framework as your guide.

                {base_intro}

                **Specific Instructions for Severity of Impact Analysis:**
                1.  **Identify Impacts:** From the User Document and the query, identify key impacts, outcomes, or consequences resulting from actions, policies, or situations described.
                2.  **Assess Severity:** For each impact, evaluate its potential or actual severity. Consider factors like magnitude, scope (how many are affected), duration, and irreversibility.
                3.  **Distributional Equity & Severity:** How is the severity of negative impacts distributed across different groups (especially vulnerable ones, if identifiable in the User Document)? Are benefits distributed in a way that alleviates severe pre-existing inequities?
                4.  **Recognitional Equity & Severity:** Does the User Document acknowledge the varying severities of impact on different groups? Are particularly severe impacts on marginalized communities recognized and prioritized?
                5.  **Procedural Equity & Severity:** Does the User Document describe processes for assessing or addressing severe impacts? Are these processes fair and inclusive of those most severely affected?
                6.  **Structural Equity & Severity:** Do the identified severe impacts stem from or contribute to underlying structural inequities? Does the User Document propose structural changes to mitigate severe negative impacts or prevent future ones?
                7.  **Provide Evidence:** Cite specific examples or data from the *User Document*.
                8.  **Balanced View:** Discuss both severe negative impacts and significant positive impacts, analyzing them through the COEQWAL lens.
                9.  **Handle Missing Information:** If the User Document lacks detail on the severity or distribution of impacts, state that.

                **Output:** Provide a detailed analysis focused on the severity of impacts discussed in the User Document, as related to the User Query and the COEQWAL framework.
            """).strip()

        elif focus_area == "mitigation_strategies":
            return textwrap.dedent(f"""
                **Your Task:** You are a specialized equity analyst. Your primary focus is to evaluate any **mitigation strategies** proposed or discussed in the 'User Document' in response to potential negative impacts or inequities, as they relate to the user's query. Use the COEQWAL Equity Framework.

                {base_intro}

                **Specific Instructions for Mitigation Strategy Analysis:**
                1.  **Identify Mitigation Strategies:** From the User Document and the query, pinpoint any specific strategies, actions, or plans intended to prevent, reduce, or compensate for negative impacts or inequities.
                2.  **Effectiveness & Equity of Strategies:**
                    *   **Recognitional:** Do the mitigation strategies acknowledge and address the specific needs and perspectives of different groups, especially those most impacted or vulnerable?
                    *   **Procedural:** Were/are the processes for developing and implementing these mitigation strategies fair, transparent, and inclusive? Do affected communities have a voice?
                    *   **Distributional:** How are the benefits and burdens of the mitigation strategies themselves distributed? Do they effectively target those most in need or those most harmed? Do they create new inequities?
                    *   **Structural:** Do the mitigation strategies address only symptoms, or do they aim to tackle root causes and structural barriers contributing to the original inequity or negative impact?
                3.  **Unintended Consequences:** Does the User Document consider potential unintended negative consequences of the mitigation strategies, particularly for vulnerable groups?
                4.  **Sufficiency:** Are the proposed mitigation strategies likely to be sufficient to address the scale and nature of the identified impacts/inequities?
                5.  **Provide Evidence:** Cite specific details about the mitigation strategies from the *User Document*.
                6.  **Balanced View:** Discuss the strengths and weaknesses of the proposed mitigation strategies in terms of their potential equity outcomes.
                7.  **Handle Missing Information:** If the User Document lacks detail on mitigation strategies or their equity implications, state that.

                **Output:** Provide a detailed analysis focused on the mitigation strategies within the User Document, as related to the User Query and the COEQWAL framework.
            """).strip()

        # Default to the general COEQWAL analysis prompt
        return textwrap.dedent(f"""
            **Your Task:** You are a critical analytical assistant helping a user analyze their uploaded document by applying the principles and definitions from the 'COEQWAL Equity Framework'. The goal is to conduct a **balanced and critical analysis**, identifying specific examples of **both positive alignment (equity advancements/pros) and potential concerns, shortcomings, or misalignments (equity issues/cons)** within the User Document related to the user's query, based on the framework's dimensions.

            {base_intro}

            **Instructions for Critical Analysis (General):**
            1.  **Understand the Framework:** Carefully review the definitions and concepts related to the four equity dimensions (Recognitional, Procedural, Distributional, Structural) provided in the COEQWAL context.
            2.  **Search the User Document:** Actively search the *User Document* (using the provided file search tool, if available) for specific details, decisions, statements, or actions related to the **subject of the User Query**.
            3.  **Apply the Framework Critically:** For each relevant detail found in the User Document pertaining to the **User Query's subject**:
                *   Evaluate it against the definitions of the COEQWAL equity dimensions.
                *   Determine if it primarily exemplifies **positive alignment** (a strength/pro) **OR if it raises potential concerns, indicates a shortcoming, or represents potential inequity** (a weakness/con).
                *   Clearly explain *why*, referencing framework definitions.
            4.  **Provide Evidence:** Cite specific examples from the *User Document*.
            5.  **Synthesize Findings into a Balanced View:** Structure your response clearly.
            6.  **Acknowledge Sources:** State if info is from COEQWAL Context or User Document.
            7.  **Handle Missing Information:** If the User Document lacks detail, or if file search failed, state that.

            **Output:** Provide a **balanced and critical analysis** of the elements found in the User Document relevant to the User Query, using the COEQWAL equity dimensions.
        """).strip()

    def answer_question(self, session_id: str, query: str, focus_area: str = "general") -> Tuple[str, List[Dict[str, Any]]]: # Added focus_area
        """
        Answers a query for a given session using hybrid RAG.
        Selects system prompt based on focus_area.
        """
        if not query:
            return "Please provide a query.", []

        logger.info(f"Answering query for session {session_id} with focus: {focus_area}")

        session_data = self.user_sessions.get(session_id)
        user_vector_store_id = None
        original_filename = "N/A"

        if session_data:
            original_filename = session_data.get("original_filename", "N/A")
            if session_data.get("status") == "completed":
                user_vector_store_id = session_data.get("vector_store_id")
                if user_vector_store_id:
                    logger.info(f"Session {session_id}: Using VS ID {user_vector_store_id} for file '{original_filename}'.")
                else:
                    logger.error(f"Session {session_id}: VS ID missing for '{original_filename}'!")
            elif session_data.get("status") == "failed":
                 logger.warning(f"Session {session_id}: Doc '{original_filename}' failed processing. File search may not work.")
            else:
                 logger.warning(f"Session {session_id}: Doc '{original_filename}' has status: {session_data.get('status')}. File search may fail.")
        else:
            logger.warning(f"Session {session_id}: No user document info. Proceeding without file search.")

        local_chunks = []
        if self.local_db and self.local_db.model:
            logger.info(f"Session {session_id}: Searching local DB for query: '{query[:100]}...'")
            local_chunks = self.local_db.search(query, top_k=self.config.TOP_K_LOCAL)
            logger.info(f"Session {session_id}: Retrieved {len(local_chunks)} local chunks.")
        # ... (logging for no local_db or model remains the same)

        local_context_str = self._format_local_context_for_prompt(local_chunks)

        # Get the appropriate system prompt
        prompt_content_string = self._get_system_prompt(
            focus_area=focus_area,
            original_filename=original_filename,
            local_context_str=local_context_str,
            query=query
        )

        try:
            tools = []
            if user_vector_store_id:
                tools.append({
                    "type": "file_search",
                    "vector_store_ids": [user_vector_store_id],
                    "max_num_results": self.config.MAX_NUM_RESULTS
                })
                logger.info(f"Session {session_id}: Adding file_search tool with VS ID: {user_vector_store_id}")
            else:
                 logger.info(f"Session {session_id}: No file_search tool added (VS ID issue).")

            kwargs = {
                "model": self.config.RESPONSES_MODEL,
                "input": prompt_content_string,
                "temperature": self.config.TEMPERATURE,
                "max_output_tokens": self.config.MAX_OUTPUT_TOKENS,
            }
            if tools:
                kwargs["tools"] = tools

            logger.info(f"Session {session_id}: Calling client.responses.create with model '{kwargs['model']}', focus '{focus_area}', tools: {bool(tools)}")
            start_time = time.time()

            if not hasattr(self.openai_interaction.client, 'responses'):
                 logger.error("FATAL: OpenAI client object does not have 'responses' attribute.")
                 return "Error: Backend configuration issue - OpenAI API method unavailable.", local_chunks

            response = self.openai_interaction.client.responses.create(**kwargs)
            end_time = time.time()
            logger.info(f"Session {session_id}: OpenAI responses.create call completed in {end_time - start_time:.2f} seconds.")

            # ... (keep existing response parsing logic)
            final_answer = None
            if hasattr(response, "output") and response.output:
                for item in response.output:
                    if getattr(item, "type", None) == "message" and hasattr(item, "content"):
                        for content_item in item.content:
                            if getattr(content_item, "type", None) == "output_text":
                                text_content = getattr(content_item, "text", "")
                                annotations = getattr(content_item, "annotations", [])
                                final_answer = text_content.strip()
                                if annotations: logger.info(f"Session {session_id}: Received {len(annotations)} annotations.")
                                break
                        if final_answer is not None: break
                if final_answer is None:
                     logger.warning(f"Session {session_id}: Parsed response.output but no 'output_text'.")
                     final_answer = "Error: Could not parse text content from model's response."
            elif hasattr(response, "error") and response.error:
                 logger.error(f"Session {session_id}: OpenAI responses.create returned an error: {response.error}")
                 final_answer = f"Error: Model returned an error - {response.error.get('message', 'Unknown error')}"
            else:
                logger.warning(f"Session {session_id}: Unexpected response structure: {response}")
                final_answer = "Error: Received an unexpected or empty response."
            return final_answer or "Model returned an empty or unparseable response.", local_chunks

        # ... (keep existing except blocks for APIError, RateLimitError, TypeError, Exception)
        except APIError as e:
            logger.error(f"Session {session_id}: APIError: {e}", exc_info=False)
            return f"Error: OpenAI API failed ({getattr(e, 'status_code', 'N/A')}).", local_chunks
        except RateLimitError:
             logger.error(f"Session {session_id}: RateLimitError.", exc_info=False)
             return "Error: API rate limit exceeded.", local_chunks
        except TypeError as te:
            logger.error(f"Session {session_id}: TypeError: {te}", exc_info=True)
            return f"Error: Backend API argument mismatch - {te}", local_chunks
        except Exception as e:
            logger.error(f"Session {session_id}: Unexpected error: {e}", exc_info=True)
            return "Error: An unexpected issue occurred.", local_chunks


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