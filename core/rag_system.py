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
    """
    Orchestrates RAG using a local VectorDatabase (COEQWAL) and OpenAI
    Vector Stores (user docs) via the Responses API pattern.
    """
    def __init__(self, openai_interaction: OpenAIInteraction):
        self.config = settings
        self.openai_interaction = openai_interaction
        self.local_db = get_local_db() # Get the globally loaded instance

        if self.local_db is None:
            logger.warning("HybridRAGSystem initialized WITHOUT a functional local database.")
        else:
             logger.info("HybridRAGSystem initialized WITH local database.")

        # Stores metadata about processed user documents (maps session_id to OpenAI resource IDs)
        # In a real app, this would be Redis, a DB, etc. Using a simple dict for demo.
        self.user_sessions: Dict[str, Dict[str, Any]] = {}

    def add_user_document_for_session(self, session_id: str, file_path: str, original_filename: str) -> Tuple[bool, str]:
        """
        Uploads a user document for a specific session, creates a dedicated
        vector store, waits for processing, and stores IDs in session data.
        Cleans up old resources for the same session first.

        Returns: (success_status, message)
        """
        logger.info(f"Processing user document for session '{session_id}': '{original_filename}' from path '{file_path}'")

        # --- Cleanup old resources for this session ---
        self.remove_user_session_resources(session_id, delete_openai_resources=True)

        # 1. Upload the file to OpenAI
        file_id = self.openai_interaction.upload_file(file_path, purpose="assistants")
        if not file_id:
            msg = f"Failed to upload file {original_filename} for session {session_id}."
            logger.error(msg)
            # Clean up the temporary file if it exists
            try:
                if os.path.exists(file_path): os.remove(file_path)
            except OSError as e:
                logger.error(f"Error removing temporary file {file_path}: {e}")
            return False, msg

        # 2. Create a dedicated Vector Store for this file
        # Make name more unique, truncate if needed
        vs_name = f"vs_{session_id}_{original_filename}".replace(" ", "_")[:100] # Shorter, safer name
        vector_store_id = self.openai_interaction.create_vector_store_with_files(
            name=vs_name,
            file_ids=[file_id]
        )

        if not vector_store_id:
            msg = f"Failed to create Vector Store for file ID {file_id} (session {session_id}). Cleaning up uploaded file."
            logger.error(msg)
            self.openai_interaction.delete_file(file_id) # Cleanup file if VS creation failed
            try:
                if os.path.exists(file_path): os.remove(file_path)
            except OSError as e:
                logger.error(f"Error removing temporary file {file_path}: {e}")
            return False, msg

        # 3. Wait for the file to be processed within the Vector Store
        processing_success = self.openai_interaction.wait_for_vector_store_file_processing(
            vector_store_id=vector_store_id,
            file_id=file_id # Use the original file_id here
        )

        # 4. Store metadata in session
        doc_status = "completed" if processing_success else "failed"
        self.user_sessions[session_id] = {
            "file_id": file_id,
            "vector_store_id": vector_store_id,
            "original_filename": original_filename,
            "status": doc_status
        }

        # Clean up the temporary file after processing attempt
        try:
            if os.path.exists(file_path): os.remove(file_path)
            logger.info(f"Removed temporary file: {file_path}")
        except OSError as e:
            logger.error(f"Error removing temporary file {file_path}: {e}")


        if not processing_success:
            msg = f"File '{original_filename}' (ID: {file_id}) failed processing in Vector Store {vector_store_id} for session {session_id}. File search may fail."
            logger.error(msg)
            # Keep metadata with 'failed' status, user might try again
            return False, msg
        else:
            msg = f"User document '{original_filename}' processed successfully for session '{session_id}'. Ready for queries."
            logger.info(msg)
            return True, msg

    def answer_question(self, session_id: str, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Answers a query for a given session using hybrid RAG.
        Uses client.responses.create with the 'input' key.

        Returns: (answer_string, list_of_local_source_chunks)
        """
        if not query:
            return "Please provide a query.", []

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
                    logger.error(f"Session {session_id}: Metadata found but Vector Store ID missing for file '{original_filename}'!")
            elif session_data.get("status") == "failed":
                 logger.warning(f"Session {session_id}: User document '{original_filename}' failed processing earlier. File search may not work.")
            else: # Should not happen if add_user_document logic is correct
                 logger.warning(f"Session {session_id}: User document '{original_filename}' has unexpected status: {session_data.get('status')}. File search may fail.")
        else:
            logger.warning(f"Session {session_id}: No user document information found. Proceeding without file search.")


        # 1. Local RAG retrieval (COEQWAL DB)
        local_chunks = []
        if self.local_db and self.local_db.model: # Check if model is loaded too
            logger.info(f"Session {session_id}: Searching local DB for query: '{query[:100]}...'")
            local_chunks = self.local_db.search(query, top_k=self.config.TOP_K_LOCAL)
            logger.info(f"Session {session_id}: Retrieved {len(local_chunks)} chunks from local DB.")
        elif not self.local_db:
             logger.warning(f"Session {session_id}: Local DB not available, skipping local context retrieval.")
        else: # DB exists but model failed
             logger.error(f"Session {session_id}: Local DB exists but embedding model not loaded. Skipping local context retrieval.")

        # Format local context
        local_context_str = self._format_local_context_for_prompt(local_chunks)

        # 3. Build the GENERALIZED prompt string for analysis
        #    (Using the detailed prompt from the Notebook)
        prompt_content_string = textwrap.dedent(f"""
            **Your Task:** You are a critical analytical assistant helping a user analyze their uploaded document ('User Document: {original_filename}') by applying the principles and definitions from the 'COEQWAL Equity Framework' (provided below). The goal is to conduct a **balanced and critical analysis**, identifying specific examples of **both positive alignment (equity advancements/pros) and potential concerns, shortcomings, or misalignments (equity issues/cons)** within the User Document related to the user's query, based on the framework's dimensions.

            **Framework Definitions (from COEQWAL Context):**
            --- START CONTEXT FROM COEQWAL DOCUMENT ---
            {local_context_str if local_context_str else 'No definitions or context from the COEQWAL document were retrieved. Focus analysis solely on the User Document if possible, or state inability to apply the framework critically.'}
            --- END CONTEXT FROM COEQWAL DOCUMENT ---

            **User Query:** {query}

            **Instructions for Critical Analysis:**
            1.  **Understand the Framework:** Carefully review the definitions and concepts related to the four equity dimensions (Recognitional, Procedural, Distributional, Structural) provided in the COEQWAL context. Note the ideals each dimension strives for.
            2.  **Search the User Document:** Actively search the *User Document* (using the provided file search tool, if available) for specific details, decisions, statements, descriptions, or actions related to the **subject of the User Query**.
            3.  **Apply the Framework Critically:** For each relevant detail found in the User Document pertaining to the **User Query's subject**:
                *   Evaluate it against the definitions of the COEQWAL equity dimensions.
                *   Determine if the detail primarily exemplifies **positive alignment** with a dimension (a strength/pro) **OR if it raises potential concerns, indicates a shortcoming, or represents potential inequity** related to that dimension (a weakness/con).
                *   Clearly explain *why*, referencing the framework definitions. For concerns/cons, explain *how* the detail might fall short of the equity ideal defined in the framework.
            4.  **Provide Evidence:** Cite specific examples, text snippets, or findings from the *User Document* to support your analysis for *both* positive aspects and identified concerns/cons. Use file search citations if provided by the tool.
            5.  **Synthesize Findings into a Balanced View:** Structure your response clearly, addressing the user's query by presenting your critical analysis. Aim for a balanced perspective, discussing both strengths and weaknesses concerning equity as found in the User Document.
            6.  **Acknowledge Sources:** Explicitly state whether information comes from the COEQWAL Context (framework definitions) or the User Document (specific details being analyzed).
            7.  **Handle Missing Information:** If the User Document lacks sufficient detail to analyze critically against a specific dimension, or if the COEQWAL context was missing, or if file search failed, state that clearly. Acknowledge limitations in the analysis due to missing information. Do not invent pros or cons.

            **Output:** Provide a **balanced and critical analysis** of the elements found in the User Document relevant to the User Query. Explain how they align well *and* where they potentially fall short concerning the COEQWAL equity dimensions, backing claims with evidence from the User Document. If file search was used, include citations.
        """).strip()

        # 4. Call OpenAI API using client.responses.create
        try:
            tools = []
            # Only add file_search tool if we have a valid, completed VS ID
            if user_vector_store_id:
                tools.append({
                    "type": "file_search",
                    "vector_store_ids": [user_vector_store_id],
                    "max_num_results": self.config.MAX_NUM_RESULTS
                })
                logger.info(f"Session {session_id}: Adding file_search tool with VS ID: {user_vector_store_id}")
            else:
                 logger.info(f"Session {session_id}: No file_search tool added (missing or failed VS ID).")


            # Prepare arguments for responses.create
            kwargs = {
                "model": self.config.RESPONSES_MODEL,
                "input": prompt_content_string, # Use 'input' key
                "temperature": self.config.TEMPERATURE,
                # Use max_output_tokens (ensure this attribute exists in config)
                "max_output_tokens": self.config.MAX_OUTPUT_TOKENS,
            }
            if tools:
                kwargs["tools"] = tools


            logger.info(f"Session {session_id}: Calling client.responses.create with model '{kwargs['model']}' and tools: {bool(tools)}")
            start_time = time.time()

            # Ensure the client has the 'responses' attribute
            if not hasattr(self.openai_interaction.client, 'responses'):
                 logger.error("FATAL: OpenAI client object does not have 'responses' attribute.")
                 return "Error: Backend configuration issue - OpenAI API method unavailable.", local_chunks

            response = self.openai_interaction.client.responses.create(**kwargs)

            end_time = time.time()
            logger.info(f"Session {session_id}: OpenAI responses.create call completed in {end_time - start_time:.2f} seconds.")

            # Extract the response text (adapt based on actual response structure)
            # This parsing logic seems specific to the 'responses' structure seen before
            final_answer = None
            if hasattr(response, "output") and response.output:
                for item in response.output:
                    if getattr(item, "type", None) == "message" and hasattr(item, "content"):
                        for content_item in item.content:
                            if getattr(content_item, "type", None) == "output_text":
                                # Accumulate text and annotations if available
                                text_content = getattr(content_item, "text", "")
                                annotations = getattr(content_item, "annotations", [])
                                # Basic handling: just get the text for now
                                # In a real app, you'd process annotations to link citations
                                final_answer = text_content.strip()
                                if annotations:
                                     logger.info(f"Session {session_id}: Received {len(annotations)} annotations with the response.")
                                     # TODO: Process annotations if needed by frontend
                                break # Take the first output_text found
                        if final_answer is not None: break # Found answer in this message item
                if final_answer is None: # Check if loop finished without finding output_text
                     logger.warning(f"Session {session_id}: Parsed response.output but did not find 'output_text' content type.")
                     final_answer = "Error: Could not parse the specific text content from the model's response structure."

            elif hasattr(response, "error") and response.error:
                 logger.error(f"Session {session_id}: OpenAI responses.create returned an error: {response.error}")
                 final_answer = f"Error: Model returned an error - {response.error.get('message', 'Unknown error')}"
            else:
                logger.warning(f"Session {session_id}: Received unexpected response structure from responses.create: {response}")
                final_answer = "Error: Received an unexpected or empty response from the model."


            return final_answer or "Model returned an empty or unparseable response.", local_chunks

        except APIError as e:
            logger.error(f"Session {session_id}: OpenAI API Error during responses.create: Status={getattr(e, 'status_code', 'N/A')}, Message={getattr(e, 'message', str(e))}", exc_info=False)
            error_message = f"Error: OpenAI API failed ({getattr(e, 'status_code', 'N/A')})."
            # Add more specific messages based on error details if possible
            return error_message, local_chunks
        except RateLimitError:
             logger.error(f"Session {session_id}: OpenAI Rate Limit Error during responses.create.", exc_info=False)
             return "Error: API rate limit exceeded. Please try again later.", local_chunks
        except TypeError as te:
            # Catch argument mismatch errors specifically
            logger.error(f"Session {session_id}: TypeError during API call, likely incorrect arguments for responses.create: {te}", exc_info=True) # Log full trace for this
            return f"Error: Backend API argument mismatch - {te}", local_chunks
        except Exception as e:
            logger.error(f"Session {session_id}: Unexpected error generating response via responses.create: {e}", exc_info=True)
            return "Error: An unexpected issue occurred while generating the response.", local_chunks

    def _format_local_context_for_prompt(self, local_results: List[Dict[str, Any]]) -> str:
        """Formats local search results into a string for the LLM prompt."""
        if not local_results:
            return "" # Return empty string if no results

        context_parts = []
        for i, result in enumerate(local_results):
            text = result.get("text", "").strip()
            metadata = result.get("metadata", {})
            headings = metadata.get("headings", []) # Should be list of strings
            score = result.get("score")
            doc_id = result.get('id', 'N/A')
            pos_index = metadata.get("position_index", -1)
            pos_total = metadata.get("position_total", -1)

            header_parts = [f"Local Source {i+1}/{len(local_results)}"]
            if score is not None: header_parts.append(f"Score: {score:.4f}")
            if headings: header_parts.append(f"Section: '{headings[-1]}'")
            else: header_parts.append("Section: N/A")
            if pos_index != -1 and pos_total != -1: header_parts.append(f"Position: {pos_index+1}/{pos_total}")

            header = f"-- {' | '.join(header_parts)} --"
            context_parts.append(f"{header}\n{text}")

        return "\n\n".join(context_parts)

    def remove_user_session_resources(self, session_id: str, delete_openai_resources: bool = True):
        """Removes user document metadata and optionally OpenAI resources for a session."""
        logger.info(f"Attempting cleanup for session '{session_id}'. Delete OpenAI resources: {delete_openai_resources}")
        if session_id in self.user_sessions:
            doc_meta = self.user_sessions[session_id]
            vs_id = doc_meta.get("vector_store_id")
            file_id = doc_meta.get("file_id")
            original_filename = doc_meta.get("original_filename", "N/A")

            if delete_openai_resources:
                # Important: Delete the Vector Store *before* the File,
                # as deleting the file might fail if it's still linked.
                if vs_id:
                    logger.info(f"Session {session_id}: Deleting OpenAI Vector Store: {vs_id}")
                    deleted_vs = self.openai_interaction.delete_vector_store(vs_id)
                    if not deleted_vs:
                        logger.warning(f"Session {session_id}: Failed to delete Vector Store {vs_id} (or already deleted).")
                else:
                    logger.warning(f"Session {session_id}: No Vector Store ID found to delete for file '{original_filename}'.")

                # Now attempt to delete the file
                if file_id:
                    logger.info(f"Session {session_id}: Deleting OpenAI File: {file_id}")
                    # Allow some time for VS deletion to propagate if needed
                    time.sleep(1) # Small delay
                    deleted_file = self.openai_interaction.delete_file(file_id)
                    if not deleted_file:
                        logger.warning(f"Session {session_id}: Failed to delete File {file_id} (or already deleted/still linked?).")
                else:
                    logger.warning(f"Session {session_id}: No File ID found to delete for file '{original_filename}'.")
            else:
                logger.info(f"Session {session_id}: Skipping deletion of OpenAI resources for file '{original_filename}' as requested.")

            # Remove from our in-memory tracking
            del self.user_sessions[session_id]
            logger.info(f"Removed session '{session_id}' (file: '{original_filename}') from local tracking.")
            return True
        else:
            logger.warning(f"Session ID '{session_id}' not found in tracking, cannot remove resources.")
            return False