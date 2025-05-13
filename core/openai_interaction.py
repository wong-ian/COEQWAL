# core/openai_interaction.py
import os
import time
import logging
from typing import Optional, List
from openai import OpenAI, APIError, APIStatusError, RateLimitError, NotFoundError

from .config import settings # Import settings

logger = logging.getLogger("openai_interaction")

class OpenAIInteraction:
    """Handles interactions with OpenAI API: File Upload, Vector Stores, Status Checks."""
    def __init__(self, api_key: Optional[str] = None):
        resolved_key = api_key or settings.OPENAI_API_KEY
        if not resolved_key or "YOUR_OPENAI_API_KEY_HERE" in resolved_key:
            raise ValueError("OpenAI API key is required but not found or is a placeholder.")

        try:
            self.client = OpenAI(api_key=resolved_key)
            # Test connection by listing models (optional, remove if causes issues)
            # REMOVED: self.client.models.list(limit=1) # <--- This line caused the TypeError
            # If the client initializes without error, we assume basic connectivity.
            # A more robust check might involve another simple, parameterless call if needed,
            # but often initialization success is sufficient.
            logger.info("OpenAI client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
            # Changed the exception type to ValueError for consistency, but ConnectionError is also reasonable
            raise ValueError("Could not initialize OpenAI client.") from e

    def upload_file(self, file_path: str, purpose: str = "assistants") -> Optional[str]:
        """Uploads a file to OpenAI."""
        if not os.path.exists(file_path):
            logger.error(f"File not found for upload: {file_path}")
            return None

        logger.info(f"Uploading file: {file_path} with purpose: {purpose}")
        try:
            with open(file_path, "rb") as f:
                response = self.client.files.create(file=f, purpose=purpose)
            logger.info(f"File '{os.path.basename(file_path)}' uploaded. File ID: {response.id}")
            return response.id
        except (APIError, APIStatusError) as e:
            logger.error(f"OpenAI API error uploading file {file_path}: Status={getattr(e, 'status_code', 'N/A')} Message={getattr(e, 'message', str(e))}")
        except RateLimitError:
             logger.error(f"OpenAI Rate Limit Error during file upload for {file_path}.")
        except Exception as e:
            logger.error(f"Unexpected error uploading file {file_path}: {e}", exc_info=True)
        return None

    def create_vector_store_with_files(self, name: str, file_ids: List[str]) -> Optional[str]:
        """Creates an OpenAI Vector Store and attaches files."""
        if not file_ids:
            logger.error("Cannot create vector store: No file IDs provided.")
            return None

        logger.info(f"Creating OpenAI Vector Store '{name}' with files: {file_ids}")
        try:
            vector_store = self.client.vector_stores.create(name=name, file_ids=file_ids)
            logger.info(f"Created OpenAI Vector Store '{name}'. ID: {vector_store.id}, Status: {vector_store.status}")
            # Note: File processing starts automatically. Need to wait separately.
            return vector_store.id
        except (APIError, APIStatusError) as e:
            logger.error(f"API error creating vector store '{name}': Status={getattr(e, 'status_code', 'N/A')} Message={getattr(e, 'message', str(e))}")
        except RateLimitError:
             logger.error(f"OpenAI Rate Limit Error during vector store creation for '{name}'.")
        except Exception as e:
            logger.error(f"Unexpected error creating vector store '{name}': {e}", exc_info=True)
        return None

    def wait_for_vector_store_file_processing(self, vector_store_id: str, file_id: str,
                                            timeout: int = settings.PROCESSING_TIMEOUT_SECONDS,
                                            poll_interval: int = settings.POLLING_INTERVAL_SECONDS) -> bool:
        """Polls the status of a specific file within a vector store until processed or timeout."""
        start_time = time.time()
        logger.info(f"Waiting up to {timeout}s for File ID {file_id} processing within VS {vector_store_id}...")
        last_status = None
        while time.time() - start_time < timeout:
            try:
                # Use client.vector_stores.files.retrieve with the *original* file ID
                vs_file = self.client.vector_stores.files.retrieve(
                    vector_store_id=vector_store_id,
                    file_id=file_id
                )
                status = vs_file.status
                if status != last_status:
                    logger.info(f"File {file_id} in VS {vector_store_id} status: {status}")
                    last_status = status

                if status == 'completed':
                    logger.info(f"File {file_id} processing completed successfully in VS {vector_store_id}.")
                    return True
                elif status == 'failed':
                    error_message = vs_file.last_error.message if vs_file.last_error else "Unknown error"
                    logger.error(f"File {file_id} processing failed in VS {vector_store_id}. Error: {error_message}")
                    return False
                elif status in ['in_progress']:
                    pass # Continue polling
                elif status in ['cancelled', 'cancelling']:
                    logger.warning(f"File {file_id} processing cancelled in VS {vector_store_id}.")
                    return False
                else:
                    logger.warning(f"Unexpected VS File status '{status}' for file {file_id}. Continuing poll.")

            except NotFoundError:
                 # This is important: it means the file IS NOT associated with the VS *yet* or bad IDs.
                 # Let's keep polling for a short while in case it's a delay.
                 logger.warning(f"VS File {file_id} in VS {vector_store_id} not found (404). May not be linked yet or IDs incorrect. Retrying...")
                 # Optionally add a small delay specific to 404 before the main poll_interval
                 time.sleep(poll_interval / 2) # Example: wait half the interval
            except APIStatusError as e:
                 # Handle other API errors (like 500s)
                 logger.error(f"API error checking VS file status (File ID {file_id}, VS {vector_store_id}): {e}")
                 # Maybe retry once on 5xx errors? For now, treat as failure.
                 return False
            except RateLimitError:
                 logger.warning(f"Rate limit hit while checking VS file status for {file_id}. Retrying after delay...")
                 time.sleep(poll_interval * 2) # Longer delay for rate limits
            except Exception as e:
                logger.error(f"Unexpected error checking VS file status (File ID {file_id}, VS {vector_store_id}): {e}", exc_info=True)
                return False # Stop waiting on unexpected errors

            time.sleep(poll_interval)

        logger.error(f"Timeout ({timeout}s) waiting for file {file_id} processing in VS {vector_store_id}. Last status: {last_status}")
        return False

    def delete_vector_store(self, vector_store_id: str) -> bool:
        """Deletes an OpenAI Vector Store."""
        logger.info(f"Attempting to delete OpenAI Vector Store: {vector_store_id}")
        try:
            response = self.client.vector_stores.delete(vector_store_id=vector_store_id)
            if response.deleted:
                logger.info(f"Vector Store {vector_store_id} deleted successfully.")
            else:
                 logger.warning(f"Vector Store {vector_store_id} deletion response indicates not deleted: {response}")
            return response.deleted
        except NotFoundError:
             logger.warning(f"Vector Store {vector_store_id} not found (already deleted?).")
             return True # Treat as success if already gone
        except (APIError, APIStatusError) as e:
             logger.error(f"API error deleting Vector Store {vector_store_id}: Status={getattr(e, 'status_code', 'N/A')} Message={getattr(e, 'message', str(e))}")
             return False
        except RateLimitError:
             logger.error(f"Rate limit error deleting Vector Store {vector_store_id}.")
             return False
        except Exception as e:
            logger.error(f"Unexpected error deleting Vector Store {vector_store_id}: {e}", exc_info=True)
            return False

    def delete_file(self, file_id: str) -> bool:
        """Deletes an OpenAI File."""
        logger.info(f"Attempting to delete OpenAI File: {file_id}")
        try:
            response = self.client.files.delete(file_id=file_id)
            if response.deleted:
                 logger.info(f"File {file_id} deleted successfully.")
            else:
                 logger.warning(f"File {file_id} deletion response indicates not deleted: {response}")
            return response.deleted
        except NotFoundError:
             logger.warning(f"File {file_id} not found (already deleted?).")
             return True # Treat as success
        except (APIError, APIStatusError) as e:
            # Specifically check for 409 Conflict which can happen if file is attached to VS
            if getattr(e, 'status_code', None) == 409:
                 logger.warning(f"Cannot delete File {file_id} (Conflict/409). It might still be attached to a Vector Store.")
            else:
                 logger.error(f"API error deleting File {file_id}: Status={getattr(e, 'status_code', 'N/A')} Message={getattr(e, 'message', str(e))}")
            return False
        except RateLimitError:
             logger.error(f"Rate limit error deleting File {file_id}.")
             return False
        except Exception as e:
            logger.error(f"Unexpected error deleting File {file_id}: {e}", exc_info=True)
            return False