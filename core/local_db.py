# core/local_db.py
import os
import json
import logging
import numpy as np
import time
from typing import List, Dict, Any, Optional

try:
    from sentence_transformers import SentenceTransformer
    st_imported = True
except ImportError:
    logging.error("SentenceTransformers library not found. Install it: pip install sentence-transformers")
    SentenceTransformer = None
    st_imported = False

from .config import settings # Import from config.py

logger = logging.getLogger("local_db")

# Only define VectorDatabase if SentenceTransformer was imported successfully
if SentenceTransformer:
    class VectorDatabase:
        """Simple in-memory vector database using SentenceTransformers (for local DB)."""
        def __init__(self, embedding_model_name: Optional[str]):
            self.documents: List[Dict[str, Any]] = []
            self.embedding_model_name = embedding_model_name
            self.model = None
            if embedding_model_name and st_imported:
                try:
                    # Explicitly specify cache folder if needed, otherwise uses default
                    # cache_dir = os.path.join(os.getcwd(), '.cache', 'sentence_transformers')
                    # os.makedirs(cache_dir, exist_ok=True)
                    self.model = SentenceTransformer(embedding_model_name) #, cache_folder=cache_dir)
                    logger.info(f"Initialized local embedding model: {embedding_model_name}")
                except Exception as e:
                    logger.error(f"Failed to load local embedding model '{embedding_model_name}': {e}", exc_info=True)
            elif not st_imported and embedding_model_name:
                logger.error("Cannot initialize VectorDatabase model: sentence-transformers not installed.")
            else:
                logger.warning("No local embedding model specified for VectorDatabase.")

        def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
            """Adds processed chunks with their embeddings to the database."""
            # (Code is the same as in the notebook, generally not used in the final app flow)
            if not chunks: return
            if not self.model: logger.error("Cannot add documents: Local embedding model not initialized."); return
            logger.info(f"Embedding {len(chunks)} chunks locally...")
            texts = [c["text"] for c in chunks]
            try:
                embeddings = self.model.encode(texts, show_progress_bar=False)
                if len(embeddings) != len(chunks): logger.error(f"Local embedding count mismatch."); return
                for i, chunk in enumerate(chunks):
                    meta = chunk.get("metadata", {})
                    doc = { "id": chunk.get("id", -1), "text": chunk.get("text", ""), "embedding": embeddings[i].tolist(),
                            "metadata": { "headings": meta.get("headings", []), "position_index": meta.get("position_index", -1), "position_total": meta.get("position_total", -1) } }
                    self.documents.append(doc)
                logger.info(f"Added {len(chunks)} chunks to local Vector DB.")
            except Exception as e: logger.error(f"Error during local embedding or adding documents: {e}", exc_info=True)


        def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
            """Performs cosine similarity search on locally stored embeddings."""
            # (Code largely the same as the notebook)
            if not self.documents or top_k <= 0:
                logger.warning("Local DB search skipped: No documents loaded or top_k <= 0.")
                return []
            if not self.model:
                logger.error("Cannot search: Local embedding model not initialized.")
                return []
            if not query:
                logger.warning("Local DB search skipped: Empty query.")
                return []

            try:
                start_time = time.time()
                query_emb = np.array(self.model.encode(query))

                valid_docs_info = []
                for i, d in enumerate(self.documents):
                    embedding = d.get("embedding")
                    if isinstance(embedding, list) and len(embedding) > 0:
                        valid_docs_info.append((i, embedding))
                    else:
                        logger.debug(f"Document ID {d.get('id', i)} skipped in local search due to missing/invalid embedding.")

                if not valid_docs_info:
                    logger.warning("No documents with valid embeddings found in local DB for search.")
                    return []

                original_indices, doc_embeddings_list = zip(*valid_docs_info)
                doc_embeddings = np.array(doc_embeddings_list)

                if query_emb.ndim != 1 or doc_embeddings.ndim != 2 or query_emb.shape[0] != doc_embeddings.shape[1]:
                    logger.error(f"Local DB embedding dimension mismatch. Query: {query_emb.shape}, Docs: {doc_embeddings.shape}")
                    return []

                norms_docs = np.linalg.norm(doc_embeddings, axis=1)
                norm_query = np.linalg.norm(query_emb)
                valid_mask = (norms_docs > 1e-9) & (norm_query > 1e-9)
                similarities = np.zeros(len(doc_embeddings))

                if np.any(valid_mask):
                    dot_products = np.dot(doc_embeddings[valid_mask], query_emb)
                    similarities[valid_mask] = dot_products / (norms_docs[valid_mask] * norm_query)

                finite_sim_mask = np.isfinite(similarities)
                valid_indices_for_sorting = np.where(finite_sim_mask)[0]

                if len(valid_indices_for_sorting) == 0:
                    logger.info("No valid similarity scores found after calculation.")
                    return []

                actual_k = min(top_k, len(valid_indices_for_sorting))
                sorted_valid_indices = valid_indices_for_sorting[np.argsort(similarities[valid_indices_for_sorting])[::-1]]
                top_k_filtered_indices = sorted_valid_indices[:actual_k]

                top_k_original_indices = [original_indices[i] for i in top_k_filtered_indices]

                results = []
                for i, original_idx in enumerate(top_k_original_indices):
                    doc_copy = self.documents[original_idx].copy()
                    score_idx = top_k_filtered_indices[i]
                    doc_copy["score"] = float(similarities[score_idx])
                    # Optionally remove embedding
                    # doc_copy.pop("embedding", None)
                    results.append(doc_copy)

                end_time = time.time()
                logger.info(f"Local DB search completed in {end_time - start_time:.2f} seconds. Found {len(results)} results.")
                return results
            except Exception as e:
                logger.error(f"Error during local vector search: {e}", exc_info=True)
                return []

        @classmethod
        def load(cls, file_path: str, embedding_model_name: Optional[str]) -> 'VectorDatabase':
            """Loads the local database from a JSON file."""
            # (Code largely the same as notebook)
            logger.info(f"Loading local Vector DB from {file_path}")
            db = cls(embedding_model_name) # Initialize first
            if not os.path.exists(file_path):
                logger.error(f"Local Vector DB file not found: {file_path}")
                raise FileNotFoundError(f"Local Vector DB file not found: {file_path}")

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                if not isinstance(loaded_data, list):
                    raise ValueError("Invalid JSON format for DB: expected a list.")
                if loaded_data and not isinstance(loaded_data[0], dict):
                    raise ValueError("Invalid JSON format for DB: list items are not dictionaries.")

                valid_docs = []
                for i, doc in enumerate(loaded_data):
                    if not isinstance(doc.get("embedding"), list):
                        logger.warning(f"Doc ID {doc.get('id', i)} missing/invalid embedding. May be excluded.")
                    if "text" not in doc or "id" not in doc:
                        logger.warning(f"Doc at index {i} (ID: {doc.get('id', 'N/A')}) missing 'text' or 'id'.")
                    valid_docs.append(doc)

                db.documents = valid_docs
                logger.info(f"Loaded {len(db.documents)} documents into local Vector DB.")

            except FileNotFoundError: raise
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from {file_path}: {e}")
                raise ValueError(f"Invalid JSON file: {file_path}") from e
            except Exception as e:
                logger.error(f"Error loading local DB from {file_path}: {e}", exc_info=True)
                raise

            if not db.model and embedding_model_name:
                logger.error(f"Local DB loaded, but embedding model '{embedding_model_name}' failed initialization. Search will fail.")
            elif not embedding_model_name:
                logger.warning("Local DB loaded, but no embedding model specified. Search will fail.")

            return db
else:
    logger.error("VectorDatabase class cannot be defined because SentenceTransformers is not installed.")
    VectorDatabase = None

# --- Global instance ---
# We'll load the DB once when the module is imported (or FastAPI starts)
loaded_local_db: Optional[VectorDatabase] = None

def get_local_db() -> Optional[VectorDatabase]:
    """Dependency function to get the loaded local DB instance."""
    global loaded_local_db
    if loaded_local_db is None and settings:
         logger.warning("Attempting to load local DB because it wasn't loaded at startup.")
         try:
             # Ensure the path is absolute or relative to the correct root
             db_file_path = os.path.abspath(settings.LOCAL_DB_PATH)
             if os.path.exists(db_file_path):
                 loaded_local_db = VectorDatabase.load(db_file_path, settings.LOCAL_EMBEDDING_MODEL)
             else:
                 logger.error(f"Local DB file not found at calculated path: {db_file_path}")
         except Exception as e:
             logger.error(f"Failed to load local DB: {e}", exc_info=True)
    return loaded_local_db

def load_db_on_startup():
    """Function to explicitly load the DB, called from main.py startup event."""
    global loaded_local_db
    if settings and settings.LOCAL_DB_PATH and VectorDatabase:
        try:
            # Ensure the path is absolute or relative to the correct root
            db_file_path = os.path.abspath(settings.LOCAL_DB_PATH)
            if os.path.exists(db_file_path):
                logger.info(f"Loading local database from: {db_file_path}")
                loaded_local_db = VectorDatabase.load(db_file_path, settings.LOCAL_EMBEDDING_MODEL)
            else:
                logger.error(f"Local DB file specified but not found at path: {db_file_path}. Local context unavailable.")
        except Exception as e:
            logger.error(f"Failed to load local vector database on startup: {e}", exc_info=True)
            loaded_local_db = None # Ensure it's None on failure
    elif not VectorDatabase:
         logger.error("Cannot load local DB: SentenceTransformers library not installed.")
    else:
         logger.warning("Local DB path not configured. Proceeding without local context.")