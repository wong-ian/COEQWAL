# core/config.py
import os
from pydantic_settings import BaseSettings
from typing import Optional, ClassVar
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("chatbot_config")

class Settings(BaseSettings):
    """Loads configuration from environment variables or .env file."""
    OPENAI_API_KEY: str

    # --- Local DB Settings ---
    LOCAL_EMBEDDING_MODEL: str = "all-mpnet-base-v2"
    LOCAL_DB_PATH: str = "db_v9.json" # Relative to project root

    # --- OpenAI Model Settings ---
    # Model for the specialized responses.create endpoint
    RESPONSES_MODEL: str = "gpt-4o"
    # Fallback chat model (if needed elsewhere, not used by default in responses.create)
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"

    # --- Retrieval Settings ---
    TOP_K_LOCAL: int = 8 # Number of chunks from local COEQWAL DB

    # --- OpenAI Vector Store/File Settings ---
    POLLING_INTERVAL_SECONDS: int = 5
    PROCESSING_TIMEOUT_SECONDS: int = 360 # 6 minutes timeout

    # --- Generation Settings (for responses.create) ---
    TEMPERATURE: float = 0.0
    MAX_OUTPUT_TOKENS: int = 1500 # Max tokens for LLM response generation
    MAX_NUM_RESULTS: int = 10 # Max results for file_search tool

    # Class configuration for Pydantic Settings
    class Config:
        env_file = '.env' # Load from .env file
        env_file_encoding = 'utf-8'
        extra = 'ignore' # Ignore extra fields not defined in the model
        
try:
    settings = Settings()
    # Check if API key was loaded
    if not settings.OPENAI_API_KEY or "YOUR_OPENAI_API_KEY_HERE" in settings.OPENAI_API_KEY:
        logger.error("OpenAI API Key is missing in .env file.")
    # You might want to raise an exception here or handle it appropriately
    # raise ValueError("OpenAI API Key not configured correctly.")
    else:
        logger.info("Configuration loaded successfully.")
    # Optionally mask part of the key for logging
    # logger.info(f"OpenAI API Key loaded (starts with: {settings.OPENAI_API_KEY[:5]}...).")
except Exception as e:
    logger.error(f"CRITICAL: Failed to load settings: {e}", exc_info=True)
    # Handle the failure, maybe exit or provide default settings if applicable
    settings = None # Indicate failure