"""
Ollama Client - Local LLM integration for document Q&A.
Uses a small, free model running locally via Ollama.
"""

import ollama
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Default model — Llama 3.2 (3B) is fast and capable for laptops
DEFAULT_MODEL = "llama3.2"

# Ultra-simple prompt for tiny models
SYSTEM_PROMPT = "You are a document assistant. Read the provided text and answer the question directly using ONLY that text. If not in text, say 'Not found'. Keep it short."


class OllamaAgent:
    """
    Local LLM agent using Ollama for document-aware conversations.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.conversation_history: list[dict] = []
        self._available = None
        # Explicitly set the host to avoid localhost resolution issues
        self.client = ollama.Client(host='http://127.0.0.1:11434')

    def check_availability(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            models_response = self.client.list()
            # Handle different response formats from various Ollama versions
            if hasattr(models_response, 'models'):
                model_names = [m.model for m in models_response.models]
            elif isinstance(models_response, list):
                model_names = [m.get('name') for m in models_response]
            else:
                model_names = []
            
            self._available = True
            
            # Check for exact match or name:tag match
            has_model = any(self.model in name for name in model_names)
            
            if not has_model:
                logger.warning(
                    f"⚠️ Model '{self.model}' not found in local Ollama. "
                    f"Available: {model_names}. "
                )
            return True
        except Exception as e:
            logger.error(f"❌ Ollama not reachable at 127.0.0.1:11434: {e}")
            self._available = False
            return False

    def chat(
        self,
        user_message: str,
        document_context: str = "",
        stream: bool = False,
    ) -> str:
        """
        Send a message to the LLM with optional document context.
        """
        if not self.check_availability():
            return (
                "⚠️ Ollama is not reachable at http://127.0.0.1:11434. Please ensure it is running.\n"
                "Run: `ollama serve` in a terminal."
            )

        if document_context:
            augmented_message = (
                f"Context:\n{document_context}\n\n"
                f"Question: {user_message}\n"
                f"Answer:"
            )
        else:
            augmented_message = f"Tell the user no document is available. Question: {user_message}"

        # Stateless grounded chat
        messages = [
            {"role": "system", "content": "You are a direct data extractor. Use bold bullet points. If context says 'NOT FOUND', answer 'Not found'."},
            {"role": "user",   "content": augmented_message},
        ]

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
            )
            assistant_message = response["message"]["content"]
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            return assistant_message
        except Exception as e:
            return f"⚠️ Ollama Error: {str(e)}"

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    def set_model(self, model: str):
        """Switch the LLM model."""
        self.model = model
        self._available = None  # Force re-check
        self.clear_history()
        logger.info(f"🔄 Model switched to: {model}")
