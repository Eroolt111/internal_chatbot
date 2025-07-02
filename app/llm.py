import logging
from llama_index.core.settings import Settings
from .config import config

logger = logging.getLogger(__name__)

class LLMManager:
    """Language Model Manager supporting Ollama or OpenAI based on config."""

    def __init__(self):
        self.llm = None
        self.embed_model = None
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize either Ollama or OpenAI clients per config."""
        try:
            if config.LLM_BACKEND == "openai":
                # ─── OpenAI Setup ───────────────────────────────────────────
                from llama_index.llms.openai import OpenAI
                from llama_index.embeddings.openai import OpenAIEmbedding

                logger.info("Initializing OpenAI LLM & Embedding")
                self.llm = OpenAI(
                    api_key=config.OPENAI_API_KEY,
                    model=config.OPENAI_COMPLETION_MODEL,
                    request_timeout=config.OPENAI_REQUEST_TIMEOUT,
                )
                self.embed_model = OpenAIEmbedding(
                    api_key=config.OPENAI_API_KEY,
                    model=config.OPENAI_EMBED_MODEL,
                    request_timeout=config.OPENAI_REQUEST_TIMEOUT,
                )

            else:
                # ─── Ollama Setup ────────────────────────────────────────────
                from llama_index.llms.ollama import Ollama
                from llama_index.embeddings.ollama import OllamaEmbedding

                logger.info("Initializing Ollama LLM & Embedding")
                self.llm = Ollama(
                    model=config.OLLAMA_LLM_MODEL,
                    request_timeout=config.OLLAMA_REQUEST_TIMEOUT,
                    base_url=config.OLLAMA_HOST,
                )
                self.embed_model = OllamaEmbedding(
                    model_name=config.OLLAMA_EMBED_MODEL,
                    base_url=config.OLLAMA_HOST,
                )

            # Register globally for llama_index
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model

            # Quick sanity check
            self._test_connection()

            logger.info("✅ LLM & Embedding initialized successfully")

        except Exception as e:
            logger.error(f"❌ Error initializing LLM backends: {e}")
            raise

    def _test_connection(self) -> None:
        """Simple sanity-check call to ensure LLM is responsive."""
        try:
            # both Ollama and OpenAI share `.complete()` signature
            resp = self.llm.complete("Hello, are you there?")
            logger.debug(f"LLM healthcheck response: {resp}")
        except Exception as e:
            logger.error(f"LLM healthcheck failed: {e}")
            raise

    def get_llm(self):
        """Return the raw LLM instance for llama_index pipelines."""
        return self.llm

    def get_embed_model(self):
        """Return the embedding model instance."""
        return self.embed_model

    def complete(self, prompt: str) -> str:
        """Convenience wrapper around the LLM’s complete() method."""
        try:
            return str(self.llm.complete(prompt))
        except Exception as e:
            logger.error(f"Error during LLM.complete(): {e}")
            raise

    def is_available(self) -> bool:
        """Return True if LLM is reachable and responsive."""
        try:
            self._test_connection()
            return True
        except:
            return False

llm_manager = LLMManager()
