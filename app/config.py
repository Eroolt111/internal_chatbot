import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for the chatbot application"""
    
    # ─── Database Configuration ────────────────────────────────────────────────
    DB_HOST: str = os.getenv("DB_HOST", "actual_db_host")
    DB_PORT: str = os.getenv("DB_PORT", "actual_db_port") 
    DB_NAME: str = os.getenv("DB_NAME", "actual_db_name")
    DB_USER: str = os.getenv("DB_USER", "actual_db_user")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD","actual_db_password")
    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+psycopg2://{self.DB_USER}:" 
            f"{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )
    
    # ─── LLM BACKEND SELECTION ────────────────────────────────────────────────    
    # "ollama" or "openai"
    LLM_BACKEND: str = os.getenv("LLM_BACKEND", "ollama").lower()
    
    # ─── Ollama Configuration ───────────────────────────────────────────────
    OLLAMA_LLM_MODEL: str = os.getenv("OLLAMA_LLM_MODEL", "gemma3:12b")
    OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large:latest")
    OLLAMA_REQUEST_TIMEOUT: float = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "600.0"))
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # ─── OpenAI Configuration ────────────────────────────────────────────────
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    #OPENAI_COMPLETION_MODEL: str = os.getenv("OPENAI_COMPLETION_MODEL", "gpt-3.5-turbo")
    OPENAI_COMPLETION_MODEL: str = os.getenv("OPENAI_COMPLETION_MODEL", "gpt-4.1")
    OPENAI_EMBED_MODEL: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    OPENAI_REQUEST_TIMEOUT: float = float(os.getenv("OPENAI_REQUEST_TIMEOUT", "300.0"))
    
    # ─── Application Configuration ───────────────────────────────────────────
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # ─── Storage Configuration ───────────────────────────────────────────────
    TABLE_INFO_DIR: str = os.getenv("TABLE_INFO_DIR", "PostgreSQL_TableInfo")
    TABLE_INDEX_DIR: str = os.getenv("TABLE_INDEX_DIR", "table_index_dir")
    
    # ─── API (FastAPI) Configuration ────────────────────────────────────────
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    # ─── Query Pipeline Limits ──────────────────────────────────────────────
    MAX_TABLE_RETRIEVAL: int = int(os.getenv("MAX_TABLE_RETRIEVAL", "3"))
    MAX_ROW_RETRIEVAL: int = int(os.getenv("MAX_ROW_RETRIEVAL", "2"))
    MAX_ROWS_PER_TABLE: int = int(os.getenv("MAX_ROWS_PER_TABLE", "500"))
    
    def __init__(self):
        Path(self.TABLE_INFO_DIR).mkdir(exist_ok=True)
        Path(self.TABLE_INDEX_DIR).mkdir(exist_ok=True)
    
    def validate(self) -> bool:
        """Validate critical configuration values"""
        if not self.DB_PASSWORD:
            raise ValueError("DB_PASSWORD must be set in environment variables")
        if self.LLM_BACKEND not in ("ollama", "openai"):
            raise ValueError("LLM_BACKEND must be either 'ollama' or 'openai'")
        if self.LLM_BACKEND == "openai" and not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set when LLM_BACKEND=openai")
        return True

config = Config()
