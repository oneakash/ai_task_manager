import os
from dotenv import load_dotenv

load_dotenv()

# LLM providers
def get_groq_key() -> str:
    return os.getenv("GROQ_API_KEY", "").strip()

def get_openai_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()

def get_groq_model() -> str:
    return os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

def get_openai_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Storage
DATABASE_URL = "sqlite:///./tasks.db"
CHROMA_PATH = "./chroma_db"
CHROMA_COLLECTION = "tasks_collection"

# Logging
LOG_FILE = "taskflow_agent.log"