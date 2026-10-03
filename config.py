import os

from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "chatbi_mvp"),
    "connect_timeout": float(os.getenv("DB_CONNECT_TIMEOUT", "5")),
    "read_timeout": float(os.getenv("DB_READ_TIMEOUT", "8")),
    "write_timeout": float(os.getenv("DB_WRITE_TIMEOUT", "8")),
    "charset": os.getenv("DB_CHARSET", "utf8mb4"),
}

LLM_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "base_url": os.getenv(
        "OPENAI_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    "model": os.getenv("LLM_MODEL", "qwen-max"),
    "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "4000")),
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.2")),
}

LLM_ANALYZER_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "base_url": os.getenv(
        "OPENAI_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    "model": os.getenv("LLM_ANALYZER_MODEL", "qwen-max"),
    "max_tokens": int(os.getenv("LLM_ANALYZER_MAX_TOKENS", "2000")),
    "temperature": float(os.getenv("LLM_ANALYZER_TEMPERATURE", "0.1")),
}