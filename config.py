import os

from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "1234"),
    "database": os.getenv("DB_NAME", "chatbi_mvp"),
    "connect_timeout": float(os.getenv("DB_CONNECT_TIMEOUT", "5")),
    "read_timeout": float(os.getenv("DB_READ_TIMEOUT", "8")),
    "write_timeout": float(os.getenv("DB_WRITE_TIMEOUT", "8")),
    "charset": os.getenv("DB_CHARSET", "utf8mb4"),
}

_build_default_db_source = lambda: DB_CONFIG.copy()

APP_CONFIG = {
    "environment": os.getenv("APP_ENV", "dev"),
    "database": {
        "default_source": os.getenv("DB_DEFAULT_SOURCE", "mysql_main"),
        "sources": {
            os.getenv("DB_DEFAULT_SOURCE", "mysql_main"): _build_default_db_source(),
        },
        "runtime": {
            "pool_size": int(os.getenv("DB_POOL_SIZE", 5)),
            "max_overflow": int(os.getenv("DB_POOL_MAX_OVERFLOW", 5)),
            "pool_timeout": float(os.getenv("DB_POOL_TIMEOUT", 3)),
            "slow_query_threshold_ms": float(os.getenv("DB_SLOW_QUERY_THRESHOLD_MS", 200)),
        },
    },
    "llm": {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "model": os.getenv("LLM_MODEL", "gpt-4"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        "temperature": 0.1,
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", 4000)),
    },
    "features": {
        "few_shot": os.getenv("FEATURE_FEW_SHOT", "true").lower() == "true",
        "rules": os.getenv("FEATURE_RULES", "true").lower() == "true",
        "guards": os.getenv("FEATURE_GUARDS", "true").lower() == "true",
        "indicator_knowledge": os.getenv("FEATURE_INDICATOR_KNOWLEDGE", "true").lower() == "true",
        "schema_linking": os.getenv("FEATURE_SCHEMA_LINKING", "false").lower() == "true",
        "indicator_rag": os.getenv("FEATURE_INDICATOR_RAG", "false").lower() == "true",
    },
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
    "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-v3"),
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
    "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-v3"),
}