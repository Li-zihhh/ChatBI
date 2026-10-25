"""统一配置管理。

所有配置均通过环境变量注入，无硬编码业务值。
分三层：
  1. DB_CONFIG      —— pymysql 连接参数（保持与 pymysql.connect(**kwargs) 兼容）
  2. LLM_CONFIG     —— 生成 SQL 的模型
  3. APP_CONFIG     —— 运行期策略：连接池、限流、安全、评测等
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: str) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


# ==================== 数据库连接参数 ====================
# 注意：read_timeout 只是 socket 读超时，不能中断服务端查询；
# 真正的语句超时由 APP_CONFIG["database"]["runtime"]["statement_timeout_ms"] 控制。
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": _env_int("DB_PORT", "3306"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "chatbi_mvp"),
    "connect_timeout": _env_float("DB_CONNECT_TIMEOUT", "5"),
    "read_timeout": _env_float("DB_READ_TIMEOUT", "8"),
    "write_timeout": _env_float("DB_WRITE_TIMEOUT", "8"),
    "charset": os.getenv("DB_CHARSET", "utf8mb4"),
    "autocommit": True,
}


def _build_default_db_source() -> dict:
    return DB_CONFIG.copy()


APP_CONFIG = {
    "environment": os.getenv("APP_ENV", "dev"),
    "database": {
        "default_source": os.getenv("DB_DEFAULT_SOURCE", "mysql_main"),
        "sources": {
            os.getenv("DB_DEFAULT_SOURCE", "mysql_main"): _build_default_db_source(),
        },
        "runtime": {
            # 连接池：超出 pool_size 后最多再开 max_overflow 个，之后按 pool_timeout 等待
            "pool_size": _env_int("DB_POOL_SIZE", "5"),
            "max_overflow": _env_int("DB_POOL_MAX_OVERFLOW", "5"),
            "pool_timeout": _env_float("DB_POOL_TIMEOUT", "3"),
            # 慢查询阈值（毫秒）：超过则记录 EXPLAIN，用于诊断而非阻塞响应
            "slow_query_threshold_ms": _env_float("DB_SLOW_QUERY_THRESHOLD_MS", "200"),
            "explain_on_slow": _env_bool("DB_EXPLAIN_ON_SLOW", "true"),
            # 结果集行数上限：超出即截断并置 truncated 标记，避免 OOM
            "max_rows": _env_int("DB_MAX_ROWS", "5000"),
            # 语句超时：通过 MySQL MAX_EXECUTION_TIME 由服务端中断，仅作用于 SELECT
            "statement_timeout_ms": _env_int("DB_STATEMENT_TIMEOUT_MS", "8000"),
            # 会话级只读：即使使用 root 账号，DDL/DML 也会被数据库拒绝（关键防线）
            "read_only_session": _env_bool("DB_READ_ONLY_SESSION", "true"),
            "validate_on_acquire": _env_bool("DB_VALIDATE_ON_ACQUIRE", "true"),
        },
    },
    "llm": {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "model": os.getenv("LLM_MODEL", "gpt-4"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        "temperature": _env_float("LLM_TEMPERATURE", "0.2"),
        "max_tokens": _env_int("LLM_MAX_TOKENS", "4000"),
        # 超时与重试：避免单次请求挂死占满线程池
        "timeout": _env_float("LLM_TIMEOUT", "30"),
        "max_retries": _env_int("LLM_MAX_RETRIES", "2"),
        "retry_backoff": _env_float("LLM_RETRY_BACKOFF", "0.5"),
    },
    "repair": {
        # SQL 执行失败后的自动修复轮数（0 = 关闭）
        "max_attempts": _env_int("REPAIR_MAX_ATTEMPTS", "2"),
        "enabled": _env_bool("REPAIR_ENABLED", "true"),
    },
    "router": {
        # 查询路由开关（关闭则所有问题都走 Text2SQL 单轮链路）
        "enabled": _env_bool("ROUTER_ENABLED", "true"),
        # auto | simple | analysis —— 客户端未显式指定 mode 时的默认值
        "default_mode": os.getenv("ROUTER_DEFAULT_MODE", "auto"),
        # 灰区是否请 LLM 判断（关闭则只用规则，零额外成本）
        "llm_classify": _env_bool("ROUTER_LLM_CLASSIFY", "true"),
        # 分析链路预算：子任务上限与建议超时（超时用于提示，不做强制中断）
        "analysis_max_steps": _env_int("ANALYSIS_MAX_STEPS", "5"),
        "analysis_timeout_seconds": _env_int("ANALYSIS_TIMEOUT_SECONDS", "120"),
        # 子步骤是否启用 Schema Linking / 指标 RAG。
        # 默认关闭：子问题由拆解器生成、表范围已知，走全量手写 Schema 更稳，
        # 且能避开 Schema Linking 的"LLM 慢速通道"这次大 prompt 调用。
        "analysis_use_schema_linking": _env_bool("ANALYSIS_USE_SCHEMA_LINKING", "false"),
        "analysis_use_indicator_rag": _env_bool("ANALYSIS_USE_INDICATOR_RAG", "false"),
    },
    "analysis": {
        # 步骤间传参时最多带多少行结果：整份结果集塞进 Prompt 会让大结果直接爆上下文
        "context_max_rows": _env_int("ANALYSIS_CONTEXT_MAX_ROWS", "20"),
        # 报告生成时每个步骤最多带多少字表格 + 多少行数据预览
        "report_max_chars": _env_int("ANALYSIS_REPORT_MAX_CHARS", "1200"),
        "report_rows_preview": _env_int("ANALYSIS_REPORT_ROWS_PREVIEW", "3"),
        # 一次分析折算成多少次配额：它实际要 6~12 次 LLM 调用，
        # 若与普通查询一样只扣 1 次，配额形同虚设
        "quota_cost": _env_int("ANALYSIS_QUOTA_COST", "5"),
    },
    "conversation": {
        # 多轮对话开关（关闭后每个请求都是独立单轮）
        "enabled": _env_bool("MULTITURN_ENABLED", "true"),
        # 会话空闲过期时间（秒）：超过则丢弃上下文，避免"昨天的上下文"污染今天
        "ttl_seconds": _env_int("CONVERSATION_TTL_SECONDS", "1800"),
        "max_sessions": _env_int("CONVERSATION_MAX_SESSIONS", "2000"),
        "max_turns": _env_int("CONVERSATION_MAX_TURNS", "20"),
        # 送进改写 Prompt 的历史轮数（不是越多越好：过多会稀释当前意图）
        "history_window": _env_int("CONVERSATION_HISTORY_WINDOW", "3"),
        # 是否允许 LLM 兜底改写（关掉则只用规则，零额外成本）
        "llm_rewrite": _env_bool("CONVERSATION_LLM_REWRITE", "true"),
    },
    "security": {
        # 未携带身份信息时使用的角色（仅 dev 生效；prod 下直接 401）
        "default_role": os.getenv("DEFAULT_ROLE", "admin"),
        # prod 环境强制要求身份，禁止匿名访问
        "require_identity_in_prod": _env_bool("SECURITY_REQUIRE_IDENTITY", "true"),
        # 允许访问的表白名单（未列出的表一律拒绝）
        "allowed_tables": [
            t.strip()
            for t in os.getenv(
                "SECURITY_ALLOWED_TABLES",
                "dim_customers,dim_products,sales_orders,exchange_rates,finance_expenses",
            ).split(",")
            if t.strip()
        ],
        # 限流：单用户每分钟请求数与每日请求数上限
        "rate_limit_per_minute": _env_int("RATE_LIMIT_PER_MINUTE", "20"),
        "rate_limit_per_day": _env_int("RATE_LIMIT_PER_DAY", "2000"),
        # 审计日志落盘路径（JSONL）
        "audit_log_path": os.getenv("AUDIT_LOG_PATH", str(BASE_DIR / "logs" / "audit.jsonl")),
        # CORS 白名单（逗号分隔；生产环境禁止 *）
        "cors_allow_origins": [
            o.strip()
            for o in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
            if o.strip()
        ],
    },
    "features": {
        "few_shot": _env_bool("FEATURE_FEW_SHOT", "true"),
        "rules": _env_bool("FEATURE_RULES", "true"),
        "guards": _env_bool("FEATURE_GUARDS", "true"),
        "indicator_knowledge": _env_bool("FEATURE_INDICATOR_KNOWLEDGE", "true"),
        "schema_linking": _env_bool("FEATURE_SCHEMA_LINKING", "false"),
        "indicator_rag": _env_bool("FEATURE_INDICATOR_RAG", "false"),
    },
}

LLM_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "base_url": os.getenv(
        "OPENAI_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    "model": os.getenv("LLM_MODEL", "qwen-max"),
    "max_tokens": _env_int("LLM_MAX_TOKENS", "4000"),
    "temperature": _env_float("LLM_TEMPERATURE", "0.2"),
    "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-v3"),
    "timeout": _env_float("LLM_TIMEOUT", "30"),
    "max_retries": _env_int("LLM_MAX_RETRIES", "2"),
    "retry_backoff": _env_float("LLM_RETRY_BACKOFF", "0.5"),
}

LLM_ANALYZER_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "base_url": os.getenv(
        "OPENAI_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    "model": os.getenv("LLM_ANALYZER_MODEL", "qwen-max"),
    "max_tokens": _env_int("LLM_ANALYZER_MAX_TOKENS", "2000"),
    "temperature": _env_float("LLM_ANALYZER_TEMPERATURE", "0.1"),
    "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-v3"),
    "timeout": _env_float("LLM_ANALYZER_TIMEOUT", "20"),
    "max_retries": _env_int("LLM_MAX_RETRIES", "2"),
    "retry_backoff": _env_float("LLM_RETRY_BACKOFF", "0.5"),
}


def is_prod() -> bool:
    """是否为生产环境。"""
    return str(APP_CONFIG.get("environment", "dev")).lower() in {"prod", "production"}
