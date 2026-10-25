"""FastAPI 服务入口。

相对早期版本的关键变化：
  1. **身份来自受信头且 fail-close**：删除了请求体里的 `user_role` / `user_region`
     （早期版本允许调用方在 body 里自封管理员），未知角色直接 403；
     `APP_ENV=prod` 时缺少身份信息直接 401。
  2. **限流 + 审计**：LIMIT 与 LIMIT 之外的成本控制，以及合规所需的行为留痕。
  3. **错误码映射真正生效**：`ChatBISystem` 现在保证返回顶层 `error_type`，
     这里用映射表把它翻译成 4xx/5xx。
  4. **CORS 收敛**：生产环境禁止通配符。
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import unquote

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import APP_CONFIG, is_prod
from guardrails import AuditLogger, RateLimitError, RateLimiter
from router import QueryRouter, RouteDecision
from security import ROLE_POLICIES, SecurityError, UserContext, get_policy
from text2SQL.chatbi_system import ChatBISystem, _sse_event

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("chatbi.api")

_SECURITY_CFG = APP_CONFIG.get("security", {})

app = FastAPI(
    title="ChatBI MVP API",
    version="0.3.0",
    description="""
## ChatBI MVP API

企业级 ChatBI 最小可行产品的服务化接口。

### 接口概览

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/v1/query` | POST | 同步查询，一次性返回完整结果 |
| `/api/v1/query/stream` | POST | SSE 流式查询，逐步推送 SQL 和结果 |
| `/api/v1/usage` | GET | 查询当前用户的限流配额使用情况 |
| `/health` | GET | 健康检查 |

### 身份与权限

通过请求头传递受信身份（生产环境应由网关/SSO 注入，禁止由客户端直连伪造）：

| 请求头 | 说明 | 取值 |
|--------|------|------|
| `x-user-id` | 用户标识 | 任意字符串 |
| `x-user-role` | 角色 | `admin` / `finance` / `sales` |
| `x-user-region` | 用户所属区域（`sales` 必需） | 如 `欧洲` |

### SSE 流式接口事件类型

| 事件类型 | 说明 | data 结构 |
|----------|------|-----------|
| `sql_chunk` | SQL 文本片段 | `{"content": "SELECT..."}` |
| `sql_done` | SQL 完整输出 | `{"sql": "..."}` |
| `repair` | 执行失败后进入自动修复 | `{"attempt": 1, "error_type": "..."}` |
| `result` | 查询结果 | `{"columns": [...], "rows": [...], "row_count": N}` |
| `error` | 异常信息 | `{"error": "...", "error_type": "..."}` |
""",
    openapi_tags=[
        {"name": "查询", "description": "自然语言转 SQL 查询接口"},
        {"name": "系统", "description": "系统运维与监控接口"},
    ],
)

system = ChatBISystem()
rate_limiter = RateLimiter(
    per_minute=int(_SECURITY_CFG.get("rate_limit_per_minute", 20)),
    per_day=int(_SECURITY_CFG.get("rate_limit_per_day", 2000)),
)
audit_logger = AuditLogger(_SECURITY_CFG.get("audit_log_path", "logs/audit.jsonl"))

# ==================== 查询路由 ====================
# 简单取数 → Text2SQL 单轮（1~3 次 LLM，1~3s）
# 复杂归因 → Plan-and-Execute（6~12 次 LLM，15~40s）
router = QueryRouter(llm_client=system.analyzer_llm)


def analysis_runner(question: str, user: UserContext) -> dict[str, Any]:
    """分析链路入口（独立成模块级函数，便于测试替换）。"""
    from analysis_service import run_analysis

    return run_analysis(question=question, user=user, system=system)


# ==================== CORS ====================
_origins = _SECURITY_CFG.get("cors_allow_origins") or ["*"]
if is_prod() and "*" in _origins:
    logger.warning(
        "生产环境检测到 CORS 通配符，已自动收敛为同源访问。请配置 CORS_ALLOW_ORIGINS。"
    )
    _origins = []
elif "*" in _origins:
    logger.warning("CORS 允许所有来源（dev 模式）。生产环境请配置 CORS_ALLOW_ORIGINS。")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False if "*" in _origins else True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "x-user-id", "x-user-role", "x-user-region"],
)


# ==================== 请求 / 响应模型 ====================
class QueryRequest(BaseModel):
    """查询请求体。

    注意：这里**不包含** user_role / user_region。
    权限字段绝不能由业务请求体声明，否则等同于把授权开关交给调用方。
    """

    question: str = Field(..., min_length=1, max_length=2000, description="业务人员的自然语言问题")
    session_id: str | None = Field(
        default=None,
        max_length=64,
        description="会话标识；传入后启用多轮上下文（追问如“那上个月呢”会被改写为独立问题）",
    )
    mode: str | None = Field(
        default=None,
        description=(
            "查询模式：auto（默认，自动判断）/ simple（强制走 Text2SQL 单轮）/ "
            "analysis（强制走多步分析）"
        ),
        pattern="^(auto|simple|analysis)$",
    )
    use_few_shot: bool | None = Field(default=None, description="是否启用 Few-shot 示例")
    use_rules: bool | None = Field(default=None, description="是否启用业务规则约束")
    use_guards: bool | None = Field(default=None, description="是否启用错误防护")
    use_indicator_knowledge: bool | None = Field(default=None, description="是否注入指标知识")
    use_schema_linking: bool | None = Field(default=None, description="是否启用 Schema Linking")
    use_indicator_rag: bool | None = Field(default=None, description="是否启用指标 RAG")
    source_id: str | None = Field(default=None, description="数据源标识；未传时使用默认数据源")


class HealthResponse(BaseModel):
    status: str
    database_connected: bool


class QuerySuccessResponse(BaseModel):
    """统一响应。kind=table 时看 columns/rows；kind=report 时看 report/steps。"""

    success: bool = True
    kind: str = Field(default="table", description="table（单轮取数）或 report（多步分析）")
    route: dict[str, Any] | None = Field(
        default=None, description="路由决策：path / confidence / reason / signals"
    )
    question: str
    sql: str = ""
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    formatted: str = ""
    report: dict[str, Any] | None = Field(default=None, description="kind=report 时的结构化报告")
    steps: list[dict[str, Any]] = Field(
        default_factory=list, description="kind=report 时的各步骤执行摘要"
    )
    metadata: dict[str, Any]


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    error_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# ==================== 身份解析 ====================
def _decode_header(value: str | None) -> str | None:
    """解码请求头中的非 ASCII 值。

    HTTP 头部按 latin-1 传输，中文区域名（如"欧洲"）无法直接放入头部。
    调用方应做百分号编码（`欧洲` → `%E6%AC%A7%E6%B4%B2`），这里统一解码。
    """
    if not value:
        return value
    try:
        return unquote(value)
    except Exception:  # pragma: no cover - 畸形编码时保持原值
        return value


def _build_user_context(request: Request) -> UserContext:
    """从受信请求头解析权限上下文。

    fail-close 策略：
      - prod 环境无身份 → 401
      - 未知角色       → 403
      - 缺区域（sales）→ 由 security 层拒绝
    """
    user_id = _decode_header(request.headers.get("x-user-id"))
    role = _decode_header(request.headers.get("x-user-role"))
    region = _decode_header(request.headers.get("x-user-region"))

    if not user_id and not role:
        if is_prod() and _SECURITY_CFG.get("require_identity_in_prod", True):
            raise HTTPException(
                status_code=401, detail="缺少身份信息（x-user-id / x-user-role）。"
            )
        role = _SECURITY_CFG.get("default_role", "admin")
        user_id = f"anonymous_{role}"
        logger.warning("未提供身份信息，dev 模式下回退到角色 %s", role)

    normalized_role = (role or "").strip().lower()
    if normalized_role not in ROLE_POLICIES:
        raise HTTPException(
            status_code=403, detail=f"未知角色：{role!r}。"
        )

    return UserContext(
        user_id=user_id or f"user_{normalized_role}",
        role=normalized_role,
        region=region,
    )


def _resolve_query_options(payload: QueryRequest) -> dict[str, bool]:
    feature_defaults = APP_CONFIG.get("features", {})
    return {
        "use_few_shot": payload.use_few_shot
        if payload.use_few_shot is not None
        else feature_defaults.get("few_shot", False),
        "use_rules": payload.use_rules
        if payload.use_rules is not None
        else feature_defaults.get("rules", False),
        "use_guards": payload.use_guards
        if payload.use_guards is not None
        else feature_defaults.get("guards", False),
        "use_indicator_knowledge": payload.use_indicator_knowledge
        if payload.use_indicator_knowledge is not None
        else feature_defaults.get("indicator_knowledge", False),
        "use_schema_linking": payload.use_schema_linking
        if payload.use_schema_linking is not None
        else feature_defaults.get("schema_linking", False),
        "use_indicator_rag": payload.use_indicator_rag
        if payload.use_indicator_rag is not None
        else feature_defaults.get("indicator_rag", False),
    }


# error_type → HTTP 状态码
_STATUS_MAP = {
    "validation": 400,
    "security": 403,
    "llm": 502,
    "database_sql_syntax": 422,
    "database_execution_error": 422,
    "database_permission_denied": 403,
    "database_connection_error": 503,
    "database_query_timeout": 504,
    "database_pool_exhausted": 503,
    "analysis_failed": 502,
    "analysis_steps_failed": 502,
    "rate_limited": 429,
    "internal_server_error": 500,
}


def _quota_cost(decision: RouteDecision) -> int:
    """按链路折算配额消耗：分析链路一次抵多次普通查询。"""
    if decision.path == "analysis":
        return int(APP_CONFIG.get("analysis", {}).get("quota_cost", 5))
    return 1


def _check_quota(user: UserContext, cost: int = 1) -> None:
    """扣减配额。cost 允许一次请求折算多次，用于覆盖分析类请求的真实成本。

    分析链路一次要 6~12 次 LLM 调用，若与普通查询一样只扣 1 次，
    日配额 2000 就等于放出去两万多次模型调用，限流形同虚设。
    """
    try:
        for _ in range(max(1, cost)):
            rate_limiter.check(user.user_id)
    except RateLimitError as exc:
        audit_logger.log(
            user_id=user.user_id, role=user.role, outcome="rate_limited", reason=str(exc)
        )
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": str(exc.retry_after)},
        ) from exc


def _audit(user: UserContext, question: str, result: dict | None, duration_ms: float, outcome: str = "ok") -> None:
    audit_logger.log(
        user_id=user.user_id,
        role=user.role,
        region=user.region,
        question=question,
        sql=(result or {}).get("sql"),
        error_type=(result or {}).get("error_type"),
        success=(result or {}).get("success"),
        row_count=((result or {}).get("metadata") or {}).get("row_count"),
        repair_attempts=(result or {}).get("repair_attempts"),
        duration_ms=duration_ms,
        outcome=outcome,
    )


# ==================== 异常处理器 ====================
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="请求参数校验失败",
            error_type="request_validation",
            metadata={"path": str(request.url.path), "details": exc.errors()},
        ).model_dump(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """统一错误响应。detail 为 dict 时（查询链路的结构化错误）直接映射为 ErrorResponse。"""
    if isinstance(exc.detail, dict):
        payload = ErrorResponse(
            error=str(exc.detail.get("message", "请求失败")),
            error_type=str(exc.detail.get("error_type", "http_exception")),
            metadata={
                key: value
                for key, value in exc.detail.items()
                if key not in {"message", "error_type"}
            },
        ).model_dump()
    else:
        payload = ErrorResponse(
            error=str(exc.detail),
            error_type="http_exception",
            metadata={"path": str(request.url.path)},
        ).model_dump()

    return JSONResponse(
        status_code=exc.status_code,
        content=payload,
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled server error: %s", exc)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="服务内部异常",
            error_type="internal_server_error",
            metadata={"path": str(request.url.path)},
        ).model_dump(),
    )


# ==================== 路由 ====================
@app.get("/", tags=["系统"])
def read_root() -> dict[str, str]:
    return {
        "name": "ChatBI MVP API",
        "version": "0.3.0",
        "docs": "/docs",
        "health": "/health",
        "query": "/api/v1/query",
        "query_stream": "/api/v1/query/stream",
    }


@app.get("/health", response_model=HealthResponse, tags=["系统"])
def health_check() -> HealthResponse:
    return HealthResponse(status="ok", database_connected=system.db.validate_connection())


@app.get("/api/v1/usage", tags=["系统"], summary="查询当前用户的配额使用情况")
def get_usage(request: Request) -> dict[str, Any]:
    user = _build_user_context(request)
    return {"user_id": user.user_id, "role": user.role, **rate_limiter.snapshot(user.user_id)}


@app.delete("/api/v1/sessions/{session_id}", tags=["系统"], summary="清除指定会话的多轮上下文")
def delete_session(session_id: str, request: Request) -> dict[str, Any]:
    """主动清除会话上下文。适用于"重新开始"按钮或切换主题时。

    只允许清除自己的会话：会话存储已按 user_id 绑定，
    即便这里不校验，跨用户清除也只会清掉一个空会话。
    """
    user = _build_user_context(request)
    removed = system.conversation_store.reset(session_id, user_id=user.user_id)
    if not removed:
        # 不区分"不存在"与"非本人"，避免被用来探测他人会话是否存在
        logger.info(
            "会话清除未生效（不存在或非本人）：session=%s user=%s",
            session_id,
            user.user_id,
        )
    audit_logger.log(
        user_id=user.user_id, role=user.role, outcome="session_reset", session_id=session_id
    )
    return {"session_id": session_id, "reset": removed}


@app.post(
    "/api/v1/query",
    response_model=QuerySuccessResponse,
    tags=["查询"],
    summary="同步查询（一次性返回）",
    responses={
        400: {"model": ErrorResponse, "description": "输入问题不合法"},
        401: {"model": ErrorResponse, "description": "缺少身份信息"},
        403: {"model": ErrorResponse, "description": "权限不足或安全策略拒绝"},
        422: {"model": ErrorResponse, "description": "生成的 SQL 无法解析或执行"},
        429: {"model": ErrorResponse, "description": "超出限流配额"},
        502: {"model": ErrorResponse, "description": "LLM 调用失败"},
        503: {"model": ErrorResponse, "description": "数据库连接异常或连接池耗尽"},
        504: {"model": ErrorResponse, "description": "数据库查询超时"},
    },
)
def query_chatbi(payload: QueryRequest, request: Request) -> QuerySuccessResponse:
    """执行自然语言查询：先路由，再分派到单轮链路或多步分析链路。"""
    started_at = perf_counter()
    user = _build_user_context(request)
    query_options = _resolve_query_options(payload)

    # 1) 路由决策
    mode = payload.mode or APP_CONFIG.get("router", {}).get("default_mode", "auto")
    decision = router.route(payload.question, mode=mode)  # type: ignore[arg-type]
    logger.info(
        "query user=%s role=%s route=%s conf=%.2f src=%s reason=%s q=%s",
        user.user_id,
        user.role,
        decision.path,
        decision.confidence,
        decision.source,
        decision.reason,
        payload.question,
    )

    # 2) 按真实成本扣配额（分析链路一次要 6~12 次模型调用）
    _check_quota(user, cost=_quota_cost(decision))

    # 3) 直接走分析链路
    if decision.path == "analysis":
        return _respond_analysis(payload, user, decision, started_at)

    # 3) 走单轮链路
    try:
        result = system.run(
            user_question=payload.question,
            user=user,
            session_id=payload.session_id,
            **query_options,
        )
    except SecurityError as exc:
        duration_ms = round((perf_counter() - started_at) * 1000, 2)
        _audit(user, payload.question, {"error_type": exc.error_type}, duration_ms, "security_error")
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except Exception:
        duration_ms = round((perf_counter() - started_at) * 1000, 2)
        logger.exception("查询链路异常")
        _audit(user, payload.question, {"error_type": "internal_server_error"}, duration_ms, "exception")
        raise HTTPException(status_code=500, detail="服务内部异常，请稍后重试。")

    # 4) 灰区问题：单轮失败或查不到数据 → 升级到分析链路再试一次
    if router.should_escalate(decision, result):
        logger.info("单轮结果不理想（success=%s），升级到分析链路", result.get("success"))
        escalated = RouteDecision(
            path="analysis",
            confidence=0.6,
            reason=f"单轮查询未得到有效结果，升级分析：{decision.reason}",
            signals=list(decision.signals),
            score=decision.score,
            source="escalation",
        )
        # 升级意味着真实成本变高：补扣差额（原本已按普通查询扣过 1 次）
        _check_quota(user, cost=max(0, _quota_cost(escalated) - 1))
        return _respond_analysis(payload, user, escalated, started_at)

    duration_ms = round((perf_counter() - started_at) * 1000, 2)

    if not result.get("success"):
        error_type = result.get("error_type", "internal_server_error")
        status_code = _STATUS_MAP.get(error_type, 500)
        _audit(user, payload.question, result, duration_ms, "failed")
        raise HTTPException(
            status_code=status_code,
            detail={
                "message": result.get("error", "查询失败"),
                "error_type": error_type,
                "analysis": result.get("analysis"),
                "route": decision.as_dict(),
                "hint": "若 SQL 语法/字段有误，系统已尝试自动修复；仍失败可换一种问法。",
            },
        )

    metadata = {
        **result.get("metadata", {}),
        "duration_ms": duration_ms,
        "repair_attempts": result.get("repair_attempts", 0),
        "detected_indicators": result.get("detected_indicators", []),
        "sql_rewritten": result.get("executed_sql") != result.get("sql"),
        "role": user.role,
    }
    _audit(user, payload.question, result, duration_ms, "ok")
    logger.info("query ok in %.2f ms (repair=%s)", duration_ms, result.get("repair_attempts"))

    return QuerySuccessResponse(
        kind="table",
        route=decision.as_dict(),
        question=payload.question,
        sql=result["sql"],
        columns=result["columns"],
        rows=[dict(zip(result["columns"], row)) for row in result["results"]],
        formatted=result["formatted"],
        metadata=metadata,
    )


def _respond_analysis(
    payload: QueryRequest,
    user: UserContext,
    decision: RouteDecision,
    started_at: float,
) -> QuerySuccessResponse:
    """执行分析链路并归一化为 API 响应。"""
    result = analysis_runner(payload.question, user)
    duration_ms = round((perf_counter() - started_at) * 1000, 2)

    if not result.get("success"):
        error_type = result.get("error_type", "analysis_failed")
        _audit(user, payload.question, result, duration_ms, "analysis_failed")
        raise HTTPException(
            status_code=_STATUS_MAP.get(error_type, 500),
            detail={
                "message": result.get("error", "分析失败"),
                "error_type": error_type,
                "route": decision.as_dict(),
                "steps": result.get("steps") or [],
                "hint": "多步分析依赖多次模型调用，失败时可重试，或改用更具体的取数问题。",
            },
        )

    _audit(user, payload.question, result, duration_ms, "analysis_ok")
    logger.info(
        "analysis ok in %.2f ms (steps=%s/%s)",
        duration_ms,
        (result.get("metadata") or {}).get("step_succeeded"),
        (result.get("metadata") or {}).get("step_total"),
    )

    metadata = {
        **(result.get("metadata") or {}),
        "duration_ms": duration_ms,
        "role": user.role,
    }
    return QuerySuccessResponse(
        kind="report",
        route=decision.as_dict(),
        question=payload.question,
        sql=result.get("sql", ""),
        columns=result.get("columns") or [],
        rows=[],
        formatted=result.get("formatted", ""),
        report=result.get("report"),
        steps=result.get("steps") or [],
        metadata=metadata,
    )


@app.post("/api/v1/query/stream", tags=["查询"], summary="SSE 流式查询（逐步推送）")
async def query_chatbi_stream(payload: QueryRequest, request: Request) -> StreamingResponse:
    """以 SSE 流式返回结果。

    - `route`：先推送路由决策（path/confidence/reason）
    - 简单查询：`sql_chunk` / `sql_done` / `repair` / `result` / `error`
    - 复杂分析：`status`（进度提示）→ `report`（Markdown 报告）
    """
    user = _build_user_context(request)
    # 先按 1 次扣一次（保证即使路由阶段就失败也有记账），
    # 若路由判定为分析链路，再在生成器里补扣差额。
    _check_quota(user)
    query_options = _resolve_query_options(payload)
    logger.info("stream user=%s role=%s q=%s", user.user_id, user.role, payload.question)

    def event_generator():
        started_at = perf_counter()
        last_error_type: str | None = None
        sql_seen: str | None = None
        try:
            # 路由必须放在生成器里：它是同步阻塞调用（可能触发 LLM），
            # 在 async 函数体里直接调用会阻塞事件循环。
            mode = payload.mode or APP_CONFIG.get("router", {}).get("default_mode", "auto")
            decision = router.route(payload.question, mode=mode)  # type: ignore[arg-type]
            yield _sse_event("route", decision.as_dict())

            if decision.path == "analysis":
                # 响应头已经发出，配额不足只能以 SSE error 事件表达
                try:
                    _check_quota(user, cost=max(0, _quota_cost(decision) - 1))
                except HTTPException as exc:
                    last_error_type = "rate_limited"
                    yield _sse_event(
                        "error",
                        {"error": str(exc.detail), "error_type": "rate_limited"},
                    )
                    return
                yield _sse_event(
                    "status",
                    {"message": "这是个分析类问题，正在拆解并分步查询，预计 15~40 秒..."},
                )
                result = analysis_runner(payload.question, user)
                if result.get("success"):
                    yield _sse_event(
                        "report",
                        {
                            "markdown": result.get("formatted", ""),
                            "report": result.get("report"),
                            "steps": result.get("steps") or [],
                            "duration_ms": (result.get("metadata") or {}).get("duration_ms"),
                        },
                    )
                else:
                    last_error_type = result.get("error_type") or "analysis_failed"
                    yield _sse_event(
                        "error",
                        {
                            "error": result.get("error", "分析失败"),
                            "error_type": last_error_type,
                            "steps": result.get("steps") or [],
                        },
                    )
                return

            for event_str in system.run_stream(
                user_question=payload.question,
                user=user,
                session_id=payload.session_id,
                **query_options,
            ):
                if event_str.startswith("event: sql_done"):
                    sql_seen = event_str
                elif event_str.startswith("event: error"):
                    last_error_type = "stream_error"
                yield event_str
        except Exception as exc:  # 兜底：不让异常穿透到 ASGI 层
            logger.exception("SSE 流异常")
            last_error_type = "internal_server_error"
            yield (
                'event: error\ndata: {"error": "服务内部异常", '
                '"error_type": "internal_server_error"}\n\n'
            )
        finally:
            _audit(
                user,
                payload.question,
                {"error_type": last_error_type, "sql": sql_seen},
                round((perf_counter() - started_at) * 1000, 2),
                "stream_failed" if last_error_type else "stream_ok",
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx 环境下禁止缓冲
        },
    )


# ==================== 静态文件 ====================
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("api_service:app", host="127.0.0.1", port=8000, reload=False)