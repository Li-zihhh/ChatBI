"""ChatBI 核心编排：解析 → 检索 → Prompt → LLM → 安全 → 执行 → 脱敏 → 格式化。

相对早期版本的关键变化：
  1. **安全层真正接入**：执行前 `prepare()`（AST 白名单 + 行级过滤），
     返回前 `mask()`（列血缘脱敏）。早期版本 `security.py` 从未被调用。
  2. **错误自修复闭环**：SQL 执行失败时把**数据库原始报错 + 修复指引**
     回灌给 LLM 重新生成（默认最多 2 轮），而不是只把错误展示给用户。
  3. **统一错误契约**：所有失败分支都返回顶层 `error_type`，
     上层可以直接映射 HTTP 状态码（早期版本这项映射因字段缺失而永久失效）。
  4. **分阶段耗时**：metadata 里记录检索 / 生成 / 执行各段耗时，
     既用于线上诊断，也用于评估体系采集延迟指标。
"""

from __future__ import annotations

import json
import logging
import re
from decimal import Decimal
from time import perf_counter
from typing import Any, Generator

from indicator_rag.indicator_knowledge import IndicatorKnowledge
from security import QuerySecurityManager, SecurityError, UserContext

from conversation import (
    ContextResolver,
    ConversationStore,
    InMemoryConversationStore,
    Resolution,
    Session,
    Turn,
    build_default_store,
)

from .database import DatabaseClient, QueryExecutionError
from .error_analyzer import ErrorAnalyzer
from .llm_client import LLMClient, clean_sql
from .prompt_builder import build_prompt
from .query_parser import QueryParser
from .result_formatter import ResultFormatter
from config import APP_CONFIG, LLM_ANALYZER_CONFIG, is_prod

logger = logging.getLogger("chatbi.core")

REPAIR_PROMPT_TEMPLATE = """你之前生成的 SQL 执行失败了，请修正后重新生成。

【上一次生成的 SQL】
{bad_sql}

【失败原因】
错误类型：{error_type}
根本原因：{root_cause}
数据库报错：{error_message}

【修复要求】
{repair_hint}

【原始用户问题】
{question}

请只输出修正后的完整 SQL，不要输出任何解释或注释。"""


class ChatBISystem:
    """ChatBI 系统主类。"""

    def __init__(
        self,
        security: QuerySecurityManager | None = None,
        db: DatabaseClient | None = None,
        llm: LLMClient | None = None,
        analyzer_llm: LLMClient | None = None,
        default_user: UserContext | None = None,
        conversation_store: ConversationStore | None = None,
        context_resolver: ContextResolver | None = None,
    ):
        self.parser = QueryParser()
        self.llm = llm or LLMClient()
        self.analyzer_llm = analyzer_llm or LLMClient(config=LLM_ANALYZER_CONFIG)
        self.db = db or DatabaseClient()
        self.formatter = ResultFormatter()
        self.indicator_knowledge = IndicatorKnowledge()
        self.security = security or QuerySecurityManager()
        self.default_user = default_user or UserContext.demo_admin()
        self.error_analyzer = ErrorAnalyzer(llm_client=self.analyzer_llm)

        repair_cfg = APP_CONFIG.get("repair", {})
        self.repair_enabled = bool(repair_cfg.get("enabled", True))
        self.max_repair_attempts = int(repair_cfg.get("max_attempts", 2))

        conversation_cfg = APP_CONFIG.get("conversation", {})
        self.multiturn_enabled = bool(conversation_cfg.get("enabled", True))
        self.conversation_store: ConversationStore = (
            conversation_store or build_default_store()
        )
        self.context_resolver = context_resolver or ContextResolver(
            llm_client=self.analyzer_llm,
            history_window=int(conversation_cfg.get("history_window", 3)),
            enable_llm=bool(conversation_cfg.get("llm_rewrite", True)),
        )

    # ==================== 权限上下文 ====================
    def _resolve_user(self, user: UserContext | None) -> UserContext:
        if user is not None:
            return user
        if is_prod():
            raise SecurityError("缺少身份上下文，拒绝访问。", code="missing_identity")
        # dev/CLI 便捷路径：显式回退到 demo 管理员，prod 下不会走到这里
        return self.default_user

    # ==================== 指标上下文 ====================
    def _resolve_indicator_context(
        self,
        user_question: str,
        use_indicator_knowledge: bool,
        use_indicator_rag: bool,
        indicator_knowledge: IndicatorKnowledge,
    ) -> tuple[list[str], str]:
        """RAG 优先、关键词兜底。检索失败会降级但不静默——必须留日志。"""
        if use_indicator_rag:
            try:
                from indicator_rag.indicator_retriever import retrieve_indicator_context

                context = retrieve_indicator_context(user_question)
                if context["indicator_block"]:
                    return context["detected_indicators"], context["indicator_block"]
            except Exception as exc:
                logger.warning("指标 RAG 检索失败，降级到关键词匹配：%s", exc)

            if use_indicator_knowledge:
                context = indicator_knowledge.get_indicator_context(user_question)
                return context["detected_indicators"], context["indicator_block"]
            return [], ""

        if use_indicator_knowledge:
            context = indicator_knowledge.get_indicator_context(user_question)
            return context["detected_indicators"], context["indicator_block"]

        return [], ""

    # ==================== 多轮上下文 ====================
    def _prepare_context(
        self,
        session_id: str | None,
        user_context: UserContext,
        question: str,
        enable_multiturn: bool | None,
    ) -> tuple[Any | None, Any | None]:
        """取出会话并把追问改写为独立问题。

        返回 (session, resolution)；未启用或计算失败时返回 (None, None)，
        调用方据此退回单轮行为——**多轮是增强项，不能成为故障源**。
        """
        enabled = self.multiturn_enabled if enable_multiturn is None else enable_multiturn
        if not enabled or not session_id:
            return None, None

        try:
            session = self.conversation_store.get_or_create(
                session_id=session_id,
                user_id=user_context.user_id,
                role=user_context.role,
            )
            resolution = self.context_resolver.resolve(session, question)
            return session, resolution
        except Exception as exc:
            logger.warning("多轮上下文处理失败，降级为单轮：%s", exc)
            return None, None

    def _record_turn(
        self,
        session: Any,
        question: str,
        resolution: Any,
        result: dict,
    ) -> None:
        """记录本轮结果，供下一轮改写使用。只存结构化摘要，不存结果数据。"""
        try:
            turn = Turn(
                index=len(session.turns) + 1,
                question=question,
                resolved_question=(
                    resolution.question if resolution is not None else question
                ),
                strategy=resolution.strategy if resolution is not None else "none",
                confidence=resolution.confidence if resolution is not None else 1.0,
                sql=result.get("sql"),
                columns=list(result.get("columns") or []),
                row_count=(
                    len(result.get("results") or []) if result.get("success") else None
                ),
                error_type=None if result.get("success") else result.get("error_type"),
            )
            self.conversation_store.append_turn(session, turn)
        except Exception as exc:
            logger.warning("多轮对话记录失败（不影响本次结果）：%s", exc)

    # ==================== 主流程（同步） ====================
    def run(
        self,
        user_question: str,
        user: UserContext | None = None,
        session_id: str | None = None,
        enable_multiturn: bool | None = None,
        use_few_shot: bool = True,
        use_rules: bool = False,
        use_guards: bool = False,
        use_indicator_knowledge: bool = False,
        use_schema_linking: bool = False,
        use_indicator_rag: bool = False,
        enable_repair: bool | None = None,
    ) -> dict:
        """运行完整链路（支持多轮），返回结构化结果。"""
        started = perf_counter()

        # 1) 权限上下文（prod 下缺失身份直接拒绝）
        try:
            user_context = self._resolve_user(user)
        except SecurityError as exc:
            return self._error_result(exc.error_type, str(exc), metadata=exc.metadata)

        # 2) 多轮上下文：把"那上个月呢"改写为独立问题
        session, resolution = self._prepare_context(
            session_id=session_id,
            user_context=user_context,
            question=user_question,
            enable_multiturn=enable_multiturn,
        )
        effective_question = (
            resolution.question if resolution is not None else user_question
        )

        # 3) 走原有单轮链路
        result = self._run_pipeline(
            user_question=effective_question,
            user_context=user_context,
            started=started,
            enable_repair=enable_repair,
            use_few_shot=use_few_shot,
            use_rules=use_rules,
            use_guards=use_guards,
            use_indicator_knowledge=use_indicator_knowledge,
            use_schema_linking=use_schema_linking,
            use_indicator_rag=use_indicator_rag,
        )

        # 4) 回填多轮元信息 + 记录本轮
        metadata = result.setdefault("metadata", {})
        if resolution is not None:
            metadata["multiturn"] = {
                **resolution.as_dict(),
                "session_id": session_id,
                "turn_index": len(session.turns) + 1,
            }
            if resolution.inherited:
                metadata["effective_question"] = effective_question
        if session is not None:
            self._record_turn(session, user_question, resolution, result)

        return result

    def _run_pipeline(
        self,
        user_question: str,
        user_context: UserContext,
        started: float,
        enable_repair: bool | None = None,
        use_few_shot: bool = True,
        use_rules: bool = False,
        use_guards: bool = False,
        use_indicator_knowledge: bool = False,
        use_schema_linking: bool = False,
        use_indicator_rag: bool = False,
    ) -> dict:
        """单轮链路本体：解析 → 检索 → Prompt → 生成 → 安全 → 执行 → 脱敏。"""
        repair_budget = (
            self.max_repair_attempts
            if (self.repair_enabled if enable_repair is None else enable_repair)
            else 0
        )

        # 2) 输入校验
        parsed = self.parser.parse(user_question)
        if not self.parser.validate(parsed):
            return self._error_result("validation", "输入问题为空")

        # 3) 指标上下文
        retrieval_started = perf_counter()
        detected_indicators, indicator_block = self._resolve_indicator_context(
            user_question=user_question,
            use_indicator_knowledge=use_indicator_knowledge,
            use_indicator_rag=use_indicator_rag,
            indicator_knowledge=self.indicator_knowledge,
        )
        retrieval_ms = (perf_counter() - retrieval_started) * 1000

        # 4) Prompt
        try:
            system_msg, prompt = build_prompt(
                user_question,
                use_few_shot=use_few_shot,
                use_rules=use_rules,
                use_guards=use_guards,
                indicator_knowledge=indicator_block,
                use_schema_linking=use_schema_linking,
            )
        except Exception as exc:
            logger.exception("Prompt 构造失败")
            return self._error_result("internal_server_error", f"Prompt 构造失败：{exc}")

        base_prompt = prompt
        generated_sql = ""
        analysis: dict | None = None
        last_error: dict[str, Any] | None = None
        llm_ms = 0.0
        db_ms = 0.0

        # 5) 生成 → 安全校验 → 执行 → 脱敏，失败则回灌重试
        for attempt in range(repair_budget + 1):
            llm_started = perf_counter()
            try:
                generated_sql = self.llm.generate_sql(system_msg, prompt)
            except Exception as exc:
                logger.warning("LLM 调用失败：%s", exc)
                return self._error_result(
                    "llm",
                    f"LLM 调用失败：{exc}",
                    metadata={"duration_ms": self._elapsed_ms(started)},
                )
            llm_ms += (perf_counter() - llm_started) * 1000

            if not generated_sql:
                last_error = {"error_type": "database_sql_syntax", "error": "模型未返回 SQL"}
                analysis = self.error_analyzer.categorize_error(
                    "", "模型返回空 SQL", user_question
                )
            else:
                # 5.1 安全校验 + 行级过滤
                try:
                    secured = self.security.prepare(generated_sql, user_context)
                except SecurityError as exc:
                    last_error = {
                        "error_type": exc.error_type,
                        "error": str(exc),
                        "code": exc.code,
                        "errno": None,
                    }
                    analysis = self.error_analyzer.categorize_error(
                        generated_sql, str(exc), user_question, error_type=exc.error_type
                    )
                    if not exc.is_repairable:
                        return self._error_result(
                            exc.error_type,
                            str(exc),
                            sql=generated_sql,
                            analysis=analysis,
                            metadata={**exc.metadata, "duration_ms": self._elapsed_ms(started)},
                        )
                else:
                    # 5.2 执行
                    db_started = perf_counter()
                    try:
                        columns, rows, db_meta = self.db.execute_query(secured.sql)
                    except QueryExecutionError as exc:
                        db_ms += (perf_counter() - db_started) * 1000
                        last_error = {
                            "error_type": exc.error_type,
                            "error": str(exc),
                            "errno": exc.errno,
                            "repairable": exc.repairable,
                        }
                        analysis = self.error_analyzer.categorize_error(
                            generated_sql,
                            str(exc),
                            user_question,
                            error_type=exc.error_type,
                            errno=exc.errno,
                        )
                        if not exc.repairable:
                            return self._error_result(
                                exc.error_type,
                                str(exc),
                                sql=generated_sql,
                                analysis=analysis,
                                metadata={**exc.metadata, "duration_ms": self._elapsed_ms(started)},
                            )
                    else:
                        db_ms += (perf_counter() - db_started) * 1000
                        # 5.3 脱敏
                        masked_columns, masked_rows = secured.mask(columns, rows)
                        formatted = self.formatter.format(masked_columns, masked_rows)
                        return {
                            "success": True,
                            "question": user_question,
                            "sql": generated_sql,
                            "executed_sql": secured.sql,
                            "columns": masked_columns,
                            "results": masked_rows,
                            "formatted": formatted,
                            "detected_indicators": detected_indicators,
                            "repair_attempts": attempt,
                            "metadata": {
                                "duration_ms": self._elapsed_ms(started),
                                "retrieval_ms": round(retrieval_ms, 2),
                                "llm_ms": round(llm_ms, 2),
                                "db_ms": round(db_ms, 2),
                                "row_count": db_meta.get("row_count"),
                                "truncated": db_meta.get("truncated"),
                                "slow_query": db_meta.get("slow_query"),
                                "explain": db_meta.get("explain"),
                                "row_filters": [
                                    f"{table}.{column}"
                                    for table, column in secured.policy.row_level_filters
                                ],
                                "masked_column_count": secured.masked_column_count(
                                    masked_columns
                                ),
                            },
                        }

            # 5.4 还有修复预算则回灌重试
            if attempt < repair_budget and analysis is not None:
                logger.info(
                    "SQL 第 %d 次执行失败（%s），回灌错误后重试",
                    attempt + 1,
                    last_error.get("error_type") if last_error else "unknown",
                )
                prompt = base_prompt + "\n\n" + REPAIR_PROMPT_TEMPLATE.format(
                    bad_sql=generated_sql,
                    error_type=analysis.get("error_type", "unknown"),
                    root_cause=analysis.get("root_cause", ""),
                    error_message=(last_error or {}).get("error", ""),
                    repair_hint=analysis.get("repair_hint", ""),
                    question=user_question,
                )

        return self._error_result(
            (last_error or {}).get("error_type", "internal_server_error"),
            (last_error or {}).get("error", "查询失败"),
            sql=generated_sql,
            analysis=analysis,
            metadata={
                "duration_ms": self._elapsed_ms(started),
                "retrieval_ms": round(retrieval_ms, 2),
                "llm_ms": round(llm_ms, 2),
                "db_ms": round(db_ms, 2),
                "repair_attempts": repair_budget,
            },
        )

    # ==================== 主流程（SSE 流式） ====================
    def run_stream(
        self,
        user_question: str,
        user: UserContext | None = None,
        session_id: str | None = None,
        enable_multiturn: bool | None = None,
        use_few_shot: bool = True,
        use_rules: bool = True,
        use_guards: bool = True,
        use_indicator_knowledge: bool = True,
        use_schema_linking: bool = False,
        use_indicator_rag: bool = False,
    ) -> Generator[str, None, None]:
        """流式执行（支持多轮），逐步推送 SSE 事件。

        事件类型：context / sql_chunk / sql_done / repair / result / error
        """
        try:
            user_context = self._resolve_user(user)
        except SecurityError as exc:
            yield _sse_event("error", {"error": str(exc), "error_type": exc.error_type})
            return

        session, resolution = self._prepare_context(
            session_id=session_id,
            user_context=user_context,
            question=user_question,
            enable_multiturn=enable_multiturn,
        )
        if resolution is not None and resolution.inherited:
            # 把改写结果显式推给前端，让"AI 理解成了什么"对用户透明
            yield _sse_event(
                "context", {**resolution.as_dict(), "session_id": session_id}
            )

        effective_question = (
            resolution.question if resolution is not None else user_question
        )
        outcome: dict[str, Any] = {}
        for event in self._stream_pipeline(
            user_question=effective_question,
            user_context=user_context,
            outcome=outcome,
            use_few_shot=use_few_shot,
            use_rules=use_rules,
            use_guards=use_guards,
            use_indicator_knowledge=use_indicator_knowledge,
            use_schema_linking=use_schema_linking,
            use_indicator_rag=use_indicator_rag,
        ):
            yield event

        if session is not None:
            self._record_turn(session, user_question, resolution, outcome)

    def _stream_pipeline(
        self,
        user_question: str,
        user_context: UserContext,
        outcome: dict[str, Any],
        use_few_shot: bool = True,
        use_rules: bool = True,
        use_guards: bool = True,
        use_indicator_knowledge: bool = True,
        use_schema_linking: bool = False,
        use_indicator_rag: bool = False,
    ) -> Generator[str, None, None]:
        """单轮流式链路本体；结果摘要写入 outcome，供多轮记录使用。"""
        parsed = self.parser.parse(user_question)
        if not self.parser.validate(parsed):
            outcome.update({"success": False, "error_type": "validation"})
            yield _sse_event("error", {"error": "输入问题为空", "error_type": "validation"})
            return

        detected_indicators, indicator_block = self._resolve_indicator_context(
            user_question=user_question,
            use_indicator_knowledge=use_indicator_knowledge,
            use_indicator_rag=use_indicator_rag,
            indicator_knowledge=self.indicator_knowledge,
        )

        try:
            system_msg, prompt = build_prompt(
                user_question,
                use_few_shot=use_few_shot,
                use_rules=use_rules,
                use_guards=use_guards,
                indicator_knowledge=indicator_block,
                use_schema_linking=use_schema_linking,
            )
        except Exception as exc:
            outcome.update({"success": False, "error_type": "internal_server_error"})
            yield _sse_event("error", {"error": f"Prompt 构造失败：{exc}", "error_type": "internal_server_error"})
            return

        base_prompt = prompt
        sql_parts: list[str] = []

        # 第一轮：流式产出，前端可逐字展示
        try:
            for chunk in self.llm.generate_sql_stream(system_msg, prompt):
                sql_parts.append(chunk)
                yield _sse_event("sql_chunk", {"content": chunk})
        except Exception as exc:
            outcome.update({"success": False, "error_type": "llm"})
            yield _sse_event("error", {"error": str(exc), "error_type": "llm"})
            return

        sql = clean_sql("".join(sql_parts))
        yield _sse_event("sql_done", {"sql": sql})

        repair_budget = self.max_repair_attempts if self.repair_enabled else 0
        for attempt in range(repair_budget + 1):
            analysis: dict | None = None
            last_error_message = ""
            try:
                secured = self.security.prepare(sql, user_context)
            except SecurityError as exc:
                if not exc.is_repairable or attempt >= repair_budget:
                    outcome.update(
                        {"success": False, "error_type": exc.error_type, "sql": sql}
                    )
                    yield _sse_event(
                        "error",
                        {"error": str(exc), "error_type": exc.error_type, "sql": sql, "code": exc.code},
                    )
                    return
                analysis = self.error_analyzer.categorize_error(
                    sql, str(exc), user_question, error_type=exc.error_type
                )
                last_error_message = str(exc)
            else:
                try:
                    columns, rows, db_meta = self.db.execute_query(secured.sql)
                except QueryExecutionError as exc:
                    if not exc.repairable or attempt >= repair_budget:
                        outcome.update(
                            {"success": False, "error_type": exc.error_type, "sql": sql}
                        )
                        yield _sse_event(
                            "error",
                            {
                                "error": str(exc),
                                "error_type": exc.error_type,
                                "sql": sql,
                                "errno": exc.errno,
                            },
                        )
                        return
                    analysis = self.error_analyzer.categorize_error(
                        sql,
                        str(exc),
                        user_question,
                        error_type=exc.error_type,
                        errno=exc.errno,
                    )
                    last_error_message = str(exc)
                else:
                    masked_columns, masked_rows = secured.mask(columns, rows)
                    outcome.update(
                        {
                            "success": True,
                            "sql": sql,
                            "columns": masked_columns,
                            "results": masked_rows,
                        }
                    )
                    yield _sse_event(
                        "result",
                        {
                            "columns": masked_columns,
                            "rows": [dict(zip(masked_columns, row)) for row in masked_rows],
                            "row_count": len(masked_rows),
                            "truncated": db_meta.get("truncated"),
                            "duration_ms": db_meta.get("duration_ms"),
                            "repair_attempts": attempt,
                            "detected_indicators": detected_indicators,
                        },
                    )
                    return

            if attempt >= repair_budget or analysis is None:
                outcome.update({"success": False, "error_type": "database", "sql": sql})
                yield _sse_event(
                    "error",
                    {"error": "重试次数已用尽", "error_type": "database", "sql": sql},
                )
                return

            # 回灌错误，非流式重新生成（避免流式重试造成前端重复输出）
            yield _sse_event(
                "repair",
                {
                    "attempt": attempt + 1,
                    "max_attempts": repair_budget,
                    "error_type": analysis.get("error_type"),
                    "root_cause": analysis.get("root_cause"),
                },
            )
            repair_prompt = base_prompt + "\n\n" + REPAIR_PROMPT_TEMPLATE.format(
                bad_sql=sql,
                error_type=analysis.get("error_type", "unknown"),
                root_cause=analysis.get("root_cause", ""),
                error_message=last_error_message,
                repair_hint=analysis.get("repair_hint", ""),
                question=user_question,
            )
            try:
                sql = self.llm.generate_sql(system_msg, repair_prompt)
            except Exception as exc:
                yield _sse_event("error", {"error": f"修复重试失败：{exc}", "error_type": "llm"})
                return
            yield _sse_event("sql_done", {"sql": sql, "repaired": True})

    # ==================== 辅助 ====================
    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((perf_counter() - started) * 1000, 2)

    @staticmethod
    def _error_result(
        error_type: str,
        message: str,
        sql: str | None = None,
        analysis: dict | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """统一错误返回结构：error_type 必须在顶层，供 API 映射状态码。"""
        result: dict[str, Any] = {
            "success": False,
            "error_type": error_type,
            "error": message,
            "metadata": metadata or {},
        }
        if sql:
            result["sql"] = sql
        if analysis:
            result["analysis"] = {
                key: analysis.get(key)
                for key in ("error_type", "root_cause", "suggestion", "layer", "repairable")
            }
        return result


def _sse_event(event_type: str, data: dict) -> str:
    """构造 SSE 格式的事件字符串。"""
    return (
        f"event: {event_type}\n"
        f"data: {json.dumps(data, ensure_ascii=False, default=_json_serializer)}\n\n"
    )


def _json_serializer(obj):
    """JSON 序列化补充：处理 Decimal 等非标准类型。"""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def main() -> None:
    """命令行入口：python -m text2SQL.chatbi_system "问题" 。"""
    import sys

    system = ChatBISystem()
    print("=" * 60)
    print("ChatBI Text2SQL 系统")
    print("=" * 60)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        result = system.run(question)
        print(f"\n问题：{question}")
        print(f"SQL：{result.get('sql', '')}")
        if result["success"]:
            print(f"\n{result['formatted']}")
        else:
            print(f"\n错误[{result.get('error_type')}]：{result['error']}")
        return

    print("\n请输入问题（输入 exit/quit/q 退出）：")
    while True:
        try:
            question = input("\n> ")
            if question.strip().lower() in {"exit", "quit", "q"}:
                break
            result = system.run(question)
            print(f"\nSQL：{result.get('sql', '')}")
            print(result["formatted"] if result["success"] else f"错误：{result['error']}")
        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    main()
