"""多轮对话上下文管理。

**与 `docs/典型问题/前置上下文处理机制.md` 的区别（重要）**

那份文档描述的是**单次请求内** Plan-and-Execute 的步骤间依赖注入
（`step_1` 的结果喂给 `step_3`），解决的是"一条复杂问题怎么拆成多步 SQL"。
本模块解决的是**跨请求**的用户会话：用户上一轮问了什么、这一轮的"那上个月呢"
指的是什么。两者数据形态相似（都是"把上一段结果/文本拼进 Prompt"），
但生命周期、作用域、安全要求完全不同：

| | 步骤间依赖注入 | 多轮会话上下文 |
|---|---|---|
| 生命周期 | 一次请求内 | 多次请求、分钟级 |
| 作用域 | 单用户单次调用 | 跨用户共享的进程，必须绑定身份 |
| 安全要求 | 无（同一请求内数据同源） | **必须按 user_id + role 隔离，否则串号泄露** |
| 失效控制 | 无（请求结束即释放） | TTL + 最大轮数 + 主动清除 |

**设计取舍**

1. **改写而非拼接历史**。目标是把"那上个月呢"变成一条独立问题
   （"欧洲市场上个月的销售额（人民币）"），再走原有 Text2SQL 链路。
   这样安全层、修复闭环、Schema Linking 全部不用改，且改写结果可审计、可展示给用户确认。
2. **规则优先、LLM 兜底**（与本项目 Schema Linking 的双通道思路一致）。
   真实流量里 70% 以上的追问是三类固定形态：换时间、换区域、换维度，
   用正则即可确定性处理，零成本、可测试；只有开放性指代才走 LLM。
3. **历史里不放结果数据**。只传问题、SQL、列名、行数。
   既省 Token，也避免把业务数据反复送进第三方 API——这是合规问题。
4. **失败即降级为单轮**。改写失败/置信度低时使用原始问题，不猜测。
   猜错的代价（答非所问）高于多问一次的代价。
"""

from __future__ import annotations

import json
import logging
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import time
from typing import Any

from config import APP_CONFIG

logger = logging.getLogger("chatbi.conversation")


# ==================== 数据结构 ====================
@dataclass(slots=True)
class Turn:
    """一轮对话。question 是用户原话，resolved_question 是真正送进链路的独立问题。"""

    index: int
    question: str
    resolved_question: str
    strategy: str = "none"
    confidence: float = 1.0
    sql: str | None = None
    columns: list[str] = field(default_factory=list)
    row_count: int | None = None
    error_type: str | None = None
    created_at: float = field(default_factory=time)


@dataclass
class Session:
    """一次会话。绑定 user_id + role，防止跨用户串号。"""

    session_id: str
    user_id: str
    role: str
    turns: list[Turn] = field(default_factory=list)
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)

    def last_turn(self) -> Turn | None:
        return self.turns[-1] if self.turns else None

    def recent(self, limit: int) -> list[Turn]:
        return self.turns[-limit:] if limit > 0 else []

    def is_expired(self, ttl_seconds: float, now: float | None = None) -> bool:
        return (now or time()) - self.updated_at > ttl_seconds


@dataclass(slots=True)
class Resolution:
    """问题改写结果。strategy 用于观测"哪条路径生效"。"""

    question: str
    strategy: str = "none"
    confidence: float = 1.0
    inherited: bool = False
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "resolved_question": self.question,
            "strategy": self.strategy,
            "confidence": round(self.confidence, 3),
            "inherited": self.inherited,
            "note": self.note,
        }


# ==================== 会话存储 ====================
class ConversationStore(ABC):
    """会话存储抽象。默认进程内存实现；多实例部署时应换 Redis（见文档）。"""

    @abstractmethod
    def get_or_create(self, session_id: str, user_id: str, role: str) -> Session: ...

    @abstractmethod
    def append_turn(self, session: Session, turn: Turn) -> Session: ...

    @abstractmethod
    def reset(self, session_id: str) -> bool: ...

    @abstractmethod
    def purge_expired(self) -> int: ...


class InMemoryConversationStore(ConversationStore):
    """进程内会话存储（TTL + 容量上限 + 身份绑定）。

    三个必须有的护栏：
      - TTL：过期会话自动失效，避免"昨天的上下文"污染今天的提问
      - 容量上限：按最近访问淘汰，防止会话字典无限增长吃满内存
      - 身份绑定：session_id 相同但 user_id/role 不同 → 视为新会话并告警
        （这是多轮场景最容易被忽略的安全漏洞：会话 ID 泄漏 = 上下文泄漏）
    """

    def __init__(
        self,
        ttl_seconds: float = 1800,
        max_sessions: int = 2000,
        max_turns: int = 20,
    ):
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self.max_turns = max_turns
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str, user_id: str, role: str) -> Session:
        now = time()
        with self._lock:
            session = self._sessions.get(session_id)

            if session is not None and session.is_expired(self.ttl_seconds, now):
                self._sessions.pop(session_id, None)
                session = None

            if session is not None and (
                session.user_id != user_id or session.role != role
            ):
                # 会话 ID 被复用到另一个身份上：丢弃旧上下文，防止串号
                logger.warning(
                    "会话 %s 的身份发生变化（%s/%s -> %s/%s），已重置上下文",
                    session_id,
                    session.user_id,
                    session.role,
                    user_id,
                    role,
                )
                self._sessions.pop(session_id, None)
                session = None

            if session is None:
                session = Session(
                    session_id=session_id, user_id=user_id, role=role
                )
                self._sessions[session_id] = session
                self._evict_if_needed()

            session.updated_at = now
            return session

    def append_turn(self, session: Session, turn: Turn) -> Session:
        with self._lock:
            session.turns.append(turn)
            if len(session.turns) > self.max_turns:
                # 只保留最近 N 轮，避免 Prompt 无界增长
                del session.turns[: len(session.turns) - self.max_turns]
            session.updated_at = time()
            self._sessions[session.session_id] = session
            return session

    def reset(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def purge_expired(self) -> int:
        now = time()
        with self._lock:
            expired = [
                key
                for key, session in self._sessions.items()
                if session.is_expired(self.ttl_seconds, now)
            ]
            for key in expired:
                self._sessions.pop(key, None)
        return len(expired)

    def _evict_if_needed(self) -> None:
        if len(self._sessions) <= self.max_sessions:
            return
        # 按最近更新时间淘汰最旧的 10%
        victims = sorted(
            self._sessions.items(), key=lambda item: item[1].updated_at
        )[: max(1, len(self._sessions) // 10)]
        for key, _session in victims:
            self._sessions.pop(key, None)

    # 便于测试与观测
    @property
    def size(self) -> int:
        return len(self._sessions)


# ==================== 词表 ====================
# 区域取值与数据保持一致；新增区域时改这里即可
REGION_VALUES = ("欧洲", "北美", "亚太")

# 数量词：同时支持阿拉伯数字与中文数字（"最近3个月" / "最近三个月" 都极常见）
_NUM = r"(?:\d+|[一二三四五六七八九十两]+)"

# 时间表达式（长的放前面，避免"三个月"被"月"抢先匹配）
_TIME_TOKEN = (
    rf"最近\s*{_NUM}\s*个?月"
    rf"|近\s*{_NUM}\s*个?月"
    rf"|前\s*{_NUM}\s*个?月"
    r"|上上个月|上个月|上月|这个月|本月"
    r"|上个季度|上季度|本季度|这个季度"
    r"|今年|去年|去年同[期季]"
    r"|20\d{2}\s*年\s*第?[一二三四1-4]\s*季度"
    r"|20\d{2}\s*年\s*Q[1-4]"
    r"|20\d{2}\s*年\s*\d{1,2}\s*月"
    r"|20\d{2}\s*年"
    r"|第?[一二三四1-4]\s*季度"
    r"|Q[1-4]"
)
_TIME_SPAN = re.compile(_TIME_TOKEN)

_FOLLOWUP_HEAD = r"^\s*(?:那|那么|换成|改成|再看|看下|看一下|如果)?\s*"
_FOLLOWUP_TAIL = r"\s*(?:呢|怎么样|如何|是多少|多少)?\s*[?？。!！]?\s*$"

_TIME_ONLY = re.compile(
    rf"{_FOLLOWUP_HEAD}(?P<value>{_TIME_TOKEN}){_FOLLOWUP_TAIL}", re.IGNORECASE
)
_REGION_ONLY = re.compile(
    rf"{_FOLLOWUP_HEAD}(?:只看|限定为?|换成|改成|只要)?\s*"
    rf"(?P<value>{'|'.join(REGION_VALUES)})\s*(?:市场|区域|大区|区)?"
    rf"{_FOLLOWUP_TAIL}"
)
_DIMENSION_ONLY = re.compile(
    rf"{_FOLLOWUP_HEAD}(?:按|以)\s*(?P<value>[\u4e00-\u9fa5A-Za-z]{{2,8}}?)"
    rf"\s*(?:来|看|分|统计|拆解|拆|分组|维度)?{_FOLLOWUP_TAIL}"
)

_DIMENSION_SPAN = re.compile(
    r"按\s*(?P<dim>[\u4e00-\u9fa5A-Za-z]{2,8}?)\s*(?:来|看|分|统计|拆解|拆|分组|维度)"
)


def _dimension_catalog() -> list[str]:
    """复用 Plan Agent 的维度白名单，保持单一事实来源。"""
    try:
        from plan_agent_execution.query_decomposer import AVAILABLE_DIMENSIONS

        return list(AVAILABLE_DIMENSIONS)
    except Exception:  # pragma: no cover - 导入失败不影响主链路
        return []


# ==================== 改写器 ====================
REWRITE_SYSTEM = (
    "你是 ChatBI 的多轮问题改写器，只输出合法 JSON，不要回答问题、不要生成 SQL。"
)

REWRITE_PROMPT = """请把用户的最新问题改写成一条**不依赖对话历史的独立问题**。

【对话历史】
{history}

【最新问题】
{question}

【改写要求】
1. 只做改写，不要回答问题、不要输出 SQL。
2. 继承历史中的指标（如收入/毛利）、维度（如产品线/区域）、过滤条件（如 order_status）。
3. 只替换用户明确改动的部分：
   - 换了时间 → 替换时间范围，其余保持
   - 换了区域/维度 → 替换该条件，其余保持
4. 不要编造历史中没有出现过的指标或维度。
5. 改写后必须是完整、可独立理解的一句中文问题，不超过 100 字。

【输出格式】严格输出 JSON：
{{"resolved_question": "改写后的问题", "changed": ["time"], "confidence": 0.9}}
其中 changed 可取：time / region / dimension / metric / filter / none
confidence 为 0~1 的浮点数，表示你对改写结果的把握。

只输出 JSON："""


class ContextResolver:
    """把多轮追问改写为独立问题。规则优先，LLM 兜底。"""

    def __init__(
        self,
        llm_client=None,
        history_window: int = 3,
        max_resolved_length: int = 200,
        enable_llm: bool = True,
    ):
        self.llm_client = llm_client
        self.history_window = history_window
        self.max_resolved_length = max_resolved_length
        self.enable_llm = enable_llm
        self._dimensions: list[str] | None = None

    # ---------- 主入口 ----------
    def resolve(self, session: Session, question: str) -> Resolution:
        last = session.last_turn()
        text = (question or "").strip()

        if last is None or not text:
            return Resolution(question=text, strategy="none")

        base = last.resolved_question or last.question

        # 1) 换时间："那上个月呢"
        match = _TIME_ONLY.match(text)
        if match:
            value = match.group("value").strip()
            if _TIME_SPAN.search(base):
                rewritten = _TIME_SPAN.sub(value, base, count=1)
                note = f"继承上一轮指标与维度，时间替换为「{value}」"
            else:
                rewritten = f"{base}，时间范围是{value}"
                note = f"继承上一轮问题并追加时间范围「{value}」"
            return Resolution(
                question=rewritten,
                strategy="rule_time",
                confidence=0.95,
                inherited=True,
                note=note,
            )

        # 2) 换区域："只看欧洲" / "那北美呢"
        match = _REGION_ONLY.match(text)
        if match:
            value = match.group("value").strip()
            origin = next((r for r in REGION_VALUES if r in base), None)
            if origin:
                rewritten = base.replace(origin, value, 1)
                note = f"继承上一轮问题，区域由「{origin}」替换为「{value}」"
            else:
                rewritten = f"{base}，限定区域为{value}"
                note = f"继承上一轮问题并限定区域「{value}」"
            return Resolution(
                question=rewritten,
                strategy="rule_region",
                confidence=0.95,
                inherited=True,
                note=note,
            )

        # 3) 换维度："按产品线看"
        match = _DIMENSION_ONLY.match(text)
        if match:
            value = match.group("value").strip()
            if self._is_valid_dimension(value):
                rewritten, origin = self._replace_dimension(base, value)
                if origin:
                    note = f"继承上一轮指标，维度由「{origin}」替换为「{value}」"
                else:
                    note = f"继承上一轮指标，追加维度「{value}」"
                return Resolution(
                    question=rewritten,
                    strategy="rule_dimension",
                    confidence=0.9,
                    inherited=True,
                    note=note,
                )

        # 4) LLM 兜底（开放指代/省略）
        if self.enable_llm and self.llm_client is not None:
            llm_resolution = self._llm_rewrite(session, text)
            if llm_resolution is not None:
                return llm_resolution

        # 5) 无法判断 → 当作独立问题（降级为单轮）
        return Resolution(question=text, strategy="passthrough")

    # ---------- 规则辅助 ----------
    def _is_valid_dimension(self, value: str) -> bool:
        if self._dimensions is None:
            self._dimensions = _dimension_catalog()
        if not self._dimensions:
            # 目录不可用时放宽：只要不是明显的时间/区域词就接受
            return value not in REGION_VALUES
        return any(value == dim or value in dim for dim in self._dimensions)

    def _find_dimension_in(self, base: str) -> str | None:
        """在上一轮问题里找出已存在的维度（最长匹配优先，避免'产品'吃掉'产品线'）。"""
        if self._dimensions is None:
            self._dimensions = _dimension_catalog()
        for dim in sorted(self._dimensions, key=len, reverse=True):
            if dim in base:
                return dim
        return None

    def _replace_dimension(self, base: str, value: str) -> tuple[str, str | None]:
        """把上一轮问题里的旧维度替换为新维度。

        两条路径：
          1. 有显式的"按 X 看/分/统计"结构 → 直接替换该片段
          2. 没有显式结构但提到了某个已知维度（如"各产品线的毛利率"）→ 替换该词
        都找不到时追加"按 X 分组"，而不是丢弃上一轮的指标。
        """
        explicit = _DIMENSION_SPAN.search(base)
        if explicit:
            return _DIMENSION_SPAN.sub(f"按{value}分组", base, count=1), explicit.group("dim")

        found = self._find_dimension_in(base)
        if found:
            return base.replace(found, value, 1), found

        return f"{base}，按{value}分组", None

    # ---------- LLM 兜底 ----------
    def _build_history(self, session: Session) -> str:
        blocks: list[str] = []
        for turn in session.recent(self.history_window):
            lines = [
                f"轮次 {turn.index}",
                f"  用户：{turn.question}",
                f"  理解：{turn.resolved_question}",
            ]
            if turn.sql:
                sql = turn.sql if len(turn.sql) <= 220 else turn.sql[:220] + " ..."
                lines.append(f"  SQL：{sql}")
            if turn.columns:
                lines.append(
                    f"  结果：{turn.row_count if turn.row_count is not None else '?'} 行，"
                    f"列为 {turn.columns}"
                )
            if turn.error_type:
                lines.append(f"  状态：上一轮执行失败（{turn.error_type}）")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def _llm_rewrite(self, session: Session, question: str) -> Resolution | None:
        history = self._build_history(session)
        if not history:
            return None

        prompt = REWRITE_PROMPT.format(history=history, question=question)
        try:
            raw = self.llm_client.generate_text(REWRITE_SYSTEM, prompt)
        except Exception as exc:
            logger.warning("多轮改写 LLM 调用失败，降级为单轮：%s", exc)
            return None

        parsed = _parse_json(raw)
        if not parsed:
            logger.warning("多轮改写返回无法解析，降级为单轮：%r", (raw or "")[:120])
            return None

        resolved = str(parsed.get("resolved_question", "")).strip()
        rejection = self._validate_rewrite(resolved)
        if rejection:
            logger.warning("多轮改写结果被拒绝（%s），降级为单轮", rejection)
            return None

        try:
            confidence = float(parsed.get("confidence", 0.6))
        except (TypeError, ValueError):
            confidence = 0.6
        confidence = min(max(confidence, 0.0), 1.0)

        # 改写后与原文相同，说明模型认为不需要改写
        if resolved == question:
            return Resolution(question=question, strategy="passthrough", confidence=1.0)

        changed = parsed.get("changed") or []
        return Resolution(
            question=resolved,
            strategy="llm_rewrite",
            confidence=confidence,
            inherited=True,
            note=f"LLM 改写，变更维度：{changed}",
        )

    def _validate_rewrite(self, resolved: str) -> str:
        """返回拒绝原因；空字符串表示通过。"""
        if not resolved:
            return "改写结果为空"
        if len(resolved) > self.max_resolved_length:
            return f"改写结果过长（{len(resolved)} 字）"
        lowered = resolved.lower()
        if lowered.lstrip().startswith(("select", "with", "```", "{")):
            return "改写结果疑似 SQL 或 JSON"
        if "select " in lowered and " from " in lowered:
            return "改写结果疑似 SQL"
        return ""


def _parse_json(raw: str) -> dict | None:
    """从模型输出中稳健地取出 JSON 对象。"""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        loaded = json.loads(text)
        return loaded if isinstance(loaded, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            loaded = json.loads(match.group(0))
            return loaded if isinstance(loaded, dict) else None
        except json.JSONDecodeError:
            return None
    return None


# ==================== 默认实例 ====================
def build_default_store() -> InMemoryConversationStore:
    cfg = APP_CONFIG.get("conversation", {})
    return InMemoryConversationStore(
        ttl_seconds=float(cfg.get("ttl_seconds", 1800)),
        max_sessions=int(cfg.get("max_sessions", 2000)),
        max_turns=int(cfg.get("max_turns", 20)),
    )
