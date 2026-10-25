"""查询路由：判断一个问题该走 Text2SQL 单轮链路，还是走 Plan-and-Execute 分析链路。

**为什么需要路由**

两条链路的代价差一个数量级：

| | Text2SQL 单轮 | Plan-and-Execute |
|---|---|---|
| LLM 调用 | 1~3 次 | 拆解 1 + 每步 1~3 + 汇总 1 + 报告 1 ≈ 6~12 次 |
| 延迟 | 1~3 s | 15~40 s |
| 输出形态 | 一张表 | 多步结果 + 结构化分析报告 |
| 适用问题 | "欧洲上个月收入多少" | "最近三个月利润为什么下降" |

把所有问题都送进分析链路是灾难（延迟与成本 ×5~10）；
把所有问题都当单轮则回答不了归因类问题。所以必须有一个显式的分流决策。

**设计原则**

1. **规则优先、LLM 兜底**（与 Schema Linking、多轮改写一致的哲学）。
   高置信规则直接判定，零成本；只有信号冲突/缺失时才请模型判断。
2. **宁可走便宜链路**。默认落到单轮，因为单轮失败还能升级；
   反过来"误判为复杂"会让用户白等 30 秒。
3. **可升级（escalation）**。单轮结果失败或为空，且原问题带分析意图时，
   自动升级到分析链路 —— 用真实结果替代猜测。
4. **决策可解释**。返回命中的信号与理由，便于线上统计和调参。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from config import APP_CONFIG

logger = logging.getLogger("chatbi.router")

Path = Literal["simple", "analysis"]
Mode = Literal["auto", "simple", "analysis"]


# ==================== 信号定义 ====================
# (正则, 权重, 标签)。正权重指向"复杂分析"，负权重指向"简单查询"。
_ANALYSIS_SIGNALS: tuple[tuple[str, int, str], ...] = (
    (r"为什么|为何|怎么会|什么原因|原因是|归因|导致|影响因素|因为什么", 3, "causal"),
    (r"诊断|问题出在哪|问题在哪|怎么提升|如何提升|如何改善|怎么改善|改进建议|优化建议", 3, "diagnosis"),
    (r"拆解|拆分|分解|构成|结构占比|收入构成|成本构成", 2, "decompose"),
    (r"趋势|走势|变化情况|波动|同比|环比|相比|对比|比去年|比上月", 2, "trend"),
    (r"贡献度|贡献|拉动了|拖累|影响有多大", 2, "attribution"),
    (r"分析|评估|复盘|洞察|深度了解|全面了解", 2, "analysis"),
    (r"分别|各自|以及|同时|顺便|再帮我", 1, "multi"),
)

_SIMPLE_SIGNALS: tuple[tuple[str, int, str], ...] = (
    (r"是多少|有多少|多少|几个|几笔|几单|几家|总数|合计|总计|一共", -2, "metric_lookup"),
    (r"列出|列一下|查一下|查询|看一下|看看|给我看|有哪些|有没有|是否|明细", -2, "lookup"),
    (r"排名|top\s*\d+|前\s*[一二三四五六七八九十\d]+\s*[个名]|最高|最低|最多|最少", -1, "ranking"),
)

# 强指令：用户明确要"分析报告"，直接判定，不看分数
_EXPLICIT_ANALYSIS = re.compile(
    r"深度分析|详细分析|全面分析|帮我分析一下|做个分析|出个报告|写个报告|生成报告"
)

_SCORE_ANALYSIS_THRESHOLD = 3
_SCORE_SIMPLE_THRESHOLD = -1
_LONG_QUESTION_CHARS = 40


@dataclass(slots=True)
class RouteDecision:
    """路由结果。带上依据，便于线上排查"为什么走了这条链路"。"""

    path: Path
    confidence: float
    reason: str
    signals: list[str] = field(default_factory=list)
    score: int = 0
    source: str = "rule"          # rule / llm / override / default / escalation
    escalate_candidate: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "signals": list(self.signals),
            "score": self.score,
            "source": self.source,
            "escalate_candidate": self.escalate_candidate,
        }


CLASSIFY_SYSTEM = "你是 ChatBI 的问题路由器，只输出合法 JSON，不要解释。"

CLASSIFY_PROMPT = """判断用户问题属于哪一类，并输出 JSON。

【simple】可以用**一条 SQL 查询**直接回答的取数问题。特征是：明确的指标 + 维度 + 时间范围。
  例：上个月欧洲的销售额是多少 / 各产品线的毛利率 / 销售费用最高的三个部门

【analysis】一条 SQL 回答不了，需要**多步查询 + 归因/对比/拆解**才能回答的分析问题。
  例：最近三个月利润为什么下降 / 收入和成本各自对毛利的影响有多大 / 帮我全面评估一下储能业务

【用户问题】
{question}

【输出格式】严格输出 JSON：
{{"path": "simple 或 analysis", "analysis_type": "trend / diagnosis / dimension / none", "confidence": 0.0, "reason": "一句话理由"}}

只输出 JSON："""


class QueryRouter:
    """三层路由：显式模式 → 规则打分 → LLM 分类。"""

    def __init__(
        self,
        llm_client=None,
        enabled: bool | None = None,
        llm_classify: bool | None = None,
    ):
        cfg = APP_CONFIG.get("router", {})
        self.enabled = bool(cfg.get("enabled", True)) if enabled is None else enabled
        self.llm_classify_enabled = (
            bool(cfg.get("llm_classify", True)) if llm_classify is None else llm_classify
        )
        self.llm_client = llm_client

    # ==================== 主入口 ====================
    def route(self, question: str, mode: Mode = "auto") -> RouteDecision:
        text = (question or "").strip()

        # 1) 用户显式指定，最高的优先级（产品上通常是个"深度分析"开关）
        if mode == "analysis":
            return RouteDecision(
                path="analysis", confidence=1.0, reason="用户指定深度分析", source="override"
            )
        if mode == "simple":
            return RouteDecision(
                path="simple", confidence=1.0, reason="用户指定快速查询", source="override"
            )

        # 2) 路由关闭 / 空问题 → 走便宜链路
        if not self.enabled or not text:
            return RouteDecision(
                path="simple", confidence=1.0, reason="路由未启用", source="default"
            )

        # 3) 强指令直接判定
        explicit = _EXPLICIT_ANALYSIS.search(text)
        if explicit:
            return RouteDecision(
                path="analysis",
                confidence=0.9,
                reason=f"命中显式分析指令「{explicit.group(0)}」",
                signals=["explicit_analysis"],
                score=10,
                source="rule",
            )

        # 4) 规则打分
        score, signals = self._score(text)
        if score >= _SCORE_ANALYSIS_THRESHOLD:
            return RouteDecision(
                path="analysis",
                confidence=min(0.95, 0.6 + 0.08 * score),
                reason=f"分析类信号得分 {score}（{','.join(signals)}）",
                signals=signals,
                score=score,
                source="rule",
            )
        if score <= _SCORE_SIMPLE_THRESHOLD:
            return RouteDecision(
                path="simple",
                confidence=0.9,
                reason=f"查询类信号得分 {score}（{','.join(signals)}）",
                signals=signals,
                score=score,
                source="rule",
            )

        # 5) 分数落在灰区 → 请模型判断；模型不可用则保守走便宜链路 + 标记可升级
        if self.llm_classify_enabled and self.llm_client is not None:
            llm_decision = self._llm_classify(text, score, signals)
            if llm_decision is not None:
                return llm_decision

        return RouteDecision(
            path="simple",
            confidence=0.5,
            reason=(
                "无明确信号，默认走快速查询"
                if score == 0
                else f"分析信号偏弱（{score}），先走快速查询，失败可升级"
            ),
            signals=signals,
            score=score,
            source="default",
            escalate_candidate=score > 0,
        )

    # ==================== 规则层 ====================
    @staticmethod
    def _score(text: str) -> tuple[int, list[str]]:
        score = 0
        signals: list[str] = []

        for pattern, weight, label in _ANALYSIS_SIGNALS:
            if re.search(pattern, text):
                score += weight
                signals.append(label)

        for pattern, weight, label in _SIMPLE_SIGNALS:
            if re.search(pattern, text):
                score += weight
                signals.append(label)

        # 多个问号 = 多个子问题
        if text.count("？") + text.count("?") >= 2:
            score += 2
            signals.append("multi_question")

        # 长问题通常承载更多约束（只在没有任何查询类信号时加权，避免把
        # "查询 x、y、z、w 这几个指标分别是多少" 误判成分析）
        if len(text) >= _LONG_QUESTION_CHARS and "lookup" not in signals:
            score += 1
            signals.append("long_question")

        return score, signals

    # ==================== LLM 层 ====================
    def _llm_classify(
        self, question: str, score: int, signals: list[str]
    ) -> RouteDecision | None:
        try:
            raw = self.llm_client.generate_text(
                CLASSIFY_SYSTEM, CLASSIFY_PROMPT.format(question=question)
            )
        except Exception as exc:
            logger.warning("路由 LLM 分类失败，回退到规则结论：%s", exc)
            return None

        parsed = _parse_json(raw)
        if not parsed:
            logger.warning("路由分类返回无法解析：%r", (raw or "")[:120])
            return None

        path = str(parsed.get("path", "")).strip().lower()
        if path not in {"simple", "analysis"}:
            return None

        try:
            confidence = float(parsed.get("confidence", 0.6))
        except (TypeError, ValueError):
            confidence = 0.6
        confidence = min(max(confidence, 0.0), 1.0)

        reason = str(parsed.get("reason", "")).strip() or "LLM 分类"
        analysis_type = str(parsed.get("analysis_type", "none")).strip() or "none"

        # 置信度不足时仍落到便宜链路，并允许升级
        if confidence < 0.55:
            return RouteDecision(
                path="simple",
                confidence=confidence,
                reason=f"LLM 置信度不足（{confidence:.2f}），先走快速查询：{reason}",
                signals=signals,
                score=score,
                source="llm",
                escalate_candidate=True,
            )

        return RouteDecision(
            path=path,  # type: ignore[arg-type]
            confidence=confidence,
            reason=f"LLM 判定为 {path}（{analysis_type}）：{reason}",
            signals=signals,
            score=score,
            source="llm",
            escalate_candidate=(path == "simple" and score > 0),
        )

    # ==================== 升级策略 ====================
    @staticmethod
    def should_escalate(decision: RouteDecision, result: dict[str, Any]) -> bool:
        """单轮结果是否值得升级到分析链路。

        只在"本就是灰区、被保守地判为 simple"时考虑，避免把正常空结果
        （如"查询已取消的订单"确实没有数据）也送去跑 30 秒分析。

        权限/校验类失败绝不升级：那是安全层的拒绝，换个链路重试等于绕过。
        """
        if decision.path != "simple" or not decision.escalate_candidate:
            return False
        if result.get("error_type") in {
            "security",
            "validation",
            "missing_identity",
            "database_permission_denied",
        }:
            return False
        if not result.get("success"):
            return True
        rows = result.get("results") or []
        return len(rows) == 0


def _parse_json(raw: str) -> dict | None:
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
