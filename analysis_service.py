"""分析链路执行器：把 Plan-and-Execute Agent 包装成与单轮链路同构的返回结构。

**为什么需要这一层**

`PlanAndExecuteAgent` 的原始返回是 `{decomposition, plan, step_results, summary, report}`
（面向内部调试），而单轮链路返回的是
`{success, sql, columns, results, formatted, metadata}`（面向 API）。

上层（API / 前端）希望用同一套 envelope 处理两种结果，所以这里做一次归一化：
统一 `success` / `kind` / `formatted` / `metadata`，把报告放进 `report`、步骤放进 `steps`。

**权限必须透传**

`StepExecutor` 默认调用 `ChatBISystem.run(user_question=...)` 而不传 `user`——
在 dev 环境下这会回落到 demo 管理员，意味着一个 sales 用户只要问一句
"最近三个月利润为什么下降"就能拿到全量数据。所以这里显式把 `user` 注入
`chatbi_run_options`，让**每一个子步骤**都走同一套安全校验。
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from config import APP_CONFIG
from security import UserContext
from text2SQL.chatbi_system import ChatBISystem

logger = logging.getLogger("chatbi.analysis")


def _step_brief(step_result: dict[str, Any]) -> dict[str, Any]:
    """只保留可展示的步骤摘要，不把整张结果集塞进 API 响应。"""
    rows = step_result.get("rows") or []
    return {
        "step_id": step_result.get("step_id"),
        "step_name": step_result.get("step_name"),
        "status": step_result.get("status"),
        "success": step_result.get("success"),
        "attempts": step_result.get("attempts"),
        "sql": step_result.get("sql"),
        "columns": step_result.get("columns") or [],
        "row_count": len(rows),
        "preview": rows[:5],
        "error": step_result.get("error"),
    }


def run_analysis(
    question: str,
    user: UserContext,
    system: ChatBISystem | None = None,
    max_steps: int | None = None,
    agent: Any | None = None,
) -> dict[str, Any]:
    """执行多步分析，返回与单轮链路同构的结果。

    Args:
        agent: 可注入的 Agent（测试用）。为 None 时按 user 构造真实的 Plan-and-Execute。
    """
    from plan_agent_execution.plan_and_execute_agent import PlanAndExecuteAgent
    from plan_agent_execution.step_executor import StepExecutor

    cfg = APP_CONFIG.get("router", {})
    budget = int(max_steps or cfg.get("analysis_max_steps", 5))
    started = perf_counter()

    shared_system = system or ChatBISystem()

    # 关键：把 user 注入每个子步骤的调用参数，保证行级权限与脱敏逐步骤生效
    run_options: dict[str, Any] = {
        "user": user,
        "use_schema_linking": True,
        "use_indicator_knowledge": True,
        "use_indicator_rag": True,
        # 子问题是机器生成的，不应写进用户的多轮历史
        "enable_multiturn": False,
    }

    try:
        if agent is None:
            executor = StepExecutor(
                chatbi_system=shared_system,
                chatbi_run_options=run_options,
                max_retries=1,
                failure_policy="skip",
            )
            agent = PlanAndExecuteAgent(executor=executor)
        raw = agent.run(question, max_steps=budget)
    except Exception as exc:
        logger.exception("分析链路执行失败")
        return {
            "success": False,
            "kind": "report",
            "error_type": "analysis_failed",
            "error": f"分析链路执行失败：{exc}",
            "question": question,
            "formatted": "",
            "report": None,
            "steps": [],
            "metadata": {"duration_ms": round((perf_counter() - started) * 1000, 2)},
        }

    step_results = raw.get("step_results") or []
    report = raw.get("report") or {}
    summary = raw.get("summary") or {}
    succeeded = [step for step in step_results if step.get("success")]
    duration_ms = round((perf_counter() - started) * 1000, 2)

    if not succeeded:
        first_error = next(
            (step.get("error") for step in step_results if step.get("error")), "所有步骤均失败"
        )
        return {
            "success": False,
            "kind": "report",
            "error_type": "analysis_steps_failed",
            "error": str(first_error),
            "question": question,
            "formatted": "",
            "report": report or None,
            "steps": [_step_brief(step) for step in step_results],
            "metadata": {
                "duration_ms": duration_ms,
                "step_total": len(step_results),
                "step_succeeded": 0,
                "analysis_goal": (raw.get("plan") or {}).get("analysis_goal"),
            },
        }

    markdown = report.get("markdown") or _fallback_markdown(question, report, summary)

    return {
        "success": True,
        "kind": "report",
        "question": question,
        "sql": "",
        "columns": [],
        "results": [],
        "formatted": markdown,
        "report": {
            "title": report.get("title", "分析报告"),
            "executive_summary": report.get("executive_summary", ""),
            "key_findings": report.get("key_findings", []),
            "root_causes": report.get("root_causes", []),
            "trend_judgment": report.get("trend_judgment", ""),
            "action_suggestions": report.get("action_suggestions", []),
            "markdown": markdown,
        },
        "steps": [_step_brief(step) for step in step_results],
        "metadata": {
            "duration_ms": duration_ms,
            "step_total": len(step_results),
            "step_succeeded": len(succeeded),
            "analysis_goal": (raw.get("plan") or {}).get("analysis_goal"),
            "question_type": (raw.get("plan") or {}).get("question_type"),
            "summary_text": summary.get("summary_text"),
            "max_steps": budget,
        },
    }


def _fallback_markdown(question: str, report: dict, summary: dict) -> str:
    """报告缺少 markdown 时的兜底渲染。"""
    lines = [f"# {report.get('title') or question}", ""]
    if report.get("executive_summary"):
        lines += ["## 执行摘要", report["executive_summary"], ""]
    findings = report.get("key_findings") or []
    if findings:
        lines += ["## 关键发现", *[f"- {item}" for item in findings], ""]
    causes = report.get("root_causes") or []
    if causes:
        lines += ["## 归因分析", *[f"- {item}" for item in causes], ""]
    if report.get("trend_judgment"):
        lines += ["## 趋势判断", report["trend_judgment"], ""]
    suggestions = report.get("action_suggestions") or []
    if suggestions:
        lines += ["## 行动建议", *[f"- {item}" for item in suggestions], ""]
    if len(lines) <= 2 and summary.get("summary_text"):
        lines.append(summary["summary_text"])
    return "\n".join(lines).strip()
