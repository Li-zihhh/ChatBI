"""
ResultSummarizer 模块

将多步 StepExecutionResult 收敛为统一的 ExecutionSummary。
负责分类统计（成功/失败/跳过）、提取各步骤关键发现、生成汇总描述。
"""

from __future__ import annotations

import json

from .middle_state import (
    ExecutionPlan,
    ExecutionSummary,
    StepExecutionResult,
)


class ResultSummarizer:
    """将多步执行结果收敛为统一摘要——提取各步骤关键信息，统计成功/失败/跳过数量。"""

    def summarize(
        self,
        original_question: str,
        plan: ExecutionPlan,
        step_results: list[StepExecutionResult],
    ) -> ExecutionSummary:
        """遍历所有步骤的执行结果，分类统计（成功/失败/跳过），提取关键发现，生成摘要文本。"""
        completed = [result for result in step_results if result.success]
        failed = [result for result in step_results if result.status == "failed"]
        skipped = [result for result in step_results if result.status == "skipped"]

        findings = [
            f"{result.step_name}：{self._pick_result_brief(result)}"
            for result in step_results
        ]

        if failed:
            summary_text = (
                f"已完成 {len(completed)} 个步骤，"
                f"失败 {len(failed)} 个步骤，"
                f"跳过 {len(skipped)} 个步骤。"
                "当前链路已经暴露出真实执行问题，"
                "需要先修复失败步骤，再继续扩展中间结果管理与总结能力。"
            )
        else:
            summary_text = (
                f"已完成 {len(completed)} 个步骤，"
                f"失败 {len(failed)} 个步骤，"
                f"跳过 {len(skipped)} 个步骤。"
                "当前结果已经可以支撑后续的中间结果管理与最终报告生成。"
            )

        return ExecutionSummary(
            original_question=original_question,
            analysis_goal=plan.analysis_goal,
            completed_steps=len(completed),
            failed_steps=len(failed),
            skipped_steps=len(skipped),
            key_findings=findings,
            summary_text=summary_text,
        )

    @staticmethod
    def _pick_result_brief(result: StepExecutionResult) -> str:
        """根据执行结果类型，选择最合适的摘要描述。"""
        if not result.success:
            return f"执行失败，错误信息：{result.error or '未知错误'}"
        if result.formatted.strip() and not ResultSummarizer._looks_like_table(result.formatted):
            for line in result.formatted.strip().splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                if set(stripped) <= {"-", "+"}:
                    continue
                return stripped[:120]
        if result.rows:
            first_row = result.rows[0]
            parts = [
                f"{key}={value}"
                for key, value in first_row.items()
                if value is not None
            ]
            return "，".join(parts[:4])[:120]
        if result.formatted.strip():
            for line in result.formatted.strip().splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                if set(stripped) <= {"-", "+"}:
                    continue
                return stripped[:120]
        if result.rows:
            return json.dumps(result.rows[0], ensure_ascii=False)
        if result.result_reference:
            return f"结果已写入 {result.result_reference}"
        return "步骤执行成功，但当前无返回行。"

    @staticmethod
    def _looks_like_table(formatted: str) -> bool:
        """判断字符串是否像表格格式。"""
        text = formatted.strip()
        return "+" in text and "|" in text