"""
PlanGenerator 模块

负责把 QueryDecomposer 产出的 DecompositionPlan 转换为 StepExecutor 可消费的 ExecutionPlan。
核心转换：task_id → step_id 映射、depends_on 引用转换、子问题文本拼接。
"""

from __future__ import annotations

from typing import Any

from .middle_state import (
    DecomposedTask,
    DecompositionPlan,
    ExecutionPlan,
    PlanStep,
)


class PlanGenerator:
    """根据拆解结果生成可执行计划——衔接 query_decomposer 输出 → executor 输入。"""

    def build_plan(
        self,
        original_question: str,
        decomposition: dict[str, Any] | DecompositionPlan,
    ) -> ExecutionPlan:
        """将 DecompositionPlan 中的 subtasks 转换为 ExecutionPlan 中的 steps。
        核心转换：task_id → step_id 映射、depends_on 引用转换、子问题文本拼接。"""
        plan = self._ensure_decomposition_plan(decomposition)
        task_to_step = {
            task.task_id: f"step_{index}"
            for index, task in enumerate(plan.subtasks, start=1)
        }

        steps: list[PlanStep] = []
        for index, task in enumerate(plan.subtasks, start=1):
            steps.append(
                PlanStep(
                    step_id=f"step_{index}",
                    task_id=task.task_id,
                    step_name=task.task_name,
                    task_type=task.task_type,
                    question=self._build_step_question(task),
                    description=task.description,
                    depends_on=[task_to_step[dep] for dep in task.depends_on],
                    metrics=task.metrics,
                    dimensions=task.dimensions,
                    expected_output=self._build_expected_output(task),
                )
            )

        return ExecutionPlan(
            original_question=original_question,
            question_type=plan.question_type,
            analysis_goal=plan.analysis_goal,
            steps=steps,
        )

    @staticmethod
    def _ensure_decomposition_plan(
        decomposition: dict[str, Any] | DecompositionPlan,
    ) -> DecompositionPlan:
        """兼容 dict / Pydantic 两种输入形式，统一转为 DecompositionPlan 对象。"""
        if isinstance(decomposition, DecompositionPlan):
            return decomposition
        return DecompositionPlan.model_validate(decomposition)

    @staticmethod
    def _build_step_question(task: DecomposedTask) -> str:
        """将子任务的名称、描述、指标、维度拼接成一条完整的自然语言问题，
        用于传递给 ChatBISystem.run() 执行单次 Text2SQL 查询。"""
        lines = [f"请执行子任务：{task.task_name}。"]
        if task.description:
            lines.append(f"任务说明：{task.description}")
        if task.metrics:
            lines.append(f"关注指标：{'、'.join(task.metrics)}")
        if task.dimensions:
            lines.append(f"分析维度：{'、'.join(task.dimensions)}")
        lines.append("执行约束：本步骤只回答当前子任务，优先生成一条结构清晰、易执行的 SQL。")
        lines.append("请优先返回支撑后续分析的查询结果，不要直接跳到最终归因结论。")
        return "\n".join(lines)

    @staticmethod
    def _build_expected_output(task: DecomposedTask) -> str:
        """根据子任务的指标和维度，生成对执行结果的预期描述文本。"""
        focus_parts: list[str] = []
        if task.metrics:
            focus_parts.append(f"指标：{'、'.join(task.metrics)}")
        if task.dimensions:
            focus_parts.append(f"维度：{'、'.join(task.dimensions)}")
        if not focus_parts:
            focus_parts.append("可被后续步骤复用的结构化结果")
        return "；".join(focus_parts)