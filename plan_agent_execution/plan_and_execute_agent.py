"""
PlanAndExecuteAgent 总入口模块

串联 QueryDecomposer → PlanGenerator → StepExecutor → ResultSummarizer → ReportGenerator 全链路。
"""

from __future__ import annotations

from typing import Any

from .middle_state import print_progress
from .plan_generator import PlanGenerator
from .query_decomposer import QueryDecomposer
from .report_generator import ReportGenerator
from .result_summarizer import ResultSummarizer
from .step_executor import StepExecutor


class PlanAndExecuteAgent:
    """Plan-and-Execute Agent 总入口——串联 decomposer → planner → executor → summarizer → report_generator 全链路。

    数据流向：
    QueryDecomposer.decompose() → dict
        → PlanGenerator.build_plan() → ExecutionPlan
        → StepExecutor.execute_plan() → list[StepExecutionResult]
        → ResultSummarizer.summarize() → ExecutionSummary
        → ReportGenerator.generate() → AnalysisReport
    """

    def __init__(
        self,
        decomposer: QueryDecomposer | None = None,
        planner: PlanGenerator | None = None,
        executor: StepExecutor | None = None,
        summarizer: ResultSummarizer | None = None,
        report_generator: ReportGenerator | None = None,
    ):
        self.decomposer = decomposer or QueryDecomposer()
        self.planner = planner or PlanGenerator()
        self.executor = executor or StepExecutor()
        self.summarizer = summarizer or ResultSummarizer()
        self.report_generator = report_generator or ReportGenerator()

    def run(
        self,
        user_question: str,
        decomposition_override: dict[str, Any] | None = None,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """Agent 主入口：拆解 → 生成计划 → 执行各步骤 → 汇总 → 生成报告。
        支持 decomposition_override 跳过拆解阶段，直接使用外部传入的拆解结果。"""
        if decomposition_override is not None:
            print_progress("使用传入的拆解结果，跳过任务拆解。")
            decomposition = decomposition_override
        else:
            print_progress("开始任务拆解。")
            decomposition = self.decomposer.decompose(user_question)
            print_progress(
                f"任务拆解完成，共 {len(decomposition.get('subtasks', []))} 个子任务。"
            )

        print_progress("开始生成执行计划。")
        plan = self.planner.build_plan(user_question, decomposition)
        print_progress(f"执行计划生成完成，共 {len(plan.steps)} 个步骤。")
        step_results = self.executor.execute_plan(plan, max_steps=max_steps)
        print_progress("开始生成执行摘要。")
        summary = self.summarizer.summarize(user_question, plan, step_results)
        print_progress("执行摘要生成完成。")
        print_progress("开始生成分析报告。")
        report = self.report_generator.generate(
            original_question=user_question,
            analysis_goal=plan.analysis_goal,
            step_results=[result.model_dump() for result in step_results],
            summary=summary.model_dump(),
        )
        print_progress("分析报告生成完成。")

        return {
            "original_question": user_question,
            "decomposition": decomposition,
            "plan": plan.model_dump(),
            "step_results": [result.model_dump() for result in step_results],
            "summary": summary.model_dump(),
            "report": report.model_dump(),
        }