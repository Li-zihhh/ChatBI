"""
StepExecutor 模块

Plan-and-Execute Agent 的核心执行引擎。
负责依赖检查 → 上下文注入 → 调用 ChatBISystem → 存储中间结果，
按 plan.steps 顺序逐一执行，维护 results_by_step 字典作为状态枢纽。
"""

from __future__ import annotations

import json
from typing import Any, Callable, Literal

from text2SQL.chatbi_system import ChatBISystem
from .middle_state import (
    ExecutionPlan,
    IntermediateResultStore,
    MemoryResultStore,
    PlanStep,
    StepExecutionResult,
    TempTableResultStore,
    print_progress,
)


class StepExecutor:
    """按执行计划顺序执行每个步骤——核心执行引擎，负责依赖检查 → 上下文注入 → 调用 ChatBISystem → 存储中间结果。"""

    def __init__(
        self,
        step_runner: Callable[[str], dict[str, Any]] | None = None,
        chatbi_system: ChatBISystem | None = None,
        chatbi_run_options: dict[str, Any] | None = None,
        max_retries: int = 0,
        failure_policy: Literal["abort", "skip"] = "abort",
        storage_backend: Literal["memory", "temp_table"] = "memory",
        result_store: IntermediateResultStore | None = None,
        storage_connection_factory: Callable[[], Any] | None = None,
    ):
        self.system = chatbi_system
        self.chatbi_run_options = chatbi_run_options or {
            "use_schema_linking": True,
            "use_indicator_rag": True,
            "use_indicator_knowledge": True,
        }
        self.step_runner = step_runner or self._run_with_chatbi
        self.max_retries = max_retries
        self.failure_policy = failure_policy
        self.storage_backend = storage_backend
        self.result_store = result_store or self._build_result_store(
            storage_backend=storage_backend,
            connection_factory=storage_connection_factory,
        )
        self.memory_store = (
            self.result_store.data
            if isinstance(self.result_store, MemoryResultStore)
            else {}
        )

    def execute_plan(
        self,
        plan: ExecutionPlan,
        max_steps: int | None = None,
    ) -> list[StepExecutionResult]:
        """遍历 plan.steps，按依赖顺序逐一执行：检查失败策略 → 检查依赖是否满足 → 注入前置上下文 → 调用 ChatBISystem.run() → 存储中间结果。"""
        results: list[StepExecutionResult] = []  # 存储所有步骤的执行结果
        results_by_step: dict[str, StepExecutionResult] = {}     # 存储每个步骤的执行结果
        steps_to_run = plan.steps[:max_steps] if max_steps is not None else plan.steps
        abort_triggered = False
        total_steps = len(steps_to_run)

        print_progress(f"开始执行计划，共 {total_steps} 个步骤。")

        for index, step in enumerate(steps_to_run, start=1):
            if abort_triggered:
                print_progress(
                    f"{step.step_id} 已跳过：前序步骤失败，当前执行策略为 abort。"
                )
                skipped = self._build_skipped_result(
                    step=step,
                    error="前序步骤失败，当前执行策略为 abort，后续步骤停止执行。",
                )
                results.append(skipped)
                results_by_step[step.step_id] = skipped
                continue

            # 检查依赖是否满足
            if self._has_failed_dependency(step.depends_on, results_by_step):
                print_progress(f"{step.step_id} 已跳过：依赖步骤失败。")
                skipped = self._build_skipped_result(
                    step=step,
                    error="依赖步骤失败，当前步骤已跳过。",
                )
                results.append(skipped)
                results_by_step[step.step_id] = skipped
                continue

            print_progress(
                f"开始执行 {step.step_id}（{index}/{total_steps}）：{step.step_name}"
            )

            # 构建依赖上下文
            dependency_context = self._build_dependency_context(
                step.depends_on,
                results_by_step,
            )

            # 组合问题，注入依赖上下文
            composed_question = self._compose_question(step.question, dependency_context)
            normalized = self._execute_with_retry(
                step=step,
                question=composed_question,
                context_used=dependency_context,
            )

            # 存储中间结果
            if normalized.success:
                normalized.result_reference = self._store_intermediate_result(
                    step.step_id,
                    normalized.columns,
                    normalized.rows,
                )
                # 记录存储后的结果引用
                normalized.storage_backend = self.storage_backend
                print_progress(
                    f"{step.step_id} 执行完成：{step.step_name}"
                )
            elif self.failure_policy == "abort":
                # 记录失败信息
                print_progress(
                    f"{step.step_id} 执行失败：{normalized.error or '未知错误'}"
                )
                abort_triggered = True
            else:
                # 记录失败信息
                print_progress(
                    f"{step.step_id} 执行失败：{normalized.error or '未知错误'}"
                )

            results.append(normalized)
            results_by_step[step.step_id] = normalized

        completed_steps = sum(1 for result in results if result.status == "completed")
        failed_steps = sum(1 for result in results if result.status == "failed")
        skipped_steps = sum(1 for result in results if result.status == "skipped")
        print_progress(
            f"执行计划结束：completed={completed_steps}, failed={failed_steps}, skipped={skipped_steps}"
        )
        return results

    def _run_with_chatbi(self, question: str) -> dict[str, Any]:
        """默认的单步执行器：调用 ChatBISystem.run() 完成 Text2SQL → 执行 → 格式化全链路。"""
        if self.system is None:
            self.system = ChatBISystem()

        return self.system.run(
            user_question=question,
            **self.chatbi_run_options,
        )

    def _execute_with_retry(
        self,
        step: PlanStep,
        question: str,
        context_used: str,
    ) -> StepExecutionResult:
        """带重试的单步执行：max_retries+1 次尝试，任意一次成功即返回。"""
        last_result: StepExecutionResult | None = None

        for attempt in range(1, self.max_retries + 2):
            if attempt > 1:
                print_progress(
                    f"{step.step_id} 开始第 {attempt} 次尝试：{step.step_name}"
                )
            raw_result = self.step_runner(question)
            normalized = self._normalize_result(
                step=step,
                question=question,
                context_used=context_used,
                raw_result=raw_result,
                attempts=attempt,
                storage_backend=self.storage_backend,
            )
            if normalized.success:
                return normalized
            last_result = normalized

        assert last_result is not None
        return last_result

    @staticmethod
    def _compose_question(step_question: str, dependency_context: str) -> str:
        """将前置步骤的关键结果作为上下文注入当前步骤的问题中。"""
        if not dependency_context:
            return step_question
        return (
            f"{step_question}\n\n"
            f"前置步骤关键结果如下，请在本次查询中延续这些上下文：\n"
            f"{dependency_context}"
        )

    def _build_dependency_context(
        self,
        dependency_ids: list[str],
        results_by_step: dict[str, StepExecutionResult],
    ) -> str:
        """从已完成步骤中提取结果，序列化为 JSON 字符串，作为后续步骤的上下文。"""
        if not dependency_ids:
            return ""

        lines: list[str] = []
        for step_id in dependency_ids:
            result = results_by_step[step_id]
            rows = result.rows or self.get_intermediate_result(result.result_reference)
            payload = {
                "step_id": result.step_id,
                "step_name": result.step_name,
                "columns": result.columns,
                "rows": rows,
                "result_reference": result.result_reference,
            }
            lines.append(json.dumps(payload, ensure_ascii=False, default=str))
        return "\n".join(lines)

    @staticmethod
    def _has_failed_dependency(
        dependency_ids: list[str],
        results_by_step: dict[str, StepExecutionResult],
    ) -> bool:
        """检查当前步骤所依赖的前置步骤是否有失败的。"""
        return any(not results_by_step[step_id].success for step_id in dependency_ids)

    @staticmethod
    def _normalize_result(
        step: PlanStep,
        question: str,
        context_used: str,
        raw_result: dict[str, Any],
        attempts: int = 1,
        storage_backend: str | None = None,
    ) -> StepExecutionResult:
        """将 ChatBISystem.run() 返回的原始 dict 标准化为 StepExecutionResult。"""
        columns = raw_result.get("columns", [])
        rows = raw_result.get("results") or raw_result.get("rows") or []

        if rows and columns and isinstance(rows[0], tuple):
            normalized_rows = [dict(zip(columns, row)) for row in rows]
        else:
            normalized_rows = rows

        return StepExecutionResult(
            step_id=step.step_id,
            task_id=step.task_id,
            step_name=step.step_name,
            success=raw_result.get("success", False),
            status="completed" if raw_result.get("success", False) else "failed",
            attempts=attempts,
            question=question,
            depends_on=step.depends_on,
            context_used=context_used,
            storage_backend=storage_backend,
            sql=raw_result.get("sql"),
            columns=columns,
            rows=normalized_rows,
            formatted=raw_result.get("formatted", ""),
            error=raw_result.get("error"),
        )

    def _store_intermediate_result(
        self,
        step_id: str,
        columns: list[str],
        rows: list[dict[str, Any]],
    ) -> str:
        """将当前步骤执行结果存入 result_store，返回引用地址供后续步骤读取。"""
        return self.result_store.put(step_id=step_id, columns=columns, rows=rows)

    def get_intermediate_result(
        self,
        reference: str | None,
    ) -> list[dict[str, Any]]:
        if not reference:
            return []
        return self.result_store.get(reference)

    def cleanup(self) -> None:
        self.result_store.cleanup()

    @staticmethod
    def _build_result_store(
        storage_backend: Literal["memory", "temp_table"],
        connection_factory: Callable[[], Any] | None = None,
    ) -> IntermediateResultStore:
        if storage_backend == "temp_table":
            return TempTableResultStore(connection_factory=connection_factory)
        return MemoryResultStore()

    @staticmethod
    def _build_skipped_result(step: PlanStep, error: str) -> StepExecutionResult:
        """构建一个跳过的步骤执行结果，记录错误信息。"""
        return StepExecutionResult(
            step_id=step.step_id,
            task_id=step.task_id,
            step_name=step.step_name,
            success=False,
            status="skipped",
            question=step.question,
            depends_on=step.depends_on,
            error=error,
        )