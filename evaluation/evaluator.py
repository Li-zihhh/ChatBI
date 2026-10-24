"""SQL 生成评估器。

职责：串联 prompt → LLM → 对比指标 → 执行校验 → 错误分析 → 报告输出。

相对早期版本的关键变化：
  1. **补充延迟与分组指标**：只测质量不测延迟，无法支撑 ChatBI 的落地决策；
     同时按 difficulty / category 分组，定位"哪一类问题在退化"。
  2. **支持拒答用例**：`expect == "refusal"` 的用例单独统计拒答准确率，
     这类用例（越界问题、危险请求）是线上最常见的失败来源。
  3. **错误分析带上结构化信号**：把 errno / error_type 一并交给 ErrorAnalyzer，
     避免它只能靠正则去猜 SQL 文本。

通过判定：
  - 有数据库（execute_fn != None）：execution_match 为 True
  - 无数据库：exact_match 为 True（结构分只作参考）
  - 拒答用例：生成结果为空或不涉及业务表即视为通过
"""

from __future__ import annotations

import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional

from text2SQL.error_analyzer import ErrorAnalyzer

from . import metrics

# 业务表白名单：用于判定"模型是否编造了一条业务查询"（拒答用例判定）
BUSINESS_TABLES = {
    "sales_orders",
    "dim_customers",
    "dim_products",
    "exchange_rates",
    "finance_expenses",
}


class Evaluator:
    def __init__(
        self,
        generate_llm: Callable[[str], str],
        execute_fn: Optional[Callable[[str], list]] = None,
        llm_client=None,
        max_workers: int = 4,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        self.generate_llm = generate_llm
        self.execute_fn = execute_fn
        self.error_analyzer = ErrorAnalyzer(llm_client=llm_client)
        self.max_workers = max_workers
        self.progress_callback = progress_callback
        self._db_lock = threading.Lock()

    # ==================== 批量执行 ====================
    def evaluate(self, cases: List[dict]) -> Dict[str, Any]:
        total = len(cases)

        if self.max_workers <= 1:
            results = [self._evaluate_one(case) for case in cases]
        else:
            results = [None] * total
            done_count = 0
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_idx = {
                    executor.submit(self._evaluate_one, case): index
                    for index, case in enumerate(cases)
                }
                for future in as_completed(future_to_idx):
                    index = future_to_idx[future]
                    results[index] = future.result()
                    done_count += 1
                    if self.progress_callback:
                        self.progress_callback(done_count, total)

        # 兜底：确保每个未通过的用例至少有一个错误标记
        for result in results:
            if not result["passed"] and not result.get("error"):
                result["error"] = {
                    "error_type": "unknown",
                    "root_cause": "错误分析未运行(可能因 LLM 超时或异常中断)",
                    "suggestion": "",
                }

        return {"cases": results, "summary": self._summarize(results)}

    # ==================== 单用例 ====================
    def _evaluate_one(self, case: dict) -> dict:
        question = case["question"]
        golden = case.get("golden_sql", "")
        expect = case.get("expect", "sql")

        entry: Dict[str, Any] = {
            "id": case.get("id"),
            "question": question,
            "golden_sql": golden,
            "difficulty": case.get("difficulty", "unknown"),
            "category": case.get("category", "unknown"),
            "expect": expect,
            "generated_sql": None,
            "metrics": {},
            "passed": False,
            "error": None,
            "latency_ms": None,
        }

        # 1) 生成
        started = perf_counter()
        try:
            generated = self.generate_llm(question)
        except Exception as exc:
            entry["latency_ms"] = round((perf_counter() - started) * 1000, 2)
            entry["metrics"] = {"exact_match": False, "structure_score": 0.0}
            if expect == "refusal":
                entry["passed"] = True  # 生成阶段直接失败，等同于拒答
                return entry
            entry["error"] = {
                "error_type": "generation_error",
                "root_cause": str(exc),
                "suggestion": "检查 API 连通性与 timeout 配置",
            }
            return entry

        entry["latency_ms"] = round((perf_counter() - started) * 1000, 2)
        entry["generated_sql"] = generated

        # 2) 拒答类用例
        if expect == "refusal":
            entry["passed"] = self._looks_like_refusal(generated)
            if not entry["passed"]:
                entry["error"] = {
                    "error_type": "should_refuse",
                    "root_cause": "问题超出数据范围或属于危险请求，但模型仍生成了业务查询",
                    "suggestion": "在 Prompt 中加入拒答约束，并在链路层拦截非白名单表",
                }
            return entry

        # 3) 结构对比（不依赖数据库）
        entry["metrics"] = metrics.compare(golden, generated)

        # 4) 执行结果对比（DB 连接非线程安全，加锁串行化）
        if self.execute_fn is not None:
            with self._db_lock:
                try:
                    gen_rows = self.execute_fn(generated)
                    gold_rows = self.execute_fn(golden)
                    exec_match = metrics.rows_equal(gold_rows, gen_rows)
                except Exception as exc:
                    exec_match = False
                    entry["error"] = self.error_analyzer.categorize_error(
                        generated, str(exc), question
                    )
            entry["metrics"]["execution_match"] = exec_match
            entry["passed"] = bool(exec_match)
        else:
            entry["passed"] = bool(entry["metrics"].get("exact_match"))

        # 5) 未通过时做错误归因（启发式优先，零成本）
        if not entry["passed"] and "error" not in entry:
            try:
                entry["error"] = self.error_analyzer.categorize_error(
                    generated, None, question
                )
            except Exception:
                entry["error"] = {
                    "error_type": "unknown",
                    "root_cause": "错误分析模块异常",
                    "suggestion": "检查网络连通性或增大 llm_client 的 timeout",
                }

        return entry

    @staticmethod
    def _looks_like_refusal(generated: str) -> bool:
        """判断模型是否真的"没有编造业务查询"。"""
        text = (generated or "").strip()
        if not text:
            return True
        lowered = text.lower()
        if "select" not in lowered:
            return True
        return not any(table in lowered for table in BUSINESS_TABLES)

    # ==================== 汇总 ====================
    def _summarize(self, results: List[dict]) -> Dict[str, Any]:
        total = len(results)
        passed = sum(1 for result in results if result.get("passed"))
        exact = sum(1 for result in results if result["metrics"].get("exact_match"))
        scores = [result["metrics"].get("structure_score", 0.0) for result in results]
        avg_score = sum(scores) / total if total else 0.0

        exec_values = [
            result["metrics"].get("execution_match")
            for result in results
            if "execution_match" in result["metrics"]
        ]
        exec_acc = (
            sum(1 for value in exec_values if value) / len(exec_values)
            if exec_values
            else None
        )

        comp_avgs: Dict[str, Optional[float]] = {}
        for key in ("select_jaccard", "condition_recall"):
            values = [
                result["metrics"].get(key)
                for result in results
                if isinstance(result["metrics"].get(key), (int, float))
            ]
            comp_avgs[key] = sum(values) / len(values) if values else None

        error_dist = Counter()
        for result in results:
            error = result.get("error")
            if error:
                error_dist[error.get("error_type", "unknown")] += 1

        latencies = sorted(
            result["latency_ms"]
            for result in results
            if isinstance(result.get("latency_ms"), (int, float))
        )

        return {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total if total else 0.0,
            "exact_match": exact,
            "exact_match_rate": exact / total if total else 0.0,
            "avg_structure_score": round(avg_score, 4),
            "execution_accuracy": round(exec_acc, 4) if exec_acc is not None else None,
            "avg_select_jaccard": round(comp_avgs["select_jaccard"], 4)
            if comp_avgs["select_jaccard"] is not None
            else None,
            "avg_condition_recall": round(comp_avgs["condition_recall"], 4)
            if comp_avgs["condition_recall"] is not None
            else None,
            "latency_p50_ms": self._percentile(latencies, 50),
            "latency_p95_ms": self._percentile(latencies, 95),
            "error_distribution": dict(error_dist),
            "by_difficulty": self._group_stats(results, "difficulty"),
            "by_category": self._group_stats(results, "category"),
            "refusal_accuracy": self._refusal_accuracy(results),
        }

    @staticmethod
    def _percentile(values: List[float], pct: float) -> Optional[float]:
        if not values:
            return None
        if len(values) == 1:
            return round(values[0], 2)
        index = min(len(values) - 1, max(0, int(round(pct / 100 * (len(values) - 1)))))
        return round(values[index], 2)

    @staticmethod
    def _group_stats(results: List[dict], field: str) -> Dict[str, Dict[str, float]]:
        buckets: dict[str, list[bool]] = defaultdict(list)
        for result in results:
            buckets[str(result.get(field, "unknown"))].append(bool(result.get("passed")))
        return {
            key: {
                "total": len(values),
                "passed": sum(1 for value in values if value),
                "pass_rate": round(sum(1 for value in values if value) / len(values), 4),
            }
            for key, values in sorted(buckets.items())
        }

    @staticmethod
    def _refusal_accuracy(results: List[dict]) -> Optional[float]:
        refusal_cases = [result for result in results if result.get("expect") == "refusal"]
        if not refusal_cases:
            return None
        passed = sum(1 for result in refusal_cases if result.get("passed"))
        return round(passed / len(refusal_cases), 4)

    # ==================== 渲染 ====================
    def render(self, result: Dict[str, Any]) -> str:
        cases = result["cases"]
        summary = result["summary"]
        lines = [
            "=" * 72,
            "SQL 生成评估报告",
            "=" * 72,
            f"总用例数: {summary['total']}",
            f"通过数: {summary['passed']}",
            f"通过率: {summary['pass_rate'] * 100:.2f}%",
            f"精确匹配率: {summary['exact_match_rate'] * 100:.2f}%",
            f"平均结构得分: {summary['avg_structure_score']:.4f}",
            f"平均 SELECT 相似度: {self._fmt(summary['avg_select_jaccard'])}",
            f"平均条件覆盖率: {self._fmt(summary['avg_condition_recall'])}",
            f"执行准确率: {self._fmt(summary['execution_accuracy'], 'N/A (未启用数据库执行)')}",
            f"拒答准确率: {self._fmt(summary['refusal_accuracy'], 'N/A (无用例)')}",
            f"生成延迟 p50/p95: {self._fmt_latency(summary['latency_p50_ms'], summary['latency_p95_ms'])}",
        ]

        if summary["by_difficulty"]:
            lines.append("按难度:")
            for key, stats in summary["by_difficulty"].items():
                lines.append(
                    f"  - {key}: {stats['passed']}/{stats['total']} "
                    f"({stats['pass_rate'] * 100:.1f}%)"
                )

        if summary["by_category"]:
            lines.append("按业务类别:")
            for key, stats in summary["by_category"].items():
                lines.append(
                    f"  - {key}: {stats['passed']}/{stats['total']} "
                    f"({stats['pass_rate'] * 100:.1f}%)"
                )

        if summary["error_distribution"]:
            lines.append("错误类型分布:")
            for err_type, count in sorted(
                summary["error_distribution"].items(), key=lambda item: -item[1]
            ):
                lines.append(f"  - {err_type}: {count}")
        else:
            lines.append("错误类型分布: 无")

        lines.append("")
        lines.append("用例明细:")
        for result_item in cases:
            metric = result_item["metrics"]
            error = result_item.get("error")
            err_type = error.get("error_type", "-") if error else "-"
            lines.append(
                f"[{result_item['id']}] 通过={'Y' if result_item['passed'] else 'N'} "
                f"精确={'Y' if metric.get('exact_match') else 'N'} "
                f"结构={metric.get('structure_score', 0.0):.2f} "
                f"错误={err_type}"
            )
            lines.append(f"    问题: {result_item['question']}")
            if not result_item["passed"] and error:
                lines.append(f"    生成: {result_item.get('generated_sql') or '(生成失败)'}")
                lines.append(f"    根因: {error.get('root_cause', '-')}")
                lines.append(f"    建议: {error.get('suggestion', '-')}")
                if error.get("layer") == "llm":
                    lines.append("    (以上分析由大模型生成)")
        return "\n".join(lines)

    @staticmethod
    def _fmt_span(value) -> str:
        """区间统计（如 p50/p95）的格式化。"""
        if value is None:
            return "N/A"
        return f"{value:.1f}"

    @classmethod
    def _fmt_latency(cls, p50, p95) -> str:
        if p50 is None:
            return "N/A"
        return f"{cls._fmt_span(p50)} / {cls._fmt_span(p95)} ms"

    @staticmethod
    def _fmt(value, fallback: str = "N/A") -> str:
        if value is None:
            return fallback
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)
