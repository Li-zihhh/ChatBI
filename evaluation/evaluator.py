"""SQL 生成评估器。

职责: 串联 prompt → LLM → 对比指标 → 错误分析 → 报告输出。

通过判定:
  - 有数据库 (execute_fn != None): execution_match 为 True
  - 无数据库: exact_match 为 True (结构分只作为参考)

并行执行:
  - 通过 max_workers 控制并发线程数,设为 1 时退化为串行
  - DB 执行受线程锁保护,避免多线程共用同一连接
"""

from __future__ import annotations

import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from error_analyzer import ErrorAnalyzer

from . import metrics


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

    def evaluate(self, cases: List[dict]) -> Dict[str, Any]:
        total = len(cases)

        if self.max_workers <= 1:
            # 单线程 (workers=1): 保持与预期完全一致的执行顺序
            results = [self._evaluate_one(case) for case in cases]
        else:
            # 多线程: 用 as_completed 先到先收, 用下标数组保证最终顺序
            results = [None] * total
            done_count = 0
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_idx = {
                    executor.submit(self._evaluate_one, case): i
                    for i, case in enumerate(cases)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    results[idx] = future.result()
                    done_count += 1
                    if self.progress_callback:
                        self.progress_callback(done_count, total)

        # 兜底: 确保每个未通过的用例至少有一个错误标记 (必须在 _summarize 之前)
        for r in results:
            if not r["passed"] and not r.get("error"):
                r["error"] = {
                    "error_type": "unknown",
                    "root_cause": "错误分析未运行(可能因 LLM 超时或异常中断)",
                    "suggestion": "",
                }

        summary = self._summarize(results)

        return {"cases": results, "summary": summary}

    def _evaluate_one(self, case: dict) -> dict:
        """单个用例的评估流程: 生成 → 对比 → (可选) 执行 → (可选) 错误分析。"""
        question = case["question"]
        golden = case.get("golden_sql", "")
        entry: Dict[str, Any] = {
            "id": case.get("id"),
            "question": question,
            "golden_sql": golden,
            "generated_sql": None,
            "metrics": {},
            "passed": False,
            "error": None,
        }

        # 1. 调用 LLM (或 Mock) 生成 SQL
        try:
            generated = self.generate_llm(question)
        except Exception as exc:
            entry["metrics"] = {"exact_match": False, "structure_score": 0.0}
            entry["error"] = {
                "error_type": "generation_error",
                "root_cause": str(exc),
                "suggestion": "检查 API 连通性与 timeout 配置",
                "message": str(exc),
            }
            return entry

        entry["generated_sql"] = generated

        # 2. 结构对比 (始终执行,不依赖数据库)
        m = metrics.compare(golden, generated)
        entry["metrics"] = m

        # 3. 执行结果对比 (仅 --db 模式; DB 连接非线程安全,加锁串行化)
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
            entry["passed"] = bool(m["exact_match"])

        # 4. 未通过时尝试错误分析 (先走启发式规则,再走 LLM; 前者零成本)
        if not entry["passed"] and "error" not in entry:
            try:
                entry["error"] = self.error_analyzer.categorize_error(
                    generated, None, question
                )
            except Exception:
                entry["error"] = {
                    "error_type": "unknown",
                    "root_cause": "错误分析模块异常,可能是 LLM 超时或 JSON 解析失败",
                    "suggestion": "检查网络连通性,或增大 llm_client 的 timeout 参数",
                }

        return entry

    def _summarize(self, results: List[dict]) -> Dict[str, Any]:
        """汇总通过率、精确匹配率、平均结构分、错误类型分布等。"""
        # 通过 / 精确匹配 / 结构得分统计
        total = len(results)
        passed = sum(1 for r in results if r.get("passed"))
        exact = sum(1 for r in results if r["metrics"].get("exact_match"))
        scores = [r["metrics"].get("structure_score", 0.0) for r in results]
        avg_score = sum(scores) / total if total else 0.0

        exec_vals = [
            r["metrics"].get("execution_match")
            for r in results
            if "execution_match" in r["metrics"]
        ]
        exec_acc = sum(1 for v in exec_vals if v) / len(exec_vals) if exec_vals else None

        # 各组件级别的平均 (只统计数值型)
        comp_avgs: Dict[str, Optional[float]] = {}
        for key in ["select_jaccard", "condition_recall"]:
            vals = [
                r["metrics"].get(key)
                for r in results
                if isinstance(r["metrics"].get(key), (int, float))
            ]
            comp_avgs[key] = sum(vals) / len(vals) if vals else None

        # 错误类型分布 (按频次排序)
        err_dist = Counter()
        for r in results:
            err = r.get("error")
            if err:
                err_dist[err.get("error_type", "unknown")] += 1

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
            "error_distribution": dict(err_dist),
        }

    def render(self, result: Dict[str, Any]) -> str:
        """把评估结果渲染成可读的终端报告字符串。"""
        cases = result["cases"]
        s = result["summary"]
        lines = [
            "=" * 72,
            "SQL 生成评估报告",
            "=" * 72,
            f"总用例数: {s['total']}",
            f"通过数: {s['passed']}",
            f"通过率: {s['pass_rate'] * 100:.2f}%",
            f"精确匹配率: {s['exact_match_rate'] * 100:.2f}%",
            f"平均结构得分: {s['avg_structure_score']:.4f}",
            f"平均 SELECT 相似度: {s['avg_select_jaccard'] if s['avg_select_jaccard'] is not None else 'N/A'}",
            f"平均条件覆盖率: {s['avg_condition_recall'] if s['avg_condition_recall'] is not None else 'N/A'}",
            f"执行准确率: {s['execution_accuracy'] if s['execution_accuracy'] is not None else 'N/A (未启用数据库执行)'}",
        ]

        if s["error_distribution"]:
            lines.append("错误类型分布:")
            for err_type, count in sorted(
                s["error_distribution"].items(), key=lambda x: -x[1]
            ):
                lines.append(f"  - {err_type}: {count}")
        else:
            lines.append("错误类型分布: 无")

        lines.append("")
        lines.append("用例明细:")
        for r in cases:
            m = r["metrics"]
            err = r.get("error")
            err_type = err.get("error_type", "-") if err else "-"
            lines.append(
                f"[{r['id']}] 通过={'Y' if r['passed'] else 'N'} "
                f"精确={'Y' if m.get('exact_match') else 'N'} "
                f"结构={m.get('structure_score', 0.0):.2f} "
                f"错误={err_type}"
            )
            lines.append(f"    问题: {r['question']}")
            # 失败用例: 展开生成 SQL + 错误根因 + 修复建议
            if not r["passed"] and err:
                gen_sql = r.get("generated_sql") or "(生成失败)"
                lines.append(f"    生成: {gen_sql}")
                lines.append(f"    根因: {err.get('root_cause', '-')}")
                lines.append(f"    建议: {err.get('suggestion', '-')}")
                if err.get("layer") == "llm":
                    lines.append("    (以上分析由大模型生成)")
        return "\n".join(lines)